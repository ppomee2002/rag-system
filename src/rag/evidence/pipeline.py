"""MODE B 공통 파이프라인.

고정 검색 결과 -> grading -> 판정 -> refinement -> correction -> generation.

C1 (grader x evidence 정책), C2 (문서별 grading 상태), C3 (budget) 이 같은 경로를 쓴다.

stage 이름: rag.pipelines.crag.STAGES (retrieval / grading / rewrite / web_search / generation)
+ "refinement" (공식 knowledge refinement, evaluator 계산). 기존 STAGES 는 바꾸지 않고 여기서만
추가한다 (STAGES_OFFICIAL). 실행하지 않은 stage 는 0 이 아니라 None 으로 기록한다.

LangGraph 를 쓰지 않는 이유: 이 경로는 분기 하나 (correction 여부) 뿐인 순차 흐름이고, C2/C3 가
문서 단위 루프 안에 hook (stop rule, snapshot) 을 넣어야 해서 일반 함수가 더 투명하다.
기존 rag.graph.CRAGGraph 는 비교군으로 그대로 남는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from rag.crag_official import generation as offgen
from rag.crag_official import source
from rag.crag_official.refinement import (
    ambiguous_knowledge,
    external_knowledge,
    internal_knowledge,
)
from rag.eval.metrics import score_all
from rag.evidence.budget import BudgetPolicy
from rag.evidence.frozen import FrozenItem
from rag.evidence.graders import DocumentGrader, LlamaPromptGrader, MemoStripScorer, decide
from rag.evidence.policies import EvidencePolicySpec, check_combination, policy_id
from rag.evidence.state import GradingTracker
from rag.pipelines import prompts
from rag.pipelines.crag import STAGES
from rag.types import Document
from rag.utils.timing import StageTimer, stopwatch

STAGES_OFFICIAL = (*STAGES, "refinement")
QUERY_MODES = ("legacy_rewrite", "official_keywords")
PROMPT_MODES = ("legacy", "official")


@dataclass
class EvidenceUnit:
    id: str
    text: str
    kind: str  # retrieved | retrieved_strip | web | web_strip
    origin: str | None  # 원 문서 id (strip 이면 출처 문서)
    score: float | None = None

    def to_dict(self, with_text: bool) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "score": self.score,
            "chars": len(self.text),
            "text": self.text if with_text else None,
        }


@dataclass
class PipelineConfig:
    web_top_k: int = 5
    query_mode: str = "legacy_rewrite"  # legacy_rewrite | official_keywords
    wiki_prefix: bool = False  # official_keywords 일 때 "Wikipedia, " 접두 (공식 mode=wiki)
    on_web_error: str = "fail"  # fail (오류 행) | continue (web 없이 진행, correction_valid=False)
    prompt: str = "legacy"  # legacy (rag.pipelines.prompts.GENERATE) | official (공식 PopQA)
    generation_max_tokens: int | None = None
    decompose_mode: str = source.DEFAULT_DECOMPOSE_MODE
    record_texts: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"correction.query_mode: {QUERY_MODES}")
        if self.prompt not in PROMPT_MODES:
            raise ValueError(f"generation.prompt: {PROMPT_MODES}")
        if self.on_web_error not in ("fail", "continue"):
            raise ValueError("correction.on_web_error: fail | continue")


class WebSearchFailed(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ControlledCRAG:
    def __init__(
        self,
        policy: EvidencePolicySpec,
        grader: DocumentGrader | None,
        budget: BudgetPolicy,
        web_search=None,
        cfg: PipelineConfig | None = None,
    ):
        if policy.grading and grader is None:
            raise ValueError(f"{policy.name} 정책에는 grader 가 필요")
        self.policy = policy
        self.grader = grader if policy.grading else None
        self.budget = budget
        self.web_search = web_search
        self.cfg = cfg or PipelineConfig()
        gtype = self.grader.grader_type if self.grader else None
        self.combination = check_combination(policy, gtype, budget)
        self.policy_id = policy_id(policy.name, gtype, budget.name)

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "policy_id": self.policy_id,
            "grader": self.grader.describe() if self.grader else None,
            "budget": self.budget.describe(),
            "combination": self.combination,
            "pipeline": self.cfg.__dict__,
            "stages": STAGES_OFFICIAL,
        }

    # ------------------------------------------------------------ steps
    def _grade(self, item: FrozenItem, llm, tracker: GradingTracker, qs) -> tuple[str, dict]:
        stop_info = {"stopped_early": False, "stop_rule": self.budget.stop.name}
        for rank, doc in enumerate(item.docs):
            j = self.grader.judge(item.question, doc, rank, llm=llm)
            snap = tracker.add(j)
            if self.budget.stop.should_stop(snap):
                stop_info.update(stopped_early=rank < len(item.docs) - 1, stopped_after=rank + 1)
                break
        action = decide(tracker.judgments)
        last = tracker.last
        stop_info["action_final"] = last.action_final if last else True
        if qs is not None and hasattr(qs, "on_graded"):
            qs.on_graded(_GradeView(action, len(tracker.judgments)))
        return action, stop_info

    def _correction_query(self, question: str, llm) -> str:
        if self.cfg.query_mode == "legacy_rewrite":
            q = llm.generate(prompts.REWRITE.format(question=question), kind="rewrite")
        else:
            prompt = source.POPQA_KEYWORD_PROMPT.replace("{question}", question)
            q = llm.generate(prompt, kind="rewrite")
            q = q.strip().splitlines()[0] if q.strip() else ""
            if self.cfg.wiki_prefix:
                q = source.WIKI_QUERY_PREFIX + q
        return q.strip()

    def _search(self, web, query: str) -> tuple[list[Document], dict]:
        info: dict[str, Any] = {"query": query}
        try:
            docs = web.search(query, self.cfg.web_top_k)
        except KeyError as e:  # replay miss (on_miss=error)
            info.update(status="replay_miss", error=str(e)[:300])
            docs = None
        except Exception as e:  # noqa: BLE001 - API 오류 / 키 없음 등
            info.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
            docs = None
        last = getattr(web, "last", None)
        if last:
            info["replay"] = dict(last)
        if docs is None:
            if self.cfg.on_web_error == "fail":
                raise WebSearchFailed(f"web search {info['status']}: {info.get('error')}")
            return [], info
        if last and last.get("source") == "empty":
            info["status"] = "replay_miss_empty"
        else:
            info["status"] = "ok" if docs else "empty"
        info["n_docs"] = len(docs)
        return docs, info

    def _strip_scorer(self, llm, tracker: GradingTracker, item: FrozenItem):
        if isinstance(self.grader, LlamaPromptGrader):
            memo = {item.docs[j.rank].text.strip(): j.score for j in tracker.judgments}
            return MemoStripScorer(self.grader, memo, llm)
        return self.grader  # official_t5: evaluator.score_strips (truncation=True)

    # ------------------------------------------------------------ run
    def run(self, item: FrozenItem, llm, grader_llm=None, query_state=None, web=None) -> dict:
        """한 질문 실행 -> 결과 dict (rag.evidence.schema 의 필드). 예외는 호출자가 기록한다."""
        timer = StageTimer()
        qs = query_state
        web = web or self.web_search
        grader_llm = grader_llm or llm
        docs = item.docs
        tracker = GradingTracker(item.question_id, len(docs))
        out: dict[str, Any] = {}
        with stopwatch() as sw:
            _enter(qs, "retrieval")
            with timer.stage("retrieval"):  # 고정 입력: 검색 대신 고정 Top-k 사용
                retrieved = list(docs)
            if qs is not None and hasattr(qs, "on_retrieved"):
                qs.on_retrieved(retrieved)

            action, stop_info = None, {}
            if self.policy.grading:
                _enter(qs, "grading")
                with timer.stage("grading"):
                    action, stop_info = self._grade(item, grader_llm, tracker, qs)

            # ---- 내부 evidence
            hit = tracker.last.first_pass_rank if tracker.last else None
            internal: list[EvidenceUnit] = []
            internal_text = None
            refinement = None
            retained = self.budget.retention.select(retrieved, hit, action) if action else None
            mode = "all" if action is None else self.policy.internal_mode(action)
            if retained is not None:
                mode = f"budget:{self.budget.retention.name}"
                internal = [_unit(d, "retrieved") for d in retained]
            elif mode == "refine":
                with timer.stage("refinement"):
                    refinement = internal_knowledge(
                        item.question,
                        [d.text for d in retrieved],
                        self._strip_scorer(grader_llm, tracker, item),
                        self.cfg.decompose_mode,
                        origins=[d.id for d in retrieved],
                    )
                internal_text = refinement.text
                internal = [
                    EvidenceUnit(
                        f"{s.origin}#s{s.index}", s.text, "retrieved_strip", s.origin, s.score
                    )
                    for s in refinement.selected
                ]
            elif mode != "none":
                internal = [
                    _unit(d, "retrieved")
                    for d, keep in zip(
                        retrieved, self._keep_mask(mode, tracker, len(retrieved)), strict=True
                    )
                    if keep
                ]

            # ---- correction (web)
            correction: dict[str, Any] = {"performed": False}
            web_units: list[EvidenceUnit] = []
            external_text = None
            external_ref = None
            if action is not None and self.policy.uses_web(action):
                if web is None:
                    raise WebSearchFailed("web_search 가 설정되지 않음")
                _enter(qs, "rewrite")
                with timer.stage("rewrite"):
                    query = self._correction_query(item.question, llm)
                _enter(qs, "web_search")
                with timer.stage("web_search"):
                    web_docs, winfo = self._search(web, query)
                correction = {
                    "performed": True,
                    "query_mode": self.cfg.query_mode,
                    "valid": winfo["status"] == "ok",
                    **winfo,
                    "web_doc_ids": [d.id for d in web_docs],
                }
                if self.policy.web_form == "refine":
                    with timer.stage("refinement"):
                        external_ref = external_knowledge(
                            item.question, web_docs, self._strip_scorer(grader_llm, tracker, item)
                        )
                    external_text = external_ref.text
                    if external_ref.selected:
                        web_units = [
                            EvidenceUnit(
                                f"{s.origin}#p{s.index}", s.text, "web_strip", s.origin, s.score
                            )
                            for s in external_ref.selected
                        ]
                    elif external_text:
                        web_units = [EvidenceUnit("web#snippets", external_text, "web", None)]
                else:
                    web_units = [_unit(d, "web") for d in web_docs]

            # ---- fallback ablation: Incorrect 에서 web evidence 가 비면 internal refine 사용
            fallback = None
            if (
                action == "incorrect"
                and self.policy.incorrect_fallback_on_empty_web
                and not (external_text or "").strip()
                and not web_units
            ):
                with timer.stage("refinement"):
                    refinement = internal_knowledge(
                        item.question,
                        [d.text for d in retrieved],
                        self._strip_scorer(grader_llm, tracker, item),
                        self.cfg.decompose_mode,
                        origins=[d.id for d in retrieved],
                    )
                internal_text = refinement.text
                internal = [
                    EvidenceUnit(
                        f"{s.origin}#s{s.index}", s.text, "retrieved_strip", s.origin, s.score
                    )
                    for s in refinement.selected
                ]
                external_text = None
                mode = "refine (fallback: web evidence 없음)"
                fallback = {"applied": True, "reason": correction.get("status") or "no_web"}

            units = internal + web_units
            # ---- generation
            paragraph = self._official_paragraph(action, internal_text, external_text, units)
            if self.cfg.prompt == "official":
                prompt = offgen.format_popqa_prompt(item.question, paragraph)
            else:
                prompt = prompts.GENERATE.format(
                    context=prompts.format_context([Document(u.id, u.text) for u in units]),
                    question=item.question,
                )
            _enter(qs, "generation")
            gen_kwargs: dict[str, Any] = {"kind": "generate"}
            if self.cfg.generation_max_tokens is not None:
                gen_kwargs["max_tokens"] = self.cfg.generation_max_tokens
            with timer.stage("generation"):
                raw_answer = llm.generate(prompt, **gen_kwargs)
            answer = (
                offgen.postprocess_answer(raw_answer)
                if self.cfg.prompt == "official"
                else raw_answer
            ).strip()

        # ---- record
        retrieved_ids = [d.id for d in retrieved]
        kept = _ordered_unique([u.origin if u.kind != "retrieved" else u.id for u in internal])
        evidence_text = "\n\n".join(u.text for u in units)
        stage_timings = {
            s: (timer.totals[s] if s in timer.totals else None) for s in STAGES_OFFICIAL
        }
        compute = [j.compute_seconds for j in tracker.judgments]
        out.update(
            grader_type=self.grader.grader_type if self.grader else None,
            score_scale=self.grader.score_scale if self.grader else None,
            thresholds=(
                {"upper": self.grader.upper, "lower": self.grader.lower} if self.grader else None
            ),
            evidence_policy=self.policy.name,
            policy_id=self.policy_id,
            budget=self.budget.describe(),
            official_policy_semantics=self.combination["official_policy_semantics"],
            correction_path=action or "none",
            decision={
                "action": action,
                "internal_mode": mode,
                "reason": _reason(action, tracker),
                **stop_info,
            },
            retrieval_doc_ids=retrieved_ids,
            grading_events=tracker.events(grader_type=out.get("grader_type")),
            grading_snapshots=[s.to_dict() for s in tracker.snapshots],
            refinement_events=[
                {"target": "internal", **refinement.to_dict(self.cfg.record_texts)}
                if refinement
                else None,
                {"target": "external", **external_ref.to_dict(self.cfg.record_texts)}
                if external_ref
                else None,
            ],
            correction=correction,
            fallback=fallback,
            evidence={
                "kept_retrieved_ids": kept,
                "removed_retrieved_ids": [i for i in retrieved_ids if i not in kept],
                "added_ids": [u.id for u in web_units],
                "selected_evidence_ids": [u.id for u in units],
                "units": [u.to_dict(self.cfg.record_texts) for u in units],
                "n_units": len(units),
                "n_chars": len(evidence_text),
                "paragraph_chars": len(paragraph),
                "sha16": _sha(paragraph),
                "internal_text": internal_text if self.cfg.record_texts else None,
                "external_text": external_text if self.cfg.record_texts else None,
            },
            generation_prompt_chars=len(prompt),
            generation_prompt=prompt if self.cfg.record_texts else None,
            stage_timings=stage_timings,
            stage_counts=dict(timer.counts),
            evaluator_compute_seconds=(
                None if not compute or any(c is None for c in compute) else sum(compute)
            ),
            e2e_latency=sw["elapsed"],
            prediction=answer,
            raw_prediction=raw_answer if raw_answer != answer else None,
            ground_truth=item.answers,
        )
        scores = score_all(answer, item.answers)
        out.update(
            is_correct=bool(scores["acc"]),
            metrics={**scores, "official_match": offgen.official_match(answer, item.answers)},
        )
        if qs is not None and hasattr(qs, "finish"):
            qs.finish(action)
        return out

    def _keep_mask(self, mode: str, tracker: GradingTracker, n: int) -> list[bool]:
        if mode == "all":
            return [True] * n
        cut = self.grader.upper if mode == "filter_upper" else self.grader.lower
        by_rank = {j.rank: j.score for j in tracker.judgments}
        return [r in by_rank and by_rank[r] >= cut for r in range(n)]

    def _official_paragraph(self, action, internal_text, external_text, units) -> str:
        """공식 prompt 의 Documents 문단. official 정책은 공식 지식 문자열, 그 외는 공식 rag 처럼
        문서를 ' [sep] ' 로 결합."""
        if self.policy.name == "official" and action is not None:
            if action == "correct":
                return internal_text or ""
            if action == "incorrect":
                return external_text or ""
            return ambiguous_knowledge(internal_text or "", external_text or "")
        if self.policy.official_structure and action is not None:
            # fallback ablation: 있는 지식만으로 공식과 같은 형식 (둘 다 있으면 결합 템플릿)
            if internal_text is not None and external_text is not None:
                return ambiguous_knowledge(internal_text, external_text)
            if internal_text is not None:
                return internal_text
            return external_text or ""
        return source.RAG_PASSAGE_JOIN.join(u.text for u in units)


@dataclass
class _GradeView:
    """QueryState.on_graded 가 읽는 필드만 (action, n_graded)."""

    action: str
    n_graded: int


def _enter(qs, stage: str) -> None:
    if qs is not None and hasattr(qs, "enter_stage"):
        qs.enter_stage(stage)


def _unit(d: Document, kind: str) -> EvidenceUnit:
    return EvidenceUnit(d.id, d.text, kind, d.id if kind == "retrieved" else None, d.score)


def _ordered_unique(xs):
    out = []
    for x in xs:
        if x is not None and x not in out:
            out.append(x)
    return out


def _reason(action: str | None, tracker: GradingTracker) -> str:
    if action is None:
        return "no_grading (vanilla)"
    s = tracker.last
    if action == "correct":
        return f"flag2 (score>=upper) at rank {s.first_pass_rank}"
    if action == "ambiguous":
        return f"no flag2, {s.n_partial} doc(s) with lower<=score<upper"
    return f"all {s.n_evaluated} evaluated doc(s) score<lower"
