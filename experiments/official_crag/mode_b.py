"""MODE B — Controlled Server Comparison runner.

고정 Top-5 (frozen inputs) -> grader (공식 T5 기본 / Llama legacy) -> evidence 정책
-> vLLM generator.
결과: results/official_crag/<purpose>/<exp_id>/<run_id>/
  config.yaml, run_meta.json   설정, git commit, 패키지, GPU, 엔진·evaluator·가중치 정보, 입력 hash
  queries.jsonl                질문 단위 (rag.evidence.schema)
  calls.jsonl                  LLM 호출 단위 (CallRecord)
  summary.json                 조건 요약 (정확도, 판정 분포, evidence, latency)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag.crag_official import source as off_source
from rag.crag_official.completion_engine import build_generator_engine
from rag.crag_official.evaluator import build_evaluator, weights_fingerprint
from rag.evidence.budget import build_budget
from rag.evidence.doc_log import doc_rows, t5_doc_token_counter
from rag.evidence.doc_log import write_csv as write_doc_csv
from rag.evidence.frozen import file_sha256, load_frozen
from rag.evidence.graders import LlamaPromptGrader, OfficialT5Grader
from rag.evidence.manifest import build_manifest
from rag.evidence.pipeline import ControlledCRAG, PipelineConfig
from rag.evidence.policies import get_policy
from rag.evidence.schema import SCHEMA_VERSION, validate_record
from rag.llm.engine_llm import EngineLLM
from rag.scheduling import QueryState, SchedulingGate, build_policy
from rag.telemetry.calls import CallRecorder
from rag.telemetry.run_meta import (
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    git_info,
    write_new,
    write_run_files,
)

PURPOSE_PREFIX = {"smoke": "SMOKE-", "pre": "PRE-", "paper": "C1-"}


class PurposeError(ValueError):
    pass


def check_purpose(cfg: dict, paper_approved: bool, git: dict | None = None) -> dict:
    run = cfg.get("run") or {}
    purpose = run.get("purpose", "smoke")
    exp_id = cfg.get("exp_id", "")
    if purpose not in PURPOSE_PREFIX:
        raise PurposeError(f"run.purpose: {tuple(PURPOSE_PREFIX)}")
    if not exp_id.startswith(PURPOSE_PREFIX[purpose]):
        raise PurposeError(
            f"purpose={purpose} 의 exp_id 는 {PURPOSE_PREFIX[purpose]}* (got {exp_id})"
        )
    if purpose == "paper":
        problems = []
        if not paper_approved:
            problems.append("--paper-approved 필요")
        git = git if git is not None else git_info()
        if git.get("dirty") is not False:
            problems.append("git 작업트리가 clean 하지 않음")
        if (cfg.get("inputs") or {}).get("limit"):
            problems.append("inputs.limit 로 문항을 줄인 run")
        if problems:
            raise PurposeError("본실험 조건 미충족: " + "; ".join(problems))
    return {"purpose": purpose, "exp_id": exp_id}


def build_web_search(cfg: dict, run_dir: Path):
    """crag_system runner 와 같은 규칙 (rag.retrieval.web_cache 재사용)."""
    from rag.retrieval.web_cache import RecordingWebSearch, ReplayWebSearch, load_replay_index

    w = cfg.get("web_search") or {}
    provider = w.get("provider", "replay")
    query_mode = (cfg.get("correction") or {}).get("query_mode", "legacy_rewrite")

    def tavily():
        from rag.retrieval.tavily_search import TavilyWebSearch

        return TavilyWebSearch(
            search_depth=w.get("search_depth", "basic"),
            include_raw_content=bool(w.get("include_raw_content", False)),
        )

    def recorder(ws):
        return RecordingWebSearch(
            ws, run_dir / "web_cache.jsonl", "tavily", ws.params(), query_mode
        )

    if provider == "none":
        return None
    if provider == "tavily":
        ws = tavily()
        return recorder(ws) if w.get("record") else ws
    if provider == "replay":
        paths = [Path(p) for p in w.get("replay_paths") or []]
        runs = w.get("replay_from_runs") or []
        paths += [Path(r) / "web_cache.jsonl" for r in runs]
        if not paths:
            raise ValueError(
                "web_search.provider=replay 에는 replay_from_runs 또는 replay_paths 필요"
            )
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"web replay 기록 없음: {missing}")
        key_mode = w.get("replay_key", "query_id")
        index = load_replay_index(runs) if key_mode == "query_id" else None
        if key_mode == "query_id" and not runs:
            raise ValueError("replay_key=query_id 에는 replay_from_runs 필요")
        on_miss = w.get("replay_on_miss", "error")
        fallback = None
        if on_miss == "live":  # miss 면 실제 검색 + 이 run 의 web_cache.jsonl 에 v2 기록
            fallback, on_miss = recorder(tavily()), "error"
        expect = None
        if key_mode == "query_strict":
            from rag.retrieval.tavily_search import tavily_params

            expect = {
                "provider": w.get("replay_provider", "tavily"),
                "params": tavily_params(
                    w.get("search_depth", "basic"), w.get("include_raw_content", False)
                ),
                "query_mode": query_mode,
            }
        return ReplayWebSearch(
            paths,
            w.get("latency_mode", "recorded"),
            fallback=fallback,
            key_mode=key_mode,
            on_miss=on_miss,
            query_index=index,
            expect=expect,
            accept_legacy=bool(w.get("accept_legacy_records", False)),
        )
    raise ValueError(f"web_search.provider: replay | tavily | none (got {provider})")


@dataclass
class Components:
    engine: Any
    web_search: Any = None
    evaluator: Any = None
    extra_closers: list = field(default_factory=list)


def needs_evaluator(cfg: dict) -> bool:
    gtype = (cfg.get("grader") or {}).get("type", "official_t5")
    return cfg.get("evidence_policy") != "vanilla" and gtype == "official_t5"


def build_pipeline(cfg: dict, comp: Components) -> ControlledCRAG:
    policy = get_policy(cfg.get("evidence_policy", "official"))
    budget = build_budget(cfg.get("budget"))
    gcfg = cfg.get("grader") or {}
    grader = None
    if policy.grading:
        if gcfg.get("type", "official_t5") == "official_t5":
            if comp.evaluator is None:
                raise ValueError("official_t5 grader 에 evaluator 가 없음")
            grader = OfficialT5Grader(comp.evaluator)
        elif gcfg["type"] == "llama_prompt":
            grader = LlamaPromptGrader(
                gcfg.get("upper", 0.7), gcfg.get("lower", 0.3), bool(gcfg.get("record_texts"))
            )
        else:
            raise ValueError(f"grader.type: official_t5 | llama_prompt (got {gcfg.get('type')})")
    corr = cfg.get("correction") or {}
    gen = cfg.get("generation") or {}
    pcfg = PipelineConfig(
        web_top_k=(cfg.get("web_search") or {}).get("top_k", 5),
        query_mode=corr.get("query_mode", "legacy_rewrite"),
        wiki_prefix=bool(corr.get("wiki_prefix", False)),
        on_web_error=corr.get("on_web_error", "fail"),
        prompt=gen.get("prompt", "legacy"),
        generation_max_tokens=gen.get("max_tokens"),
        decompose_mode=(cfg.get("refinement") or {}).get("decompose_mode", "selection"),
        record_texts=bool((cfg.get("run") or {}).get("record_texts")),
    )
    return ControlledCRAG(policy, grader, budget, comp.web_search, pcfg)


def _pct(x) -> dict:
    if not len(x):
        return {"n": 0}
    a = np.asarray(x, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def summarize(rows: list[dict]) -> dict:
    meas = [r for r in rows if not r.get("warmup")]
    ok = [r for r in meas if not r.get("error")]
    out: dict[str, Any] = {
        "n_measured": len(meas),
        "n_ok": len(ok),
        "n_errors": len(meas) - len(ok),
        "error_examples": [r["error"] for r in meas if r.get("error")][:10],
    }
    if not ok:
        return out
    stage_names = sorted({k for r in ok for k in r["stage_timings"]})
    out.update(
        policy_id=ok[0]["policy_id"],
        accuracy=float(np.mean([r["metrics"]["acc"] for r in ok])),
        official_match=float(np.mean([r["metrics"]["official_match"] for r in ok])),
        f1=float(np.mean([r["metrics"]["f1"] for r in ok])),
        action_counts={
            a: sum(r["correction_path"] == a for r in ok)
            for a in ("correct", "ambiguous", "incorrect", "none")
        },
        correction_invalid=sum(
            1 for r in ok if r["correction"].get("performed") and not r["correction"].get("valid")
        ),
        evidence_units=_pct([r["evidence"]["n_units"] for r in ok]),
        evidence_chars=_pct([r["evidence"]["n_chars"] for r in ok]),
        removed_retrieved=_pct([len(r["evidence"]["removed_retrieved_ids"]) for r in ok]),
        n_docs_graded=_pct([len(r["grading_events"]) for r in ok]),
        e2e_latency=_pct([r["e2e_latency"] for r in ok]),
        stage_timings={
            s: _pct([r["stage_timings"][s] for r in ok if r["stage_timings"].get(s) is not None])
            for s in stage_names
        },
        generation_input_tokens=_pct(
            [r["generation_input_tokens"] for r in ok if r["generation_input_tokens"] is not None]
        ),
        generation_output_tokens=_pct(
            [r["generation_output_tokens"] for r in ok if r["generation_output_tokens"] is not None]
        ),
    )
    return out


def _calls_by_kind(calls: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for c in calls:
        k = out.setdefault(
            c["kind"], {"n": 0, "prompt_tokens": 0, "out_tokens": 0, "wall": 0.0, "gate_wait": 0.0}
        )
        k["n"] += 1
        for f in ("prompt_tokens", "out_tokens"):
            k[f] = None if k[f] is None or c[f] is None else k[f] + c[f]
        k["wall"] += c["wall"]
        k["gate_wait"] += c["gate_wait"]
    return out


class ModeBRunner:
    def __init__(
        self,
        cfg: dict,
        components: Components | None = None,
        run_id: str | None = None,
        paper_approved: bool = False,
        workload: dict | None = None,
    ):
        self.cfg = cfg
        self.purpose = check_purpose(cfg, paper_approved)
        self.mode = (cfg.get("run") or {}).get("mode", "sequential")
        if self.mode not in ("sequential", "live"):
            raise ValueError("run.mode: sequential | live")
        if self.mode == "live" and workload is None:
            raise ValueError("live 모드는 --workload 필요")
        inputs = cfg.get("inputs") or {}
        self.items, self.input_info = load_frozen(
            inputs["frozen_path"], verify_manifest=inputs.get("verify_manifest", True)
        )
        if inputs.get("limit"):
            self.items = self.items[: inputs["limit"]]
        self.by_id = {it.question_id: it for it in self.items}
        n_warm = (cfg.get("run") or {}).get("warmup_queries", 0) or 0
        self.warmup = self.items[:n_warm]
        self.requests = self._requests(workload)
        self.workload = workload
        # 설정 검증 (정책·budget·grader 조합) 을 결과 폴더 생성 전에
        get_policy(cfg.get("evidence_policy", "official"))
        build_budget(cfg.get("budget"))
        p, e = self.purpose["purpose"], self.purpose["exp_id"]
        self.run_dir = create_run_dir(
            f"official_crag/{p}/{e}", cfg.get("output_dir", "results"), run_id
        )
        self.run_id = self.run_dir.name
        if components is None:
            evaluator = build_evaluator(cfg["evaluator"]) if needs_evaluator(cfg) else None
            components = Components(
                engine=build_generator_engine(cfg["llm"]),
                web_search=build_web_search(cfg, self.run_dir)
                if cfg.get("evidence_policy") != "vanilla"
                else None,
                evaluator=evaluator,
            )
        self.comp = components
        self.pipeline = build_pipeline(cfg, components)
        sched = cfg.get("scheduler") or {}
        self.gate = None
        if sched.get("policy", "none") not in (None, "none"):
            self.gate = SchedulingGate(
                build_policy(sched, None),
                slots=sched.get("slots", 1),
                engine_priority=sched.get("engine_priority", "none"),
            )
        self.queries = JsonlWriter(self.run_dir / "queries.jsonl")
        self.calls = JsonlWriter(self.run_dir / "calls.jsonl")
        self.grading_docs = JsonlWriter(self.run_dir / "grading_docs.jsonl")
        self.doc_tokens, self.doc_tokenizer = (
            t5_doc_token_counter(components.evaluator)
            if components.evaluator is not None
            else (None, None)
        )
        self.doc_rows: list[dict] = []
        self.rows: list[dict] = []
        self.lock = threading.Lock()

    def _requests(self, workload):
        if workload is None:
            return [
                {"i": i, "query_id": it.question_id, "arrival": None}
                for i, it in enumerate(self.items)
            ]
        missing = {r["query_id"] for r in workload["requests"]} - set(self.by_id)
        if missing:
            raise ValueError(f"workload 질문이 고정 입력에 없음: {sorted(missing)[:5]}")
        return workload["requests"]

    def _meta_extra(self) -> dict:
        ev = self.comp.evaluator
        return {
            "experiment_mode": "B_controlled",
            "purpose": self.purpose,
            "mode": self.mode,
            "inputs": self.input_info,
            "pipeline": self.pipeline.describe(),
            "evaluator": ev.describe() if ev is not None else None,
            "evaluator_weights": (
                weights_fingerprint(ev.backend.model_path)
                if ev is not None and hasattr(ev.backend, "model_path")
                else None
            ),
            "official_source": {
                "repo": off_source.OFFICIAL_REPO_URL,
                "commit": off_source.OFFICIAL_COMMIT,
                "license": off_source.OFFICIAL_LICENSE,
                "weights_url": off_source.EVALUATOR_WEIGHTS_URL,
            },
            "scheduler": self.gate.describe() if self.gate else None,
            "n_requests": len(self.requests),
            "n_warmup": len(self.warmup),
            "workload": None if self.workload is None else self.workload.get("sha256"),
            "not_official_reproduction": (
                "MODE B 는 공식 요소를 기존 serving 환경에 이식한 통제 비교이며 공식 논문 수치의 재현이 아님"  # noqa: E501
            ),
        }

    def _execute(self, req: dict, t0: float, warmup: bool) -> dict:
        item = self.by_id[req["query_id"]]
        arrival = req["arrival"] if self.mode == "live" and not warmup else None
        t_start = time.perf_counter() - t0
        q = QueryState(item.question_id, arrival=arrival if arrival is not None else t_start, t0=t0)
        rec = CallRecorder(t0, record_texts=bool((self.cfg.get("run") or {}).get("record_texts")))
        kw: dict[str, Any] = {"recorder": rec, "query_state": q}
        if self.gate is not None:
            kw["scheduler"] = self.gate
        llm = EngineLLM(self.comp.engine, **kw)
        row: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "request_id": f"{self.run_id}:{req['i']}:{item.question_id}",
            "request_index": req["i"],
            "question_id": item.question_id,
            "question": item.question,
            "experiment_mode": "B_controlled",
            "warmup": warmup,
            "evidence_policy": self.cfg.get("evidence_policy"),
            "arrival": arrival,
        }
        try:
            web = self.comp.web_search
            web = web.bind(item.question_id) if web is not None and hasattr(web, "bind") else web
            row.update(self.pipeline.run(item, llm, query_state=q, web=web))
        except Exception as e:  # noqa: BLE001 - 기록하고 계속
            row["error"] = f"{type(e).__name__}: {e}"[:1000]
            q.finish()
        calls = rec.to_dicts()
        by_kind = _calls_by_kind(calls)
        gen = by_kind.get("generate") or {}
        row.update(
            llm_calls=by_kind,
            generation_input_tokens=gen.get("prompt_tokens") if gen else None,
            generation_output_tokens=gen.get("out_tokens") if gen else None,
            queue_wait={
                "client_gate": sum(c["gate_wait"] for c in calls) if calls else None,
                "engine_queue": None,  # vLLM 은 요청 단위 대기시간을 주지 않음
            },
            finish=time.perf_counter() - t0,
        )
        missing = validate_record(row)
        if missing:
            row["schema_missing"] = missing
        for c in calls:
            self.calls.write({"request_index": req["i"], "question_id": item.question_id, **c})
        self.queries.write(row)
        # 문서별 grading 로그 (측정 구간 밖. 기록된 값만 펼침 + T5 문서 토큰 수)
        drows = doc_rows(row, item.docs, self.doc_tokens, self.doc_tokenizer)
        for d in drows:
            self.grading_docs.write(d)
        with self.lock:
            self.rows.append(row)
            self.doc_rows += drows
        return row

    def _manifest(self) -> dict:
        ev = self.comp.evaluator
        ws = self.comp.web_search
        web = None
        if ws is not None:
            web = {**(self.cfg.get("web_search") or {})}
            web["stats_at_start"] = ws.stats() if hasattr(ws, "stats") else None
            paths = getattr(ws, "paths", None) or []
            web["cache_files"] = [
                {"path": p, "sha256": file_sha256(p) if Path(p).is_file() else None} for p in paths
            ]
        here = Path(__file__).resolve().parent
        return build_manifest(
            {**self.cfg, "experiment_mode": "B_controlled"},
            self.input_info,
            [it.question_id for it in self.items],
            self.pipeline.describe(),
            ev.describe() if ev is not None else None,
            (
                weights_fingerprint(
                    ev.backend.model_path,
                    full_hash=bool((self.cfg.get("manifest") or {}).get("full_weight_hash")),
                )
                if ev is not None and hasattr(ev.backend, "model_path")
                else None
            ),
            web,
            git_info(),
            {
                f: file_sha256(here / f) if (here / f).is_file() else None
                for f in ("mode_b.py", "analyze.py", "analyze_correction.py")
            },
        )

    def run(self) -> Path:
        engines = {"generator": self.comp.engine}
        write_run_files(
            self.run_dir, self.cfg, collect_run_metadata(self.cfg, engines, self._meta_extra())
        )
        write_new(
            self.run_dir / "manifest.json",
            json.dumps(self._manifest(), indent=2, ensure_ascii=False, default=str),
        )
        t0 = time.perf_counter()
        for i, it in enumerate(self.warmup):
            self._execute({"i": -1 - i, "query_id": it.question_id, "arrival": None}, t0, True)
        t0 = time.perf_counter()
        if self.mode == "sequential":
            for req in self.requests:
                self._execute(req, t0, False)
        else:
            threads = []
            for req in sorted(self.requests, key=lambda r: r["arrival"]):
                delay = t0 + req["arrival"] - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                th = threading.Thread(target=self._execute, args=(req, t0, False), daemon=True)
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
        write_doc_csv(self.doc_rows, self.run_dir / "grading_docs.csv")
        summary = summarize(self.rows)
        summary["grading_docs"] = {
            "n_rows": len(self.doc_rows),
            "n_evaluated": sum(bool(d["evaluated"]) for d in self.doc_rows),
            "timing_kind": {
                k: sum(d["timing_kind"] == k for d in self.doc_rows)
                for k in sorted({str(d["timing_kind"]) for d in self.doc_rows})
            },
        }
        ws = self.comp.web_search
        summary["web_search"] = ws.stats() if ws is not None and hasattr(ws, "stats") else None
        ev = self.comp.evaluator
        if ev is not None and hasattr(ev.backend, "hits"):
            summary["evaluator_cache"] = {"hits": ev.backend.hits, "misses": ev.backend.misses}
        write_new(
            self.run_dir / "summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        )
        return self.run_dir
