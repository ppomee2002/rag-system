"""C2: 문서별 Grading 로그 (질문 x 검색 문서 한 행).

MODE B 결과 행 (queries.jsonl) 의 grading_events / grading_snapshots 를 검색 문서 단위로 펼친다.
새 값을 계산하지 않고 이미 기록된 값만 옮긴다 (예외: document_tokens — T5 토크나이저가 있을 때
문서 단독 토큰 수를 셈. 측정 구간 밖에서 계산하므로 latency 에 영향 없음).

연결: (request_id, eval_order) -> queries.jsonl 의 grading_events[eval_order],
      snapshot_index -> grading_snapshots[snapshot_index] (이 문서까지 평가한 뒤의 누적 상태).
평가하지 않은 문서 (조기 중단 등) 는 evaluated=false 이고 점수·시간·토큰은 모두 null.

점수 체계 (섞지 않는다)
  grader_score        판정에 쓴 점수 (official_doc_flag 의 입력)
    official_t5       T5 회귀 logit. 빈 문서는 모델을 돌리지 않고 공식 규칙 -1.0
    legacy_llama      GRADE prompt 출력에서 parse_score 로 읽은 0~1 숫자 (생성 텍스트)
  t5_raw_logit        T5 모델이 실제로 낸 logit (모델·캐시·mock). 규칙 점수 (empty_rule) 면 null
  llama_parsed_score  legacy_llama 의 0~1 파싱 점수 (= grader_score). T5 면 null
  llama_yes_logprob   항상 null: 기존 Llama grader 는 logprobs 를 요청하지 않는다 (EngineLLM 은
                      텍스트만 반환). 0~1 점수로 대신하지 않는다. 상태는 llama_yes_logprob_status
토큰
  grading_input_tokens   official_t5: T5 입력 ("질문 [SEP] 문서") 토큰 수 (attention mask 합,
                         판정 경로는 truncation 없음 -> 실제 모델 입력 전체). 캐시면 기록 당시 값.
                         legacy_llama: 엔진이 보고한 prompt 토큰 수
  grading_output_tokens  official_t5: null (회귀 출력, 생성 토큰 없음 — grading_output_kind 참고)
                         legacy_llama: 엔진이 보고한 생성 토큰 수
시간
  grading_latency_ms  이번 run 에서 이 문서 평가에 걸린 wall 시간 (judge 호출 전체)
  grading_compute_ms  backend 계산 시간. 캐시 재생이면 기록 당시 값 (is_cache_replay=true)
  timing_kind         live_model | live_llm | cache_replay | rule_no_model | mock
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

DOC_LOG_SCHEMA = "official_crag.grading_doc.v1"

GRADER_TYPE_LABEL = {"official_t5": "official_t5", "llama_prompt": "legacy_llama"}

LLAMA_LOGPROB_STATUS = (
    "not_extracted: legacy grader (rag.pipelines.prompts.GRADE + parse_score) 는 logprobs 를 "
    "요청하지 않음"
)

TIMING_KIND = {
    "model": "live_model",
    "llm": "live_llm",
    "cache": "cache_replay",
    "empty_rule": "rule_no_model",
    "mock": "mock",
}

INPUT_TOKENS_KIND = {
    "model": "t5_attention_mask_sum_no_truncation",
    "cache": "t5_attention_mask_sum_recorded",
    "empty_rule": "no_model_call",
    "mock": "mock_word_count",
    "llm": "engine_reported_prompt_tokens",
}

FIELDS = (
    "schema",
    "run_id",
    "request_id",
    "query_id",
    "warmup",
    "policy_id",
    "evidence_policy",
    "final_action",
    "doc_rank",
    "doc_id",
    "retrieval_score",
    "document_chars",
    "document_tokens",
    "document_tokens_tokenizer",
    "grader_type",
    "grader_impl",
    "score_scale",
    "evaluated",
    "eval_order",
    "snapshot_index",
    "grader_score",
    "t5_raw_logit",
    "llama_parsed_score",
    "llama_yes_logprob",
    "llama_yes_logprob_status",
    "grader_flag",
    "grader_pass",
    "grader_decision",
    "decision_rule",
    "threshold_upper",
    "threshold_lower",
    "grading_input_tokens",
    "grading_input_tokens_kind",
    "grading_output_tokens",
    "grading_output_kind",
    "grading_latency_ms",
    "grading_compute_ms",
    "timing_kind",
    "is_cache_replay",
    "grading_source",
    "provisional_action_after",
    "action_final_after",
    "grader_raw_output",
)


def decision_rule(upper: float, lower: float) -> str:
    return (
        f"doc: score>={upper}->pass(flag2), score>={lower}->partial(flag1), else fail(flag0); "
        "query: any pass->correct, else any partial->ambiguous, else incorrect"
    )


def _ms(sec):
    return None if sec is None else sec * 1000.0


def doc_rows(
    row: dict,
    docs: list,
    doc_tokens: Callable[[str], int | None] | None = None,
    tokenizer_name: str | None = None,
) -> list[dict]:
    """queries.jsonl 한 행 + 고정 입력 문서 (검색 순서) -> 문서별 로그 행.

    grading 을 하지 않은 행 (vanilla, 오류) 은 [] .
    """
    if row.get("error") or not row.get("grader_type"):
        return []
    gimpl = row["grader_type"]
    th = row.get("thresholds") or {}
    upper, lower = th.get("upper"), th.get("lower")
    events = {e["rank"]: e for e in row.get("grading_events") or []}
    snaps = row.get("grading_snapshots") or []
    is_t5 = gimpl == "official_t5"
    out = []
    for rank, d in enumerate(docs):
        e = events.get(rank)
        ev = e is not None
        src = e.get("source") if ev else None
        order = e.get("eval_order") if ev else None
        snap = snaps[order] if ev and order is not None and order < len(snaps) else None
        flag = e.get("flag") if ev else None
        score = e.get("score") if ev else None
        n_tok = doc_tokens(d.text) if (doc_tokens is not None and is_t5) else None
        out.append(
            {
                "schema": DOC_LOG_SCHEMA,
                "run_id": row.get("run_id"),
                "request_id": row.get("request_id"),
                "query_id": row.get("question_id"),
                "warmup": row.get("warmup"),
                "policy_id": row.get("policy_id"),
                "evidence_policy": row.get("evidence_policy"),
                "final_action": row.get("correction_path"),
                "doc_rank": rank,
                "doc_id": d.id,
                "retrieval_score": d.score,
                "document_chars": len(d.text),
                "document_tokens": n_tok,
                "document_tokens_tokenizer": tokenizer_name if n_tok is not None else None,
                "grader_type": GRADER_TYPE_LABEL.get(gimpl, gimpl),
                "grader_impl": gimpl,
                "score_scale": row.get("score_scale"),
                "evaluated": ev,
                "eval_order": order,
                "snapshot_index": order,
                "grader_score": score,
                "t5_raw_logit": score if is_t5 and src in ("model", "cache", "mock") else None,
                "llama_parsed_score": score if not is_t5 else None,
                "llama_yes_logprob": None,
                "llama_yes_logprob_status": (
                    "not_applicable" if is_t5 else LLAMA_LOGPROB_STATUS if ev else "not_evaluated"
                ),
                "grader_flag": flag,
                "grader_pass": None if flag is None else flag == 2,
                "grader_decision": e.get("label") if ev else None,
                "decision_rule": decision_rule(upper, lower),
                "threshold_upper": upper,
                "threshold_lower": lower,
                "grading_input_tokens": e.get("input_tokens") if ev else None,
                "grading_input_tokens_kind": INPUT_TOKENS_KIND.get(src) if ev else None,
                "grading_output_tokens": e.get("output_tokens") if ev else None,
                "grading_output_kind": (
                    None
                    if not ev
                    else "regression_scalar_no_generated_tokens"
                    if is_t5
                    else "engine_reported_completion_tokens"
                ),
                "grading_latency_ms": _ms(e.get("latency")) if ev else None,
                "grading_compute_ms": _ms(e.get("compute_seconds")) if ev else None,
                "timing_kind": TIMING_KIND.get(src, src) if ev else None,
                "is_cache_replay": (src == "cache") if ev else None,
                "grading_source": src,
                "provisional_action_after": snap.get("provisional_action") if snap else None,
                "action_final_after": snap.get("action_final") if snap else None,
                "grader_raw_output": e.get("raw_output") if ev else None,
            }
        )
    return out


def t5_doc_token_counter(evaluator) -> tuple[Callable[[str], int] | None, str | None]:
    """evaluator backend 에 T5 토크나이저가 있으면 (문서 정규화 후 토큰 수 함수, 이름)."""
    backend = getattr(evaluator, "backend", None)
    tok = getattr(backend, "tokenizer", None)
    if tok is None:
        return None, None

    def count(text: str) -> int:
        return len(tok(evaluator._prep(text))["input_ids"])  # </s> 포함

    return count, f"{type(tok).__name__} (normalize_passage, +eos)"


def write_csv(jsonl_rows: list[dict], path: Path) -> None:
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS), extrasaction="ignore")
        w.writeheader()
        for r in jsonl_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
