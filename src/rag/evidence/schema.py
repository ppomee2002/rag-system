"""MODE B 결과 (queries.jsonl) 한 줄의 필드 정의와 검사.

측정하지 못한 값은 0 이 아니라 None (JSON null). 실행하지 않은 stage 도 None.
wall-clock (stage_timings, e2e_latency) 과 GPU 계산 시간은 섞지 않는다:
  evaluator_compute_seconds : evaluator backend 가 보고한 계산 시간 (cache 면 기록 당시 값)
  queue_wait.client_gate    : client scheduler gate 대기 (rag.scheduling)
  queue_wait.engine_queue   : vLLM 내부 대기 — 요청 단위로 주지 않으므로 None (run 단위
                              /metrics 샘플은 engine_metrics.jsonl)
"""

from __future__ import annotations

SCHEMA_VERSION = "official_crag.query.v1"

FIELDS = {
    # 식별
    "schema": "SCHEMA_VERSION",
    "run_id": "결과 폴더 이름",
    "request_id": "<run_id>:<request_index>:<question_id>",
    "request_index": "요청 순서 (warmup 은 음수)",
    "question_id": "PopQA id",
    "question": "질문",
    "experiment_mode": "B_controlled (MODE A 는 별도 형식)",
    "warmup": "측정 제외 여부",
    # 조건
    "grader_type": "official_t5 | llama_prompt | None (vanilla)",
    "score_scale": "t5_regression_logit | llm_0_1_parsed",
    "thresholds": "{upper, lower} (grader 별)",
    "evidence_policy": "rag.evidence.policies 이름",
    "policy_id": "<정책>@<grader>[+budget]",
    "budget": "{name, stop_rule, retention}",
    "official_policy_semantics": "official 정책 + official_t5 grader 인지",
    # 흐름
    "correction_path": "correct | ambiguous | incorrect | none",
    "decision": "{action, internal_mode, reason, stopped_early, action_final, ...}",
    "retrieval_doc_ids": "고정 Top-k id (순서)",
    "grading_events": "문서별 {rank, doc_id, score, flag, label, latency, input_tokens, ...}",
    "grading_snapshots": "C2: 문서마다 누적 상태 (rag.evidence.state.GradingSnapshot)",
    "refinement_events": "[internal, external] 공식 refinement (strip 점수·선택·순위) 또는 null",
    "correction": "{performed, query, status (ok|empty|error|replay_miss|...), valid, web_doc_ids}",
    "evidence": "{kept_retrieved_ids, removed_retrieved_ids, added_ids, selected_evidence_ids, ...}",  # noqa: E501
    # 비용
    "generation_input_tokens": "generate 호출 prompt 토큰 (엔진 보고, 없으면 None)",
    "generation_output_tokens": "generate 호출 출력 토큰",
    "llm_calls": "호출 종류별 {n, prompt_tokens, out_tokens, wall, gate_wait}",
    "stage_timings": "STAGES_OFFICIAL 별 wall 초 (실행 안 함 = None)",
    "evaluator_compute_seconds": "evaluator backend 계산 시간 합",
    "queue_wait": "{client_gate, engine_queue}",
    "e2e_latency": "질문 하나 wall 초",
    # 결과
    "prediction": "최종 답 (공식 prompt 면 공식 후처리 후)",
    "ground_truth": "정답 목록",
    "is_correct": "metrics.acc (기존 정규화 포함 판정) 기준",
    "metrics": "{acc, em, f1, official_match}",
    "error": "실패 시 메시지 (정확도 집계에서 제외, 개수는 summary 에 보고)",
}

REQUIRED_OK = tuple(k for k in FIELDS if k != "error")
REQUIRED_ERROR = (
    "schema",
    "run_id",
    "request_id",
    "question_id",
    "experiment_mode",
    "evidence_policy",
    "error",
)


def validate_record(row: dict) -> list[str]:
    """빠진 필드 목록 (빈 목록이면 통과). 값이 None 인 것은 허용 (미측정)."""
    need = REQUIRED_ERROR if row.get("error") else REQUIRED_OK
    return [k for k in need if k not in row]
