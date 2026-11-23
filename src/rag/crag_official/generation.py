"""공식 CRAG generation prompt / 후처리 / 정답 판정 (PopQA).

format_popqa_prompt    CRAG_Inference.py:41-60 (task=popqa 는 instruction == question 경로)
postprocess_answer     CRAG_Inference.py:79-91
official_match         metrics.py:83-87 — 정답 문자열이 예측에 (대소문자 구분) 포함되면 1

이 레포 기존 판정 rag.eval.metrics.accuracy 는 소문자·구두점·관사 정규화 후 포함 여부라 값이
다를 수 있다. 결과에는 두 값을 모두 기록한다 (official_match, acc).
"""

from __future__ import annotations

from rag.crag_official import source


def format_popqa_prompt(question: str, paragraph: str | None) -> str:
    if paragraph is None:  # method=no_retrieval: 원본은 None 과 문자열 결합으로 실패한다
        raise ValueError("공식 PopQA prompt 는 paragraph 가 필요 (no_retrieval 은 원본에서도 실패)")
    paragraph = " ".join(paragraph.split(" ")[:])  # 원본과 같은 (사실상 항등) 처리
    # str.format 을 쓰지 않는다: 문서에 '{' 가 있으면 깨진다
    head, _, tail = source.POPQA_PROMPT.partition("{paragraph}")
    mid, _, end = tail.partition("{question}")
    return head + paragraph + mid + question + end


def postprocess_answer(answer: str) -> str:
    for token in source.CONTROL_TOKENS:
        answer = answer.replace(token, "")
    for token in ("</s>", "\n", "<|endoftext|>"):
        answer = answer.replace(token, "")
    return answer


def official_match(prediction: str, answers: list[str]) -> int:
    return int(any(gt in prediction for gt in answers))


def rag_paragraph(passages: list[str], n_docs: int = source.OFFICIAL_NDOCS) -> str:
    """method=rag 의 문단 (CRAG_Inference.py:99-131 data_preprocess)."""
    return source.RAG_PASSAGE_JOIN.join(passages[:n_docs])
