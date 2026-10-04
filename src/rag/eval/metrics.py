import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in answers))


def _f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, answers: list[str]) -> float:
    return max((_f1(prediction, a) for a in answers), default=0.0)


def accuracy(prediction: str, answers: list[str]) -> float:
    """정답 문자열이 답변에 포함되면 1 (PopQA 등 C-RAG 논문 평가 방식)."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) in pred for a in answers))


def score_all(prediction: str, answers: list[str]) -> dict[str, float]:
    return {
        "em": exact_match(prediction, answers),
        "f1": f1_score(prediction, answers),
        "acc": accuracy(prediction, answers),
    }
