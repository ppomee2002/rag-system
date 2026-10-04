from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query_id: str
    pipeline: str
    question: str
    answer: str
    contexts: list[Document]
    e2e_latency: float
    stage_latencies: dict[str, float] = field(default_factory=dict)
    # C-RAG 전용 (vanilla 는 기본값 유지)
    corrective_action: str | None = None   # "correct" | "incorrect" | "ambiguous"
    grader_scores: list[float] = field(default_factory=list)
    n_corrective_steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
