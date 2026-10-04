import json
from pathlib import Path

from rag.types import QAExample


def load_qa_jsonl(path: str | Path, limit: int | None = None) -> list[QAExample]:
    """한 줄당 {"id", "question", "answers", ...} 형식의 jsonl 을 읽는다."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                QAExample(
                    id=str(row.pop("id")),
                    question=row.pop("question"),
                    answers=row.pop("answers"),
                    metadata=row,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples
