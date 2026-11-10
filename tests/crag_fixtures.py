"""crag_system runner 테스트 공용 가짜 구성요소와 fixture (test_runner, test_paper_validity)."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from rag.llm.engine import LLMResponse
from rag.telemetry.calls import infer_call_kind
from rag.types import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "crag_system"))
from runner import (  # noqa: E402
    Components,
    load_examples,
    load_layered_config,
    split_warmup,
)


class FakeEngine:
    """query 번호가 짝수면 relevant (correct), 홀수면 incorrect 로 응답하는 in-process 엔진."""

    name, model = "fake", "fake-model"

    def __init__(self, delay=0.002, batch_reply=None):
        self.delay = delay
        self.batch_reply = batch_reply
        self.active = self.max_active = 0
        self.lock = threading.Lock()
        self.priorities = []

    def complete(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.priorities.append(request.priority)
        t0 = time.perf_counter()
        time.sleep(self.delay)
        kind = request.kind if request.kind != "other" else infer_call_kind(request.prompt)
        good = "Question: q0" in request.prompt or "even" in request.prompt
        if kind == "grade":
            text = "0.9" if good else "0.1"
        elif kind == "grade_batch":
            text = self.batch_reply or (
                '{"scores": [0.9, 0.1, 0.1]}' if good else '{"scores": [0.1, 0.1, 0.1]}'
            )
        elif kind == "rewrite":
            text = "rewritten"
        else:
            text = "ans"
        with self.lock:
            self.active -= 1
        return LLMResponse(
            text,
            "fake",
            model="fake-model",
            t_start=t0,
            t_end=time.perf_counter(),
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=2,
        )

    def describe(self):
        return {"engine": "fake"}


class Retriever:
    def retrieve(self, query, top_k):
        tag = "even" if query.endswith(("0?", "2?", "4?", "6?", "8?")) else "odd"
        return [
            Document(id=f"{tag}{i}", text=f"{tag} doc {i}", score=0.8 - 0.1 * i)
            for i in range(top_k)
        ]


class Web:
    def __init__(self):
        self.n = 0

    def search(self, query, top_k):
        self.n += 1
        return [Document(id=f"w{i}", text=f"web {i}") for i in range(top_k)]


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps(
                {"id": f"id{i}", "question": f"Question {i}?", "answers": ["ans"], "s_pop": 10 * i}
            )
            + "\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "results"),
                "dataset": {"path": str(data), "limit": None},
                "retriever": {"top_k": 3},
                "crag": {"web_search": {"top_k": 2}},
                "run": {"warmup_queries": 2},
            }
        ),
        encoding="utf-8",
    )

    def make(sets=(), engine=None, web=None):
        cfg = load_layered_config(ROOT / "experiments/crag_system/config.yaml", [overlay], sets)
        comp = Components(Retriever(), web or Web(), engine or FakeEngine())
        return cfg, comp

    return tmp_path, make


def pool_ids(cfg):
    warm, pool = split_warmup(cfg, load_examples(cfg))
    return [ex.id for ex in warm], [ex.id for ex in pool]


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]
