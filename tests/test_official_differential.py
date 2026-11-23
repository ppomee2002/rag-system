"""공식 코드 vs 재구현 차등 테스트 (공식 저장소가 있을 때만).

공식 저장소 (기준 commit) 의 함수를 직접 import 해서 같은 입력에 같은 출력을 내는지 비교한다.
공식 코드는 이 레포에 포함하지 않는다: experiments/official_crag/fetch_official.py repo 로
third_party/CRAG 에 받거나 환경변수 CRAG_OFFICIAL_REPO 로 위치를 지정한다. 없으면 skip.

무거운 import (vllm, openai, bs4, sklearn, jsonlines) 는 빈 모듈로 대체한다. evaluator 는 텍스트로
점수를 정하는 가짜 모델을 양쪽에 똑같이 넣는다 (가중치 추론 검증이 아님).
"""

import hashlib
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rag.crag_official import source
from rag.crag_official.evaluator import (
    MockScoreBackend,
    OfficialRetrievalEvaluator,
    official_doc_flag,
    pair_text,
    process_flag,
)
from rag.crag_official.generation import format_popqa_prompt, official_match, postprocess_answer
from rag.crag_official.official_data import load_official_queries, retrieved_psgs_line
from rag.crag_official.refinement import (
    extract_strips,
    internal_knowledge,
    official_passages_from_line,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("CRAG_OFFICIAL_REPO", ROOT / "third_party" / "CRAG"))
pytestmark = pytest.mark.skipif(
    not (REPO / "scripts" / "CRAG_Inference.py").exists(),
    reason="공식 저장소 없음 (fetch_official.py repo)",
)
torch = pytest.importorskip("torch")


def _commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return None


STUBS = {
    "vllm": {"LLM": object, "SamplingParams": object},
    "openai": {},
    "jsonlines": {},
    "bs4": {"BeautifulSoup": object},
    "sklearn": {},
    "sklearn.utils": {"shuffle": lambda x: x},
}


@pytest.fixture(scope="module")
def official():
    import importlib.machinery

    import transformers  # noqa: F401  (stub 을 넣기 전에 실제 의존성을 먼저 로드)
    from transformers import T5ForSequenceClassification  # noqa: F401

    saved = {k: sys.modules.get(k) for k in STUBS}
    for name, attrs in STUBS.items():
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
    sys.path.insert(0, str(REPO / "scripts"))
    mods = {}
    try:
        for name in ("utils", "CRAG_Inference", "internal_knowledge_preparation", "metrics"):
            spec = importlib.util.spec_from_file_location(
                f"official_{name}", REPO / "scripts" / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mods[name] = mod
    finally:
        sys.path.remove(str(REPO / "scripts"))
        for k, v in saved.items():  # 다른 테스트에 stub 이 남지 않도록
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in [k for k in sys.modules if k in ("utils", "metrics")]:
            sys.modules.pop(k, None)
    return mods


def score_of(text: str) -> float:
    """텍스트 -> 결정적 점수 (-2 ~ 2). 동점을 일부러 만든다 (안정 정렬 확인)."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:6], 16)
    return round((h % 9) / 2 - 2, 1)


class FakeTok:
    def __init__(self):
        self.texts = []

    def __call__(self, text, **kw):
        self.texts.append(text)
        i = len(self.texts) - 1
        return {"input_ids": torch.tensor([[i]]), "attention_mask": torch.ones(1, 1)}


class FakeModel:
    def __init__(self, tok):
        self.tok = tok

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        return {"logits": torch.tensor([[score_of(self.tok.texts[int(input_ids[0, 0])])]])}


@pytest.fixture(scope="module")
def queries():
    return load_official_queries(
        REPO / "data/popqa/test_popqa.txt", 10, REPO / "data/popqa/sources"
    )


def test_pinned_commit():
    c = _commit()
    if c:
        assert c == source.OFFICIAL_COMMIT


def test_data_preprocess_equal(official, queries):
    qs, psgs = official["CRAG_Inference"].data_preprocess(
        str(REPO / "data/popqa/test_popqa.txt"), 10
    )
    assert qs == [q.question for q in queries]
    assert psgs == [retrieved_psgs_line(q.passages) for q in queries]


def test_process_flag_equal(official):
    rng = random.Random(1)
    scores = [rng.choice([-1.0, -0.995, -0.5, 0.0, 0.592, 0.6, 1.3]) for _ in range(10 * 200)]
    lo = -0.995
    assert official["CRAG_Inference"].process_flag(scores, 10, 0.592, lo) == process_flag(
        scores, 10, 0.592, lo
    )


def test_inference_empty_passage_rule(official, queries, tmp_path):
    """공식 inference(): 빈 문서는 -1.0, 그 외는 모델 점수 -> 재구현 score_documents 와 같음."""
    sub = queries[:30]
    lines = []
    for q in sub:
        lines += [f"{q.question} [SEP] {p}" if p else f"{q.question} [SEP]" for p in q.passages]
    f = tmp_path / "t.txt"
    f.write_text("\n".join(lines), encoding="utf-8")
    tok = FakeTok()
    off = official["CRAG_Inference"].inference(tok, FakeModel(tok), str(f), n_docs=10)
    ev = OfficialRetrievalEvaluator(
        MockScoreBackend(lambda q, p: score_of(pair_text(q, p))), normalize=False
    )
    ours = [s.score for q in sub for s in ev.score_documents(q.question, q.passages)]
    assert off == pytest.approx(ours)
    flags = [official_doc_flag(s, 0.592, -0.995) for s in ours]
    assert -1.0 in ours and set(flags) <= {0, 1, 2}


@pytest.mark.parametrize("mode", ["selection", "excerption", "fixed_num"])
def test_extract_strips_equal(official, queries, mode):
    fn = official["internal_knowledge_preparation"].extract_strips_from_psg
    n_cmp = n_crash = 0
    for q in queries[:300]:
        for p in official_passages_from_line(retrieved_psgs_line(q.passages)):
            try:
                want = fn(p, mode)
            except IndexError:  # 공식 fixed_num: 10 단어 미만 문서
                n_crash += 1
                assert mode == "fixed_num" and len(p.split(" ")) < 10
                continue
            assert extract_strips(p, mode) == want
            n_cmp += 1
    assert n_cmp > 1000


@pytest.mark.parametrize("mode", ["selection", "excerption"])
def test_select_relevants_equal(official, queries, mode):
    """공식 knowledge_refinement 의 핵심 (strip 분해 -> select_relevants top_n) 과 같은 결과."""
    utils, ikp = official["utils"], official["internal_knowledge_preparation"]
    ev = OfficialRetrievalEvaluator(MockScoreBackend(lambda q, p: score_of(pair_text(q, p))))
    top_n = source.INTERNAL_TOP_N[mode]
    for q in queries[:150]:
        psgs = official_passages_from_line(retrieved_psgs_line(q.passages))
        strips = []
        for p in psgs:
            strips += ikp.extract_strips_from_psg(psg=p, mode=mode)
        tok = FakeTok()
        want_text, want_idx = utils.select_relevants(
            strips, q.question, tok, FakeModel(tok), "cpu", top_n
        )
        got = internal_knowledge(q.question, psgs, ev, mode)
        assert got.text == want_text.replace("\n", " ")
        assert ", ".join(str(i) for i in got.ranking) == want_idx


def test_generation_prompt_equal(official, queries):
    fmt = official["CRAG_Inference"].format_prompt
    for i, q in enumerate(queries[:50]):
        para = retrieved_psgs_line(q.passages)
        assert format_popqa_prompt(q.question, para) == fmt(i, "popqa", q.question, para, "llama")
    post = official["CRAG_Inference"].postprocess_answer_option_conditioned
    for s in ["[Relevant]Paris\n", "a</s>b<|endoftext|>", "[Utility:5] x"]:
        assert postprocess_answer(s) == post(s)


def test_match_equal(official):
    for pred, gts in [
        ("He is a Politician", ["politician"]),
        ("politician", ["politician"]),
        ("", ["x"]),
    ]:
        assert official_match(pred, gts) == official["metrics"].match(pred, gts)


def test_keyword_prompt_bug_documented(official):
    """공식 prompt 는 '{{question}}' 이라 format 후 질문이 빠진다 (KNOWN_OFFICIAL_ISSUES)."""
    import inspect

    src = inspect.getsource(official["utils"].extract_keywords)
    assert "{{question}}" in src
    assert any("{{question}}" in x for x in source.KNOWN_OFFICIAL_ISSUES)
