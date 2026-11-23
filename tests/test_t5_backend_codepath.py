"""T5EvaluatorBackend 코드 경로 테스트 (작은 무작위 T5 + 작은 sentencepiece 로 구성).

** 공식 가중치가 아니다 ** — 로딩 (T5Tokenizer + T5ForSequenceClassification, safetensors),
입력 형식, truncation 유무, 점수 추출, 토큰 수 기록이 현재 설치된 transformers 에서 동작하는지만
본다.
공식 가중치 추론 검증은 서버의 smoke_evaluator.py.
"""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")


@pytest.fixture(scope="module")
def tiny_t5(tmp_path_factory):
    """공식 Drive 폴더와 같은 구성 (spiece.model + tokenizer_config.json, tokenizer.json 없음)."""
    import json

    d = tmp_path_factory.mktemp("tiny_t5")
    out = d / "model"
    out.mkdir()
    words = (
        "the of and to in is was a an occupation politician born city question document " * 40
    ).split()
    lines = [" ".join(words[i : i + 12]) for i in range(0, 400, 12)]
    (d / "c.txt").write_text("\n".join(lines))
    spm.SentencePieceTrainer.train(
        input=str(d / "c.txt"),
        model_prefix=str(out / "spiece"),
        vocab_size=60,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    (out / "spiece.vocab").unlink()
    tok_cfg = {
        "tokenizer_class": "T5Tokenizer",
        "model_max_length": 512,
        "extra_ids": 0,
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tok_cfg))
    (out / "special_tokens_map.json").write_text(
        json.dumps({k: tok_cfg[k] for k in ("eos_token", "unk_token", "pad_token")})
    )
    from transformers import T5Config, T5ForSequenceClassification

    torch.manual_seed(0)
    sp = spm.SentencePieceProcessor(model_file=str(out / "spiece.model"))
    cfg = T5Config(
        vocab_size=sp.get_piece_size(),
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        num_labels=1,
    )
    T5ForSequenceClassification(cfg).save_pretrained(out)
    return out


def test_tokenizer_matches_raw_sentencepiece(tiny_t5):
    """transformers 5.x 토크나이저 id 가 원본 (4.36 slow T5Tokenizer = sentencepiece) 과 같은지."""
    from rag.crag_official.evaluator import _load_t5_tokenizer, tokenization_check

    tok = _load_t5_tokenizer(str(tiny_t5))
    rep = tokenization_check(
        tok, str(tiny_t5 / "spiece.model"), ["what is the occupation [SEP] a politician born"]
    )
    assert rep["all_equal"], rep


def test_tiny_t5_backend_scores(tiny_t5):
    from rag.crag_official.evaluator import (
        T5EvaluatorBackend,
        build_evaluator,
        check_weights_dir,
    )

    rep = check_weights_dir(tiny_t5)
    assert rep["model_type"] == "t5"
    b = T5EvaluatorBackend(tiny_t5, device="cpu")
    pairs = [("what is the occupation", "a politician born in the city"), ("q", "document")]
    s1 = b.score_pairs(pairs, truncation=False)
    assert len(s1) == 2 and all(math.isfinite(p.score) for p in s1)
    assert all(p.n_tokens and p.n_tokens > 0 for p in s1)
    # 같은 입력 -> 같은 점수 (결정성), truncation 여부는 짧은 입력에서 차이 없음
    s2 = b.score_pairs(pairs, truncation=True)
    assert [round(p.score, 5) for p in s1] == [round(p.score, 5) for p in s2]
    long = ("q", " ".join(["politician city"] * 400))
    assert b.score_pairs([long], truncation=True)[0].n_tokens <= 512
    assert b.score_pairs([long], truncation=False)[0].n_tokens > 512  # 공식 판정 입력은 자르지 않음
    # batch (truncation=True) 와 1개씩 결과가 같다
    bb = T5EvaluatorBackend(tiny_t5, device="cpu", batch_size=4)
    many = [("q", f"document number {i} politician") for i in range(5)]
    one = [p.score for p in b.score_pairs(many, truncation=True)]
    bat = [p.score for p in bb.score_pairs(many, truncation=True)]
    assert max(abs(x - y) for x, y in zip(one, bat, strict=True)) < 1e-4
    ev = build_evaluator({"task": "popqa"}, backend=b)
    v = ev.evaluate("what is the occupation", ["a politician", ""])
    assert v.scores[1] == -1.0 and v.action in ("correct", "ambiguous", "incorrect")
    assert b.describe()["weights"]["exists"]
