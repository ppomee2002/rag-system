"""공식 CRAG Retrieval Evaluator adapter.

공식 동작 (기준 commit, rag.crag_official.source):
  1. 문서마다 따로 점수를 낸다. 입력은 "질문 [SEP] 문서" 한 쌍, 모델은 T5-large 를 회귀로
     fine-tune 한 T5ForSequenceClassification (num_labels=1). 출력 logit 하나가 점수다
     (학습 label 이 -1 / +1 이라 대략 그 범위지만 범위 제한은 없다). CRAG_Inference.py:151-171
  2. 문서가 비어 있으면 모델을 돌리지 않고 -1.0. CRAG_Inference.py:159-162
  3. 문서별 flag: score >= upper -> 2, score >= lower -> 1, 그 외 0. CRAG_Inference.py:174-181
  4. 질문 단위 판정 (문서 집합): flag 2 가 하나라도 있으면 Correct, 아니면 flag 1 이 있으면
     Ambiguous, 아니면 Incorrect. CRAG_Inference.py:183-195

즉 점수는 문서별로 독립이고 (다른 문서에 의존하지 않음), 판정만 집합 단위 any() 이다.
문서별 점수를 순서대로 계산해 중간 상태를 기록해도 각 점수는 공식과 같다. 다만 일부 문서만 보고 내린
판정은 Correct (flag 2 발견) 인 경우만 최종 판정과 같다 (README "공식 판정과 중간 상태").

backend:
  T5EvaluatorBackend  실제 가중치 (torch / transformers / sentencepiece 필요, 지연 import)
  CachedScoreBackend  미리 계산한 점수 (GPU 를 vLLM 과 나눠 쓰지 못할 때 순차 실행용)
  MockScoreBackend    테스트용. 결과를 정확도로 쓰지 않는다
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rag.crag_official import source

ACTION_BY_FLAG = {2: "correct", 1: "ambiguous", 0: "incorrect"}


@dataclass
class PairScore:
    score: float
    n_tokens: int | None = None  # evaluator 입력 토큰 수 (padding 제외). 모르면 None
    compute_seconds: float | None = None  # backend 가 이 쌍에 쓴 시간. 캐시면 기록된 값
    source: str = "model"  # model | empty_rule | short_strip_rule | cache | mock | error
    error: str | None = None


class ScoreBackend(Protocol):
    name: str

    def score_pairs(self, pairs: list[tuple[str, str]], truncation: bool) -> list[PairScore]: ...

    def describe(self) -> dict[str, Any]: ...


def pair_text(query: str, passage: str) -> str:
    return query + source.PAIR_SEPARATOR + passage


def pair_key(query: str, passage: str, truncation: bool) -> str:
    raw = f"{int(truncation)}\x1f{query}\x1f{passage}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_passage(text: str) -> str:
    """공식 data_process.py:95 와 같은 정리 (strip, 줄바꿈·탭 -> 공백)."""
    return text.strip().replace("\n", " ").replace("\t", " ")


# ---------------------------------------------------------------- backends
class T5EvaluatorBackend:
    """공식 가중치로 점수 계산. batch_size=1 이 공식 동작 (한 쌍씩)."""

    name = "t5_official"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        batch_size: int = 1,
        max_length: int = source.EVAL_MAX_LENGTH,
        dtype: str = "float32",
    ):
        import torch
        from transformers import T5ForSequenceClassification

        self.torch = torch
        self.model_path = str(model_path)
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.requested_device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max_length
        self.dtype = dtype
        self.tokenizer = _load_t5_tokenizer(self.model_path)
        self.model = T5ForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        if dtype != "float32":
            self.model = self.model.to(getattr(torch, dtype))
        self.model.to(self.device).eval()

    def _run(self, texts: list[str], truncation: bool) -> tuple[list[float], list[int]]:
        torch = self.torch
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=truncation,
            max_length=self.max_length,
        )
        with torch.no_grad():
            out = self.model(
                enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
        logits = out["logits"].float().cpu().reshape(-1).tolist()
        n_tok = enc["attention_mask"].sum(dim=1).tolist()
        return logits, [int(x) for x in n_tok]

    def score_pairs(self, pairs: list[tuple[str, str]], truncation: bool) -> list[PairScore]:
        texts = [pair_text(q, p) for q, p in pairs]
        out: list[PairScore] = []
        # truncation=False (공식 evaluation) 는 512 토큰보다 긴 입력이 그대로 들어가므로
        # 길이가 달라 batch 로 묶을 수 없다 -> 한 쌍씩 (공식과 같음)
        size = self.batch_size if truncation else 1
        for i in range(0, len(texts), size):
            chunk = texts[i : i + size]
            t0 = time.perf_counter()
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
            scores, n_tok = self._run(chunk, truncation)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
            dt = (time.perf_counter() - t0) / len(chunk)
            out += [PairScore(s, n, dt, "model") for s, n in zip(scores, n_tok, strict=True)]
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "device": str(self.device),
            "requested_device": self.requested_device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "weights": weights_fingerprint(self.model_path, full_hash=False),
        }


def _load_t5_tokenizer(path: str):
    """공식은 T5Tokenizer (sentencepiece). 같은 vocab 이면 결과 토큰은 같다."""
    try:
        from transformers import T5Tokenizer

        return T5Tokenizer.from_pretrained(path)
    except Exception:  # noqa: BLE001 - sentencepiece 없음 등 -> AutoTokenizer 로 재시도
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)


class MockScoreBackend:
    """테스트용. fn(query, passage) -> score."""

    name = "mock"

    def __init__(self, fn: Callable[[str, str], float]):
        self.fn = fn
        self.calls: list[tuple[str, str, bool]] = []

    def score_pairs(self, pairs, truncation):
        out = []
        for q, p in pairs:
            self.calls.append((q, p, truncation))
            out.append(PairScore(float(self.fn(q, p)), len(pair_text(q, p).split()), 0.0, "mock"))
        return out

    def describe(self):
        return {"backend": self.name, "note": "mock — 결과를 정확도로 사용하지 않음"}


class CachedScoreBackend:
    """precompute_scores.py 가 만든 jsonl (key, score, n_tokens, compute_seconds) 을 읽는다.

    on_miss: error (기본) | delegate (fallback backend 로 계산)
    """

    name = "cache"

    def __init__(
        self,
        paths: Iterable[str | Path],
        fallback: ScoreBackend | None = None,
        on_miss: str = "error",
    ):
        if on_miss not in ("error", "delegate"):
            raise ValueError("on_miss: error | delegate")
        if on_miss == "delegate" and fallback is None:
            raise ValueError("on_miss=delegate 에는 fallback backend 가 필요")
        self.paths = [str(p) for p in paths]
        self.fallback = fallback
        self.on_miss = on_miss
        self.cache: dict[str, dict] = {}
        self.meta: list[dict] = []
        for p in self.paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("type") == "meta":
                        self.meta.append(row)
                        continue
                    self.cache.setdefault(row["key"], row)
        self.hits = 0
        self.misses = 0

    def score_pairs(self, pairs, truncation):
        out: list[PairScore | None] = []
        missing: list[int] = []
        for i, (q, p) in enumerate(pairs):
            row = self.cache.get(pair_key(q, p, truncation))
            if row is None:
                missing.append(i)
                out.append(None)
                continue
            self.hits += 1
            out.append(
                PairScore(row["score"], row.get("n_tokens"), row.get("compute_seconds"), "cache")
            )
        if missing:
            self.misses += len(missing)
            if self.on_miss == "error":
                q, p = pairs[missing[0]]
                raise KeyError(
                    f"evaluator score cache miss ({len(missing)}개): query={q[:80]!r} "
                    f"passage={p[:80]!r} truncation={truncation}"
                )
            computed = self.fallback.score_pairs([pairs[i] for i in missing], truncation)
            for i, s in zip(missing, computed, strict=True):
                out[i] = s
        return out  # type: ignore[return-value]

    def describe(self):
        return {
            "backend": self.name,
            "paths": self.paths,
            "n_entries": len(self.cache),
            "meta": self.meta,
            "on_miss": self.on_miss,
            "fallback": self.fallback.describe() if self.fallback else None,
            "hits": self.hits,
            "misses": self.misses,
        }


def weights_fingerprint(path: str | Path, full_hash: bool = False) -> dict[str, Any]:
    """가중치 폴더의 파일 목록·크기 (+ 선택: sha256). 가중치 자체는 git 에 넣지 않는다."""
    path = Path(path)
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    files = {}
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        info: dict[str, Any] = {"bytes": f.stat().st_size}
        if full_hash or f.stat().st_size < 5_000_000:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            info["sha256"] = h.hexdigest()
        files[f.name] = info
    return {"path": str(path), "exists": True, "files": files}


# ---------------------------------------------------------------- official decision
def official_doc_flag(score: float, upper: float, lower: float) -> int:
    """CRAG_Inference.py:177-182"""
    if score >= upper:
        return 2
    if score >= lower:
        return 1
    return 0


def official_action(flags: Iterable[int]) -> str:
    """CRAG_Inference.py:187-193 (문서 집합 단위). 문서가 없으면 incorrect."""
    flags = list(flags)
    if 2 in flags:
        return "correct"
    if 1 in flags:
        return "ambiguous"
    return "incorrect"


def process_flag(scores: list[float], n_docs: int, upper: float, lower: float) -> list[int]:
    """공식 process_flag 와 같은 입출력 (점수를 n_docs 개씩 끊어 질문별 2/1/0). 대조 테스트용."""
    out = []
    for i in range(0, len(scores), n_docs):
        flags = [official_doc_flag(s, upper, lower) for s in scores[i : i + n_docs]]
        out.append({"correct": 2, "ambiguous": 1, "incorrect": 0}[official_action(flags)])
    return out


@dataclass
class EvaluatorVerdict:
    scores: list[float]
    flags: list[int]
    action: str
    pairs: list[PairScore]
    upper: float
    lower: float
    wall_seconds: float
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = [asdict(p) for p in self.pairs]
        return d


class OfficialRetrievalEvaluator:
    """공식 Evaluator 의 점수 규칙 (빈 문서 / 짧은 strip) 과 판정을 backend 위에 얹는다."""

    score_scale = "t5_regression_logit"

    def __init__(
        self,
        backend: ScoreBackend,
        upper: float = source.OFFICIAL_THRESHOLDS["popqa"]["upper"],
        lower: float = source.OFFICIAL_THRESHOLDS["popqa"]["lower"],
        normalize: bool = True,
    ):
        if upper < lower:
            raise ValueError(f"upper ({upper}) < lower ({lower})")
        self.backend = backend
        self.upper = upper
        self.lower = lower
        self.normalize = normalize

    def _prep(self, passage: str) -> str:
        return normalize_passage(passage) if self.normalize else passage

    # 문서 평가 (판정용): truncation 없음, 빈 문서 -1.0
    def score_documents(self, query: str, passages: list[str]) -> list[PairScore]:
        prepped = [self._prep(p) for p in passages]
        todo = [i for i, p in enumerate(prepped) if p.strip()]
        out: list[PairScore] = [
            PairScore(source.EMPTY_PASSAGE_SCORE, 0, 0.0, "empty_rule") for _ in prepped
        ]
        if todo:
            got = self.backend.score_pairs([(query, prepped[i]) for i in todo], truncation=False)
            for i, s in zip(todo, got, strict=True):
                out[i] = s
        return out

    # refinement 용 (utils.py:274-299): truncation=True, 4 단어 미만 -1.0, 오류 -1.0
    def score_strips(self, query: str, strips: list[str]) -> list[PairScore]:
        out: list[PairScore] = []
        todo = []
        for i, s in enumerate(strips):
            if len(s.split()) < source.SHORT_STRIP_MIN_WORDS:
                out.append(PairScore(source.SHORT_STRIP_SCORE, None, 0.0, "short_strip_rule"))
            else:
                out.append(PairScore(0.0))
                todo.append(i)
        if todo:
            try:
                got = self.backend.score_pairs([(query, strips[i]) for i in todo], truncation=True)
            except KeyError:
                raise  # cache miss 는 숨기지 않는다
            except Exception as e:  # noqa: BLE001 - 공식 utils.py:288 의 except -> -1.0 과 같음
                got = [
                    PairScore(source.SHORT_STRIP_SCORE, None, None, "error", str(e)[:200])
                ] * len(todo)
            for i, s in zip(todo, got, strict=True):
                out[i] = s
        return out

    def flags(self, scores: list[float]) -> list[int]:
        return [official_doc_flag(s, self.upper, self.lower) for s in scores]

    def evaluate(self, query: str, passages: list[str]) -> EvaluatorVerdict:
        t0 = time.perf_counter()
        pairs = self.score_documents(query, passages)
        scores = [p.score for p in pairs]
        flags = self.flags(scores)
        return EvaluatorVerdict(
            scores=scores,
            flags=flags,
            action=official_action(flags),
            pairs=pairs,
            upper=self.upper,
            lower=self.lower,
            wall_seconds=time.perf_counter() - t0,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "official_t5_evaluator",
            "score_scale": self.score_scale,
            "upper": self.upper,
            "lower": self.lower,
            "normalize_passage": self.normalize,
            "backend": self.backend.describe(),
        }


# ---------------------------------------------------------------- factory
EVALUATOR_PATH_ENV = "CRAG_EVALUATOR_PATH"
# Drive 폴더 `finetuned_t5_evaluator` 의 파일 구성 (2026-11-23 확인)
REQUIRED_WEIGHT_FILES = ("config.json", "model.safetensors", "spiece.model")
OPTIONAL_WEIGHT_FILES = ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


class EvaluatorWeightsError(FileNotFoundError):
    """공식 가중치가 없음. 다른 모델로 대체하지 않는다."""


def resolve_model_path(cfg: dict) -> str:
    import os

    path = cfg.get("model_path") or os.getenv(EVALUATOR_PATH_ENV)
    if not path:
        raise EvaluatorWeightsError(
            f"공식 T5 evaluator 경로가 없음: evaluator.model_path 또는 환경변수 {EVALUATOR_PATH_ENV} "  # noqa: E501
            f"를 지정 (다운로드: {source.EVALUATOR_WEIGHTS_URL}). 다른 모델로 대체하지 않는다."
        )
    return str(Path(path).expanduser())


def check_weights_dir(path: str | Path) -> dict[str, Any]:
    """필수 파일 존재 + config 의 architecture 확인. 문제가 있으면 EvaluatorWeightsError."""
    path = Path(path)
    if not path.is_dir():
        raise EvaluatorWeightsError(f"가중치 폴더 없음: {path}")
    missing = [f for f in REQUIRED_WEIGHT_FILES if not (path / f).is_file()]
    if missing:
        raise EvaluatorWeightsError(
            f"{path}: 필수 파일 누락 {missing} (공식 Drive 폴더 전체를 받을 것)"
        )
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    arch = cfg.get("architectures") or []
    report = {
        "path": str(path),
        "architectures": arch,
        "model_type": cfg.get("model_type"),
        "d_model": cfg.get("d_model"),
        "num_layers": cfg.get("num_layers"),
        "num_labels": len(cfg.get("id2label") or {}) or cfg.get("num_labels"),
        "transformers_version_saved": cfg.get("transformers_version"),
        "optional_missing": [f for f in OPTIONAL_WEIGHT_FILES if not (path / f).is_file()],
        "safetensors_bytes": (path / "model.safetensors").stat().st_size,
    }
    if cfg.get("model_type") != "t5":
        raise EvaluatorWeightsError(f"{path}: model_type={cfg.get('model_type')} (t5 가 아님)")
    if arch and "T5ForSequenceClassification" not in arch:
        report["warning"] = f"architectures={arch} (공식은 T5ForSequenceClassification)"
    return report


def build_backend(cfg: dict) -> ScoreBackend:
    """cfg (`evaluator:`): backend t5 | cache, model_path, device, batch_size, dtype,
    cache_paths, on_miss (error | delegate)."""
    kind = cfg.get("backend", "t5")
    if kind == "t5":
        path = resolve_model_path(cfg)
        check_weights_dir(path)
        return T5EvaluatorBackend(
            path,
            device=cfg.get("device", "cpu"),
            batch_size=cfg.get("batch_size", 1),
            dtype=cfg.get("dtype", "float32"),
        )
    if kind == "cache":
        fallback = None
        if cfg.get("on_miss", "error") == "delegate":
            fallback = build_backend({**cfg, "backend": "t5"})
        return CachedScoreBackend(
            cfg.get("cache_paths") or [], fallback, cfg.get("on_miss", "error")
        )
    raise ValueError(f"evaluator.backend: t5 | cache (got {kind})")


def build_evaluator(cfg: dict, backend: ScoreBackend | None = None) -> OfficialRetrievalEvaluator:
    task = cfg.get("task", "popqa")
    th = source.OFFICIAL_THRESHOLDS[task]
    return OfficialRetrievalEvaluator(
        backend or build_backend(cfg),
        upper=cfg.get("upper", th["upper"]),
        lower=cfg.get("lower", th["lower"]),
        normalize=cfg.get("normalize_passage", True),
    )


def tokenization_check(tokenizer, spiece_path: str, texts: list[str]) -> dict[str, Any]:
    """현재 transformers 토크나이저 id == sentencepiece 직접 인코딩 + </s> 인지.

    원본 (transformers 4.36 slow T5Tokenizer) 과 같은 토큰화인지의 대리 검사.
    버전 차이로 토큰이 달라지면 점수가 달라진다.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=spiece_path)
    rows = []
    for t in texts:
        got = tokenizer(t)["input_ids"]
        want = sp.encode(t) + [sp.eos_id()]
        rows.append({"text": t[:80], "equal": got == want, "n_got": len(got), "n_want": len(want)})
    return {"all_equal": all(r["equal"] for r in rows), "rows": rows}
