"""공식 T5 Evaluator 단독 smoke test (서버에서 사용자가 실행, vLLM 불필요).

    CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
        python experiments/official_crag/smoke_evaluator.py \
        --weights $CRAG_EVALUATOR_PATH --device cuda:0
    uv run --extra crag --extra official \
        python experiments/official_crag/smoke_evaluator.py --device cpu

확인 항목 (결과 JSON: results/official_crag/smoke/evaluator/<ts>.json)
  1. 환경: torch / transformers / sentencepiece / safetensors 버전, CUDA, GPU 여유 메모리
  2. 가중치 폴더 구성 (config.json, model.safetensors, spiece.model), config 의 architecture
  3. 로딩 시간, 로딩 후 / 추론 최대 GPU 메모리 (torch.cuda.max_memory_allocated)
  4. 공식 PopQA 첫 N 문항 (third_party/CRAG/data/popqa/test_popqa.txt) 의 문서별 점수, 판정, 시간
  5. 같은 입력 반복 시 점수 일치 (결정성)
  6. 토큰화 동등성: 현재 transformers 토크나이저 id == sentencepiece 직접 인코딩 + </s>
     (원본 transformers 4.36 slow T5Tokenizer 와 같은 토큰인지의 대리 검사)
점수·판정의 "정답" 은 공식 저장소에 공개되어 있지 않으므로 이 스크립트는 값을 검증하지 않는다.
분포가 상식적인지 (logit 대략 -1~+1 근처, 빈 문서 -1.0) 와 오류 없이 도는지만 본다.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

from rag.crag_official import source
from rag.crag_official.evaluator import (
    T5EvaluatorBackend,
    build_evaluator,
    check_weights_dir,
    resolve_model_path,
    tokenization_check,
)
from rag.crag_official.official_data import load_official_queries

ROOT = Path(__file__).resolve().parents[2]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ("torch", "transformers", "sentencepiece", "safetensors", "tokenizers", "protobuf"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def cuda_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    devs = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        devs.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_gib": round(free / 2**30, 2),
                "total_gib": round(total / 2**30, 2),
            }
        )
    return {"available": True, "devices": devs, "torch_cuda": torch.version.cuda}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="기본: 환경변수 CRAG_EVALUATOR_PATH")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--official-repo", default=str(ROOT / "third_party" / "CRAG"))
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args(argv)

    report: dict = {"versions": versions(), "cuda_before": cuda_info(), "device": a.device}
    out_dir = ROOT / "results" / "official_crag" / "smoke" / "evaluator"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        path = resolve_model_path({"model_path": a.weights})
        report["weights"] = check_weights_dir(path)
        saved = report["weights"].get("transformers_version_saved")
        report["note_transformers"] = (
            f"가중치 저장 버전 {saved}, 현재 {report['versions']['transformers']}. "
            "로딩 경고 (unexpected/missing keys) 가 있으면 아래 load_warnings 확인"
        )
        import torch
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_warning()
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        backend = T5EvaluatorBackend(path, device=a.device, dtype=a.dtype)
        report["load_seconds"] = time.perf_counter() - t0
        report["n_params"] = sum(p.numel() for p in backend.model.parameters())
        report["param_dtype"] = str(next(backend.model.parameters()).dtype)
        if a.device.startswith("cuda"):
            report["gpu_mem_after_load_gib"] = torch.cuda.memory_allocated() / 2**30
        ev = build_evaluator({"task": "popqa"}, backend=backend)
        # transformers 버전이 바뀌어도 원본 (4.36 slow T5Tokenizer) 과 같은 토큰인지
        report["tokenization_check"] = tokenization_check(
            backend.tokenizer,
            str(Path(path) / "spiece.model"),
            [
                "What is Henry Feilden's occupation? [SEP] Henry Master Feilden was an English politician.",  # noqa: E501
                "In what city was Billy Carlson born? [SEP] Kärnten – Ötztal 1918 (née) “quoted”",
            ],
        )

        test = Path(a.official_repo) / "data" / "popqa" / "test_popqa.txt"
        if test.exists():
            queries = load_official_queries(test)[: a.n_queries]
            src = "official test_popqa.txt"
        else:
            queries = None
            src = "builtin (공식 저장소 없음: fetch_official.py repo)"
        rows = []
        first_raw = None
        if queries:
            for q in queries:
                v = ev.evaluate(q.question, q.passages)
                first_raw = first_raw or v.scores
                rows.append(
                    {
                        "index": q.index,
                        "question": q.question,
                        "scores": [round(s, 4) for s in v.scores],
                        "flags": v.flags,
                        "action": v.action,
                        "seconds": round(v.wall_seconds, 3),
                        "n_tokens": [p.n_tokens for p in v.pairs],
                    }
                )
            again = ev.evaluate(queries[0].question, queries[0].passages).scores
            report["determinism_max_abs_diff"] = max(
                abs(x - y) for x, y in zip(again, first_raw, strict=True)
            )
        else:
            v = ev.evaluate(
                "What is Henry Feilden's occupation?",
                [
                    "Henry Master Feilden (1818 - 1875) was an English Conservative Party politician.",  # noqa: E501
                    "Bernard Feilden (1919-2008), British conservation architect",
                    "",
                ],
            )
            rows.append({"scores": v.scores, "flags": v.flags, "action": v.action})
        report["input_source"] = src
        report["results"] = rows
        report["thresholds"] = source.OFFICIAL_THRESHOLDS["popqa"]
        if a.device.startswith("cuda"):
            report["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["cuda_after"] = cuda_info()
        report["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - 보고서에 남기고 실패 코드로 종료
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        hint = []
        msg = str(e)
        if "out of memory" in msg.lower():
            hint.append("CUDA OOM: --device cpu 또는 vLLM 과 다른 GPU (README 'GPU 메모리')")
        if "sentencepiece" in msg.lower() or "SentencePiece" in msg:
            hint.append("uv sync --extra crag --extra official (sentencepiece)")
        if "T5ForSequenceClassification" in msg:
            hint.append("transformers 버전 확인 (README '버전 충돌')")
        report["hints"] = hint
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str)[:6000])
    print(f"saved -> {out}", file=sys.stderr)
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
