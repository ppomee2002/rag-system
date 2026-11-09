"""실행 메타데이터와 결과 폴더.

모든 실험 run 은 다음을 남긴다 (`results/<experiment>/<timestamp>/`):
  config.yaml     사용한 config (deep-merge 결과)
  run_meta.json   git commit / dirty, python·패키지 버전, host,
                  GPU 상태 (다른 사용자 프로세스 포함),
                  CUDA_VISIBLE_DEVICES, 엔진 정보 (버전, 모델, quantization), seed, config hash
  queries.jsonl   query 단위 결과 (+ 호출 단위 `calls`)

이미 있는 폴더·파일은 덮어쓰지 않는다 (FileExistsError).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]

PACKAGES = (
    "rag-system",
    "numpy",
    "faiss-cpu",
    "sentence-transformers",
    "torch",
    "transformers",
    "langgraph",
    "vllm",
    "tavily-python",
    "anthropic",
)
ENV_PREFIXES = ("CUDA_", "VLLM_", "OLLAMA_", "HF_", "OMP_", "TOKENIZERS_")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(root: Path = ROOT) -> dict:
    status = _run(["git", "status", "--porcelain"], root)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else status.splitlines()[:50],
    }


def package_versions(names=PACKAGES) -> dict[str, str | None]:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def safe_env() -> dict[str, str]:
    """실험 조건에 영향을 주는 환경변수. 값에 비밀이 있을 수 있는 이름은 존재 여부만 남긴다."""
    out = {}
    for k, v in sorted(os.environ.items()):
        if not k.startswith(ENV_PREFIXES):
            continue
        out[k] = "<set>" if any(m in k.upper() for m in SECRET_MARKERS) else v
    return out


GPU_FIELDS = (
    "index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,"
    "compute_mode,temperature.gpu,power.draw"
)


def gpu_snapshot() -> dict[str, Any]:
    """nvidia-smi 기준 GPU 상태와 GPU 를 쓰는 프로세스 (공유 서버 간섭 확인용)."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    gpus = _run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"])
    apps = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    keys = GPU_FIELDS.split(",")
    return {
        "available": gpus is not None,
        "gpus": [
            dict(zip(keys, (x.strip() for x in line.split(",")), strict=False))
            for line in (gpus or "").splitlines()
            if line.strip()
        ],
        "compute_apps": [line.strip() for line in (apps or "").splitlines() if line.strip()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def collect_run_metadata(
    cfg: dict, engines: dict[str, Any] | None = None, extra: dict | None = None
) -> dict[str, Any]:
    """engines: {"main": engine, ...} — 각 engine.describe() 를 기록한다."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "experiment": cfg.get("experiment"),
        "seed": cfg.get("seed"),
        "config_hash": config_hash(cfg),
        "argv": sys.argv,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "git": git_info(),
        "packages": package_versions(),
        "env": safe_env(),
        "gpu": gpu_snapshot(),
        "engines": {k: e.describe() for k, e in (engines or {}).items()},
        **(extra or {}),
    }


def create_run_dir(
    experiment: str,
    output_dir: str | Path = "results",
    run_id: str | None = None,
    root: Path = ROOT,
) -> Path:
    """results/<experiment>/<run_id or timestamp>/ 생성. 이미 있으면 FileExistsError."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir)
    if not base.is_absolute():
        base = root / base
    path = base / experiment / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:  # "x": 이미 있으면 실패
        f.write(text)


def write_run_files(run_dir: Path, cfg: dict, meta: dict) -> None:
    write_new(run_dir / "config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    write_new(
        run_dir / "run_meta.json", json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )


class JsonlWriter:
    """한 줄씩 append + flush (중단돼도 앞부분이 남는다). 여러 thread 에서 써도 안전."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
