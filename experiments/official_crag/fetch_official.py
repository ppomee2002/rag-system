"""공식 CRAG 저장소 (고정 commit) 와 Evaluator 가중치 준비 (서버에서 사용자가 실행).

    # 1. 공식 저장소를 third_party/CRAG 에 고정 commit 으로 받기 (gitignore 대상)
    uv run python experiments/official_crag/fetch_official.py repo

    # 2. 가중치: Google Drive 폴더 `finetuned_t5_evaluator` 를 받는다 (둘 중 하나)
    #    a) 브라우저로 받아 서버로 복사 -> models/crag_evaluator/finetuned_t5_evaluator/
    #    b) gdown 사용 (별도 설치: uv pip install gdown)
    uv run python experiments/official_crag/fetch_official.py weights --out models/crag_evaluator

    # 3. 파일 구성 검사 + sha256 기록 (results/official_crag/provenance/)
    uv run python experiments/official_crag/fetch_official.py verify --weights <폴더> --hash

가중치·공식 저장소는 git 에 커밋하지 않는다 (.gitignore: third_party/, models/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.crag_official import source
from rag.crag_official.evaluator import check_weights_dir, weights_fingerprint

ROOT = Path(__file__).resolve().parents[2]
REPO_DIR = ROOT / "third_party" / "CRAG"
PROV_DIR = ROOT / "results" / "official_crag" / "provenance"


def _git(*args, cwd=None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_repo(dest: Path = REPO_DIR) -> dict:
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", source.OFFICIAL_REPO_URL, str(dest))
    _git("fetch", "--quiet", "origin", cwd=dest)
    _git("checkout", "--quiet", source.OFFICIAL_COMMIT, cwd=dest)
    head = _git("rev-parse", "HEAD", cwd=dest)
    if head != source.OFFICIAL_COMMIT:
        raise SystemExit(f"commit 불일치: {head} != {source.OFFICIAL_COMMIT}")
    dirty = _git("status", "--porcelain", cwd=dest)
    need = [
        "data/popqa/test_popqa.txt",
        "data/popqa/sources",
        "data/popqa/ref/correct",
        "data/popqa/ref/incorrect",
        "data/popqa/ref/ambiguous",
        "scripts/CRAG_Inference.py",
    ]
    missing = [p for p in need if not (dest / p).exists()]
    info = {
        "repo": source.OFFICIAL_REPO_URL,
        "commit": head,
        "path": str(dest),
        "dirty": bool(dirty),
        "license_file": any((dest / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
        "missing": missing,
    }
    if missing:
        raise SystemExit(f"공식 저장소 파일 누락: {missing}")
    return info


def fetch_weights(out: Path) -> dict:
    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "gdown 이 없음. `uv pip install gdown` 후 재실행하거나 브라우저로 받으세요:\n"
            f"  {source.EVALUATOR_WEIGHTS_URL}\n"
            f"  -> {out}/finetuned_t5_evaluator/ 에 6개 파일 전부"
        ) from None
    out.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(url=source.EVALUATOR_WEIGHTS_URL, output=str(out), quiet=False)
    if not files:
        raise SystemExit("gdown 다운로드 실패 (Drive 할당량/권한). 브라우저로 받으세요.")
    return {"downloaded": files}


def verify(weights: Path, full_hash: bool) -> dict:
    report = check_weights_dir(weights)  # 실패 시 EvaluatorWeightsError
    report["fingerprint"] = weights_fingerprint(weights, full_hash=full_hash)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("repo")
    w = sub.add_parser("weights")
    w.add_argument("--out", default="models/crag_evaluator")
    v = sub.add_parser("verify")
    v.add_argument("--weights", required=True)
    v.add_argument("--hash", action="store_true", help="model.safetensors 전체 sha256 (수십 초)")
    a = ap.parse_args(argv)
    if a.cmd == "repo":
        info = fetch_repo()
    elif a.cmd == "weights":
        info = fetch_weights(Path(a.out))
    else:
        info = verify(Path(a.weights), a.hash)
    info["timestamp_utc"] = datetime.now(UTC).isoformat()
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    path = PROV_DIR / f"{a.cmd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"saved -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
