"""vLLM 서버가 뜬 뒤, 프로필의 추정치를 실제 로그·GPU 메모리·API 응답으로 검증한다.

    uv run python scripts/server/verify_launch.py --launch-meta ~/crag_logs/vllm_<...>_launch.json
    (또는 vLLM venv 의 python. 표준 라이브러리만 사용)

결과: <launch_meta 와 같은 이름>_verify.json (덮어쓰지 않음)
  log      : Model loading took X GiB, Available KV cache memory, KV cache size N tokens,
             Maximum concurrency (vLLM 0.33.0 로그 문구 기준), 오류 줄 (OOM, NCCL, Traceback)
  gpus     : nvidia-smi 메모리 사용량 (launch 기록의 GPU uuid 기준)
  api      : /health, /v1/models (served model, max_model_len)
  compare  : estimates 대비 실제 값과 비율. 추정치는 확정값이 아니며 실험 설정은 실제 값으로 정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RE_LOADING = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
RE_WEIGHTS = re.compile(r"Loading weights took ([\d.]+) seconds")
RE_KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
RE_KV_SIZE = re.compile(
    r"(\S+) KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
    r"([\d.]+)x"
)
RE_ERROR = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|No available memory for the cache blocks|"
    r"NCCL (?:error|WARN)|Traceback \(most recent call last\)|ValueError:|RuntimeError:)"
)
STARTUP = "Application startup complete"


def parse_vllm_log(text: str) -> dict:
    loading = [(float(a), float(b)) for a, b in RE_LOADING.findall(text)]
    kv_size = RE_KV_SIZE.search(text)
    errors = [ln.strip()[:300] for ln in text.splitlines() if RE_ERROR.search(ln)]
    return {
        "startup_complete": STARTUP in text,
        "model_loading_gib": [g for g, _ in loading],  # worker (TP rank) 별
        "model_loading_s": [s for _, s in loading],
        "loading_weights_s": [float(x) for x in RE_WEIGHTS.findall(text)],
        "available_kv_cache_gib": [float(x) for x in RE_KV_MEM.findall(text)],
        "kv_cache_pool": kv_size.group(1) if kv_size else None,
        "kv_cache_tokens": int(kv_size.group(2).replace(",", "")) if kv_size else None,
        "max_model_len_logged": int(kv_size.group(3).replace(",", "")) if kv_size else None,
        "max_concurrency": float(kv_size.group(4)) if kv_size else None,
        "error_lines": errors[:30],
    }


def compare(estimates: dict | None, parsed: dict, gpus: list[dict]) -> dict:
    est = estimates or {}

    def item(key, actual):
        e = est.get(key)
        ratio = None if (e in (None, 0) or actual is None) else round(actual / e, 3)
        return {"estimate": e, "actual": actual, "actual_over_estimate": ratio}

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "weights_gib_per_gpu": item("weights_gib_per_gpu", avg(parsed["model_loading_gib"])),
        "kv_cache_gib_per_gpu": item("kv_cache_gib_per_gpu", avg(parsed["available_kv_cache_gib"])),
        "kv_cache_tokens": item("kv_cache_tokens", parsed["kv_cache_tokens"]),
        "max_concurrency_at_max_model_len": item(
            "max_concurrency_at_max_model_len", parsed["max_concurrency"]
        ),
        "gpu_memory_used_mib": [g.get("memory_used_mib") for g in gpus],
        "note": "estimate 는 사양 기반 추정. 실험 설정은 actual 로 결정",
    }


def gpu_memory(uuids: list[str]) -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []

    def query(fields: str) -> str:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.stdout if r.returncode == 0 else ""

    rows = []
    for line in query(
        "uuid,index,memory.used,memory.total,utilization.gpu,power.draw"
    ).splitlines():
        p = [x.strip() for x in line.split(",")]
        if p[0] in uuids:
            rows.append(
                {
                    "uuid": p[0],
                    "index": p[1],
                    "memory_used_mib": int(p[2]),
                    "memory_total_mib": int(p[3]),
                    "util_percent": p[4],
                    "power_w": p[5],
                }
            )
    # 드라이버에 따라 필드 이름이 다름 (clocks_event_reasons / clocks_throttle_reasons)
    for field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        out = query(f"uuid,{field}")
        if out:
            reasons = dict(line.split(",", 1) for line in out.splitlines() if "," in line)
            for r in rows:
                r["clock_event_reasons"] = reasons.get(r["uuid"], "").strip()
            break
    return rows


def api_check(base: str, key: str | None) -> dict:
    root = base.removesuffix("/v1")
    out = {}
    for name, url, auth in (
        ("health", f"{root}/health", False),
        ("models", f"{base}/models", True),
        ("version", f"{root}/version", False),
    ):
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {key}"} if auth and key else {}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read().decode("utf-8", errors="replace")
                out[name] = {
                    "status": r.status,
                    "body": json.loads(body) if body.startswith(("{", "[")) else body[:200],
                }
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-meta", required=True)
    ap.add_argument("--base-url", default=None, help="기본: http://127.0.0.1:<port>/v1")
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args(argv)
    meta_path = Path(a.launch_meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    log_text = Path(meta["log"]).read_text(encoding="utf-8", errors="replace")
    parsed = parse_vllm_log(log_text)
    uuids = [g["uuid"] for g in (meta.get("gpus") or {}).get("selected", [])]
    gpus = gpu_memory(uuids)
    server = meta["profile"]["server"]
    base = a.base_url or f"http://127.0.0.1:{server['port']}/v1"
    api = {} if a.no_api else api_check(base, os.environ.get("VLLM_API_KEY"))
    served = (((api.get("models") or {}).get("body") or {}).get("data") or [{}])[0]
    checks = {
        "startup_complete": parsed["startup_complete"],
        "no_error_lines": not parsed["error_lines"],
        "served_model_matches": served.get("id") == server["model"] if served else None,
        "max_model_len_matches": served.get("max_model_len") == server["max_model_len"]
        if served
        else None,
        "kv_fits_one_max_len_request": None
        if parsed["max_concurrency"] is None
        else parsed["max_concurrency"] >= 1.0,
        "n_workers_logged": len(parsed["model_loading_gib"]),
        "expected_workers": server.get("tensor_parallel_size", 1),
    }
    report = {
        "launch_meta": str(meta_path),
        "profile": meta["profile"]["name"],
        "smoke": meta.get("smoke"),
        "scheduling_policy": meta.get("scheduling_policy"),
        "checks": checks,
        "log": parsed,
        "gpus": gpus,
        "api": api,
        "compare": compare(meta["profile"].get("estimates"), parsed, gpus),
    }
    out = meta_path.with_name(meta_path.name.replace("_launch.json", "_verify.json"))
    with open(out, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        json.dumps({"checks": checks, "compare": report["compare"]}, indent=2, ensure_ascii=False)
    )
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
