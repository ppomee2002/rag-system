"""vLLM 설치 검증 (vLLM 전용 venv 에서 실행). 레포 코드를 import 하지 않는다.

    CUDA_VISIBLE_DEVICES=<GPU> python scripts/server/check_vllm_install.py [--out report.json]

- 모델을 로드하지 않는다. CUDA_VISIBLE_DEVICES 가 설정된 경우에만 torch 로 장치 정보를 읽는다
  (CUDA 초기화 시 장치마다 수백 MB 의 context 가 잠깐 생길 수 있으므로,
  공유 서버에서 다른 사람의 GPU 를 건드리지 않도록 사용할 GPU 만 보이게 한 뒤 실행).
- vLLM 기능 (priority scheduling, prefix caching, max_num_seqs, request priority 필드) 이
  이 버전에 있는지 코드 수준에서 확인한다. 실제 동작은 서버를 띄운 뒤 probe_engine.py 로 확인한다.
"""

import argparse
import dataclasses
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys

MODEL_GB = {  # Llama-3.1-8B 가중치 대략치 (KV cache 제외)
    "bf16/fp16": 16.1,
    "fp8 (weight)": 8.5,
    "int4 (AWQ/GPTQ)": 5.7,
}


def try_(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def torch_info():
    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": None,
        "devices": [],
    }
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        info["skipped"] = "CUDA_VISIBLE_DEVICES 미설정: 장치 초기화 생략 (공유 서버 보호)"
        return info
    info["cuda_available"] = torch.cuda.is_available()
    if not info["cuda_available"]:
        return info
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = (p.major, p.minor)
        total = p.total_memory / 2**30
        info["devices"].append(
            {
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(total, 2),
                "multi_processor_count": p.multi_processor_count,
                "bf16": cap >= (8, 0),
                "fp8_native": cap >= (8, 9),  # Ampere(8.0/8.6) 는 fp8 weight-only (Marlin) 만 가능
                "weights_fit_hint": {k: v < total * 0.85 for k, v in MODEL_GB.items()},
            }
        )
    return info


def vllm_info():
    vllm = importlib.import_module("vllm")
    out = {"vllm": getattr(vllm, "__version__", None)}
    from vllm.engine.arg_utils import EngineArgs

    names = {f.name for f in dataclasses.fields(EngineArgs)}
    out["engine_args"] = {
        k: k in names
        for k in (
            "scheduling_policy",
            "enable_prefix_caching",
            "max_num_seqs",
            "max_num_batched_tokens",
            "tensor_parallel_size",
            "quantization",
            "generation_config",
            "gpu_memory_utilization",
            "enable_chunked_prefill",
        )
    }
    defaults = EngineArgs(model="dummy")
    out["engine_arg_defaults"] = {
        k: repr(getattr(defaults, k, None))
        for k in (
            "scheduling_policy",
            "enable_prefix_caching",
            "max_num_seqs",
            "enable_chunked_prefill",
            "generation_config",
        )
    }

    def request_fields():
        try:  # vLLM 0.33.0 경로
            from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        except ImportError:  # 이전 버전 경로
            from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        fields = ChatCompletionRequest.model_fields
        doc = (fields["priority"].description or "") if "priority" in fields else ""
        out["priority_field_description"] = doc[:300]
        return {
            k: k in fields
            for k in (
                "priority",
                "seed",
                "stream_options",
                "response_format",
                "guided_json",
                "structured_outputs",
                "request_id",
            )
        }

    out["chat_request_fields"] = try_(request_fields)
    return out


def cli_help():
    exe = shutil.which("vllm")
    if exe is None:
        return {"error": "vllm CLI 없음"}
    r = subprocess.run([exe, "serve", "--help"], capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    keys = (
        "--scheduling-policy",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "--generation-config",
        "--api-key",
        "--served-model-name",
        "--enable-request-id-headers",
        "--disable-log-requests",
        "--seed",
    )
    return {k: k in text for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-cli", action="store_true", help="vllm serve --help 호출 생략")
    a = ap.parse_args()
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": try_(torch_info),
        "vllm": try_(vllm_info),
        "cli": {"skipped": True} if a.skip_cli else try_(cli_help),
        "model_weight_gib_estimates": MODEL_GB,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
