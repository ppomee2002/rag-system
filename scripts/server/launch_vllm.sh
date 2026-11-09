#!/usr/bin/env bash
# vLLM OpenAI 호환 서버 실행 템플릿. 모든 값은 환경변수로 지정한다 (기본값은 확정이 아니라 예시).
# tmux 안에서 실행 권장:  tmux new -s vllm
#
#   export CUDA_VISIBLE_DEVICES=0            # 필수: 비어 있는 GPU 를 확인 후 지정
#   export VLLM_API_KEY=$(openssl rand -hex 16)   # 필수: 클라이언트도 같은 환경변수 사용
#   MODEL=meta-llama/Llama-3.1-8B-Instruct TP=1 bash scripts/server/launch_vllm.sh
#
# API key 는 명령줄 인자로 넘기지 않는다 (공유 서버에서 ps 로 다른 사용자에게 보임). vLLM 은 VLLM_API_KEY
# 환경변수를 읽는다 — 버전에 따라 다를 수 있으므로 probe_engine.py 의 auth 점검으로 확인할 것.
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES 를 지정하세요 (nvidia-smi 로 빈 GPU 확인)}"
: "${VLLM_API_KEY:?VLLM_API_KEY 를 환경변수로 지정하세요}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                                   # tensor parallel (GPU 수와 맞춤)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"           # 기존 num_ctx 와 동일
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"             # 공유 서버: 기본 0.9 대신 명시
SCHED_POLICY="${SCHED_POLICY:-fcfs}"             # fcfs | priority
QUANT="${QUANT:-}"                               # 예: awq, gptq, fp8 (비우면 모델 기본 dtype)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"                 # SYS-2x cb_off: 1
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # 예: --no-enable-prefix-caching --generation-config vllm

if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${PORT}\b"; then
  echo "포트 ${PORT} 사용 중. PORT=... 로 바꾸세요." >&2; exit 1
fi

LOG_DIR="${HOME}/crag_logs"; mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/vllm_${TS}.log"
META="${LOG_DIR}/vllm_${TS}_launch.json"

ARGS=(serve "$MODEL" --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP"
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
      --scheduling-policy "$SCHED_POLICY" --seed 0)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")
[ -n "$MAX_NUM_SEQS" ] && ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

python3 - "$META" "$LOG" "${ARGS[@]}" <<'PY'
import json, os, subprocess, sys
meta, log, *args = sys.argv[1:]
def run(c):
    try: return subprocess.run(c, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"error: {e}"
info = {
    "argv": ["vllm", *args], "log": log,
    "env": {k: ("<set>" if "KEY" in k or "TOKEN" in k else v) for k, v in os.environ.items()
            if k.startswith(("CUDA_", "VLLM_", "HF_"))},
    "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version", "--format=csv"]),
    "gpu_apps": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]),
    "vllm_version": run(["python3", "-c", "import vllm; print(vllm.__version__)"]),
    "pip_freeze": run(["python3", "-m", "pip", "freeze"]).splitlines(),
}
open(meta, "x").write(json.dumps(info, indent=2))
print("launch metadata:", meta)
PY

echo "log: $LOG"
echo "vllm ${ARGS[*]}"
vllm "${ARGS[@]}" 2>&1 | tee "$LOG"
