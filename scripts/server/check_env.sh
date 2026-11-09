#!/usr/bin/env bash
# 서버 환경 점검 (읽기 전용). 아무것도 설치·실행·변경하지 않는다. GPU 메모리도 사용하지 않는다.
#
#   bash scripts/server/check_env.sh            # -> ~/crag_logs/env_<timestamp>.txt
#
# 결과 파일을 확인한 뒤 공유해 주세요. 토큰 / 키 값은 출력하지 않습니다 (존재 여부만).
set -u
OUT_DIR="${HOME}/crag_logs"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/env_$(date +%Y%m%d_%H%M%S).txt"

section() { printf '\n===== %s =====\n' "$1"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 || printf '(exit %s)\n' "$?"; }
have() { command -v "$1" >/dev/null 2>&1; }

{
section "OS / host"
run date -Is
run hostname
run cat /etc/os-release
run uname -r
run whoami
run id

section "CPU / memory / NUMA"
run nproc
lscpu 2>/dev/null | grep -iE "model name|socket|numa|thread|core" || true
run free -g
have numactl && run numactl -H

section "GPU (PCI)"
lspci -nn 2>/dev/null | grep -i nvidia || echo "(lspci 결과 없음)"

section "GPU (nvidia-smi)"
if have nvidia-smi; then
  run nvidia-smi
  run nvidia-smi -L
  run nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version,compute_cap,compute_mode,mig.mode.current --format=csv
  run nvidia-smi topo -m
  section "GPU processes (다른 사용자 점유 확인)"
  run nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && ps -o user=,pid=,etime=,comm= -p $(echo "$pids" | paste -sd,) 2>/dev/null
else
  echo "nvidia-smi 없음"
fi

section "CUDA toolkit"
have nvcc && run nvcc --version || echo "nvcc 없음 (vLLM wheel 은 자체 CUDA runtime 포함이라 필수는 아님)"
ls -d /usr/local/cuda* 2>/dev/null || echo "/usr/local/cuda* 없음"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

section "Python / 패키지 도구"
for c in python3 python3.11 python3.12 uv conda pip3 git tmux screen; do
  if have "$c"; then printf '%-10s %s  ' "$c" "$(command -v "$c")"; "$c" --version 2>&1 | head -1; else echo "$c: 없음"; fi
done

section "Container / 작업 스케줄러"
for c in docker podman apptainer singularity sinfo squeue; do have "$c" && echo "$c: $(command -v "$c")" || echo "$c: 없음"; done
groups

section "Disk / quota / limits"
run df -h "$HOME"
quota -s 2>/dev/null || echo "(quota 정보 없음)"
du -sh "$HOME/.cache" 2>/dev/null || true
echo "ulimit -n: $(ulimit -n)  ulimit -u: $(ulimit -u)"

section "Network (외부 접근)"
for u in https://huggingface.co https://pypi.org/simple/ https://api.tavily.com https://download.pytorch.org; do
  printf '%-36s ' "$u"; curl -s -o /dev/null -m 10 -w '%{http_code}\n' "$u" || echo "fail"
done
if [ -n "${HF_TOKEN:-}" ]; then
  printf 'HF Llama-3.1-8B-Instruct 접근 (gated): '
  curl -s -o /dev/null -m 10 -w '%{http_code}\n' -H "Authorization: Bearer ${HF_TOKEN}" \
    https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct
else
  echo "HF_TOKEN: <unset>"; [ -f "$HOME/.cache/huggingface/token" ] && echo "~/.cache/huggingface/token: 있음"
fi

section "기존 서비스 / 포트"
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E ':(8000|8001|11434|11435)\b' || echo "8000/8001/11434/11435 사용 중인 프로세스 없음"
have ollama && run ollama --version || echo "ollama 없음"
pgrep -a -f "vllm|ollama" 2>/dev/null || echo "실행 중인 vllm/ollama 프로세스 없음"

section "Logged-in users"
run who
} | tee "$OUT"

echo
echo "저장: $OUT"
