#!/usr/bin/env bash
# 스케줄링 검증 실험 (SV): 실험 A (경로별 작업량) / B (gate 효과) / C (상태 정보 기여).
# 사용자가 GPU 서버 (~/RAG = /home/rag01/RAG) 에서 실행한다. Claude 는 실행하지 않는다.
#
#   bash experiments/crag_system/run_sched_validation.sh <stage>
#
# stage (순서대로):
#   train      sequential, train split -> cost model 학습 trace + fit (wall 기준)
#   seq        sequential, eval split, 실제 Tavily 기록 -> 실험 A 데이터 + live 용 web 기록
#   calib      PRE-K: K 별 처리량 (eval split, web replay, latency 0) -> K*, mu_max
#   workloads  eval split Poisson workload (RHOS x SEEDS), capacity = mu_max
#   live       같은 workload 로 POLICIES 를 K=K_MAIN 에서 실행 (실험 B + C)
#   ksweep     (우선순위 2) KSWEEP_POLICIES 를 KSWEEP_KS 에서, KSWEEP_RHO / seed 0
#   analyze    실험 A + workload 별 B/C 비교 리포트 (analyze_sched_validation.py)
#
# 환경 변수 (기본값)
#   VLLM_API_KEY      필수 (export VLLM_API_KEY=$(cat ~/.vllm_key))
#   TAVILY_API_KEY    train / seq 에 필요 (.env 로 읽어도 됨)
#   TAG=SV1           exp_id 접두사 (PRE-$TAG-*). 다시 돌릴 때는 TAG 를 바꾼다 (덮어쓰기 금지)
#   PROFILE=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
#   LAUNCH_META=      launch_vllm.py 의 *_launch.json (지정하면 client 설정과 대조)
#   LIMIT=503         dataset.limit (warmup 3 제외 500 -> train 150 / eval 350)
#   RHOS="0.8 0.95"   SEEDS="0 1"   K_MAIN=  (비우면 calib 의 K*)   MU=  (비우면 calib 의 mu_max)
#   POLICIES="none fcfs fcfs_query sjf_pred srpt_nopath srpt_est"
#   KSWEEP_KS="1 2 4 8"   KSWEEP_POLICIES="fcfs srpt_est"   KSWEEP_RHO=0.8
#   RESET_PREFIX_CACHE=0  1 이면 run 시작 전 prefix cache 초기화 (서버를 --dev-mode 로 띄운 경우만)
#   GPU_LOG=1         run 마다 nvidia-smi 사용률 기록 (results/.../<run>.gpu.csv)
#   DRY_RUN=0         1 이면 명령만 출력
set -euo pipefail

cd "$(dirname "$0")/../.."
TAG=${TAG:-SV1}
PROFILE=${PROFILE:-experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml}
LAUNCH_META=${LAUNCH_META:-}
LIMIT=${LIMIT:-503}
RHOS=${RHOS:-"0.8 0.95"}
SEEDS=${SEEDS:-"0 1"}
POLICIES=${POLICIES:-"none fcfs fcfs_query sjf_pred srpt_nopath srpt_est"}
KSWEEP_KS=${KSWEEP_KS:-"1 2 4 8"}
KSWEEP_POLICIES=${KSWEEP_POLICIES:-"fcfs srpt_est"}
KSWEEP_RHO=${KSWEEP_RHO:-0.8}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-0}
GPU_LOG=${GPU_LOG:-1}
DRY_RUN=${DRY_RUN:-0}

PRE=results/crag_system/pre
TRAIN_DIR=$PRE/PRE-$TAG-TRAIN/train
SEQ_DIR=$PRE/PRE-$TAG-A/eval_seq
LIVE_ROOT=$PRE/PRE-$TAG-LIVE
COST_MODEL=results/crag_system/cost_models/${TAG}_train_wall.json
STATE=$PRE/PRE-$TAG-state.env   # calib 결과 (K_STAR, MU_MAX)

COMMON=(--overlay "$PROFILE" --set run.purpose=pre --set "dataset.limit=$LIMIT")
[[ -n "$LAUNCH_META" ]] && COMMON+=(--set "llm.launch_meta=$LAUNCH_META")
REPLAY=(
  --set crag.web_search.provider=replay
  --set "crag.web_search.replay_from_runs=['$SEQ_DIR']"
  --set crag.web_search.replay_key=query_id
  --set crag.web_search.replay_on_miss=empty
)

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

need() { [[ -e "$1" ]] || { echo "없음: $1 ($2 단계를 먼저 실행)" >&2; exit 1; }; }

load_state() {
  if [[ -f "$STATE" ]]; then source "$STATE"; fi
  K_MAIN=${K_MAIN:-${K_STAR:-}}
  MU=${MU:-${MU_MAX:-}}
  if [[ -z "$K_MAIN" || -z "$MU" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then K_MAIN=${K_MAIN:-4}; MU=${MU:-0.5}; return; fi
    echo "K_MAIN / MU 가 없음: calib 단계를 먼저 실행하거나 K_MAIN=, MU= 로 지정" >&2
    exit 1
  fi
}

# 같은 run 을 nvidia-smi 기록과 함께 실행. $1 = run 폴더 (= <exp root>/<run_id>)
with_gpu_log() {
  local dir=$1; shift
  if [[ "$DRY_RUN" == 1 || "$GPU_LOG" != 1 ]] || ! command -v nvidia-smi >/dev/null; then
    run "$@"; return
  fi
  local log="$dir.gpu.csv"
  [[ -e "$log" ]] && { echo "이미 있음 (덮어쓰지 않음): $log" >&2; exit 1; }
  mkdir -p "$(dirname "$dir")"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 >"$log" &
  local pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  local rc=0
  run "$@" || rc=$?
  kill "$pid" 2>/dev/null || true
  trap - EXIT
  return $rc
}

stage_train() {
  run uv run python experiments/crag_system/run.py "${COMMON[@]}" \
    --exp-id "PRE-$TAG-TRAIN" --run-id train --set dataset.split.part=train
  run uv run python experiments/crag_system/fit_cost_model.py "$TRAIN_DIR" \
    --service-key wall --out "$COST_MODEL"
}

stage_seq() {
  with_gpu_log "$SEQ_DIR" uv run python experiments/crag_system/run.py "${COMMON[@]}" \
    --exp-id "PRE-$TAG-A" --run-id eval_seq --set dataset.split.part=eval \
    --set crag.web_search.provider=tavily --set crag.web_search.record=true
}

stage_calib() {
  [[ "$DRY_RUN" == 1 ]] || need "$SEQ_DIR/web_cache.jsonl" seq
  run uv run python experiments/crag_system/calibrate_k.py "${COMMON[@]}" \
    --set dataset.split.part=eval "${REPLAY[@]}" --set crag.web_search.latency_mode=zero \
    --ks 1,2,4,6,8 --n-requests 40
  [[ "$DRY_RUN" == 1 ]] && return
  local cal
  cal=$(ls -td "$PRE"/PRE-K/*_calibration | head -1)
  uv run python - "$cal/calibration.json" "$STATE" <<'EOF'
import json, sys
rec = json.load(open(sys.argv[1], encoding="utf-8"))["recommendation"]
if rec.get("K_star") is None:
    sys.exit(f"K* 없음: {rec}")
with open(sys.argv[2], "x", encoding="utf-8") as f:  # 덮어쓰기 금지
    f.write(f"K_STAR={rec['K_star']}\nMU_MAX={rec['mu_max_qps']}\nCALIB={sys.argv[1]}\n")
print(json.dumps(rec, indent=2, ensure_ascii=False))
EOF
  echo "saved -> $STATE (결과 표를 보고 K 를 바꾸려면 K_MAIN=, MU= 로 지정)"
}

wl_path() { echo "results/crag_system/workloads/${TAG}_rho$1_s$2.json"; }

stage_workloads() {
  load_state
  for rho in $RHOS; do
    for s in $SEEDS; do
      run uv run python experiments/crag_system/make_workload.py "${COMMON[@]}" \
        --set dataset.split.part=eval --name "${TAG}_rho${rho}_s${s}" \
        --rho "$rho" --capacity-rps "$MU" --seed "$s"
    done
  done
}

# $1 rho  $2 seed  $3 policy  $4 K  $5 order_index  $6 session id
live_one() {
  local rho=$1 s=$2 pol=$3 k=$4 order=$5 session=$6
  local wl run_id
  wl=$(wl_path "$rho" "$s")
  [[ "$DRY_RUN" == 1 ]] || need "$wl" workloads
  run_id="rho${rho}_s${s}_${pol}_K${k}"
  local extra=()
  [[ "$pol" != none ]] && extra+=(--set "scheduler.cost_model={path: $COST_MODEL}")
  [[ "$RESET_PREFIX_CACHE" == 1 ]] && extra+=(--set run.reset_prefix_cache=true)
  with_gpu_log "$LIVE_ROOT/$run_id" uv run python experiments/crag_system/run.py "${COMMON[@]}" \
    --exp-id "PRE-$TAG-LIVE" --run-id "$run_id" --workload "$wl" \
    --set run.mode=live --set dataset.split.part=eval "${REPLAY[@]}" \
    --set crag.web_search.latency_mode=recorded \
    --set "scheduler.policy=$pol" --set "scheduler.slots=$k" \
    --set run.sample_engine_metrics_sec=0.5 \
    --set "run.session={id: $session, order_index: $order}" "${extra[@]}"
}

# 정책 순서를 seed 마다 회전 (실행 순서에 따른 drift / prefix cache 영향을 조건 간에 분산)
rotated() {
  local shift=$1; shift
  local arr=("$@") n=$# i
  for ((i = 0; i < n; i++)); do echo "${arr[$(((i + shift) % n))]}"; done
}

stage_live() {
  load_state
  [[ "$DRY_RUN" == 1 ]] || need "$COST_MODEL" train
  local rho s pol order si=0
  for s in $SEEDS; do
    for rho in $RHOS; do
      order=0
      # shellcheck disable=SC2086
      for pol in $(rotated "$si" $POLICIES); do
        live_one "$rho" "$s" "$pol" "$K_MAIN" "$order" "$TAG-live-rho$rho-s$s"
        order=$((order + 1))
      done
    done
    si=$((si + 1))
  done
}

stage_ksweep() {
  load_state
  local k pol order=0
  for k in $KSWEEP_KS; do
    [[ "$k" == "$K_MAIN" ]] && continue  # live 단계에서 이미 실행
    for pol in $KSWEEP_POLICIES; do
      live_one "$KSWEEP_RHO" 0 "$pol" "$k" "$order" "$TAG-ksweep"
      order=$((order + 1))
    done
  done
}

stage_analyze() {
  load_state
  local A=experiments/crag_system/analyze_sched_validation.py
  run uv run python $A --run "seq=$SEQ_DIR" --paths-only --name "$TAG-A"
  local rho s pol args gpu
  for rho in $RHOS; do
    for s in $SEEDS; do
      args=()
      for pol in $POLICIES; do
        local d="$LIVE_ROOT/rho${rho}_s${s}_${pol}_K${K_MAIN}"
        [[ -d "$d" || "$DRY_RUN" == 1 ]] || { echo "건너뜀 (없음): $d" >&2; continue; }
        args+=(--run "$pol=$d")
        gpu="$d.gpu.csv"
        [[ -f "$gpu" ]] && args+=(--gpu-log "$pol=$gpu")
      done
      run uv run python $A "${args[@]}" --ref fcfs --name "$TAG-BC-rho${rho}-s${s}"
      # gate 대기열 / 재정렬 / 추정 정확도 진단 (gate 를 쓰는 정책만)
      local dd=()
      for pol in $POLICIES; do
        [[ "$pol" == none ]] && continue
        local d="$LIVE_ROOT/rho${rho}_s${s}_${pol}_K${K_MAIN}"
        [[ -d "$d" || "$DRY_RUN" == 1 ]] && dd+=("$d")
      done
      run uv run python experiments/crag_system/diag_sched.py "${dd[@]}" \
        --name "$TAG-diag-rho${rho}-s${s}" --quiet
    done
  done
  # K sweep (있으면). K_MAIN run 도 포함 (live 단계에서 seed 0 으로 실행됨)
  args=()
  local ks
  ks=$(printf '%s\n' $KSWEEP_KS "$K_MAIN" | sort -n -u)
  for k in $ks; do
    for pol in $KSWEEP_POLICIES; do
      local d="$LIVE_ROOT/rho${KSWEEP_RHO}_s0_${pol}_K${k}"
      [[ -d "$d" ]] && args+=(--run "${pol}_K${k}=$d")
    done
  done
  if ((${#args[@]} > 2)); then
    run uv run python $A "${args[@]}" --name "$TAG-K-rho${KSWEEP_RHO}"
  fi
}

[[ $# -ge 1 ]] || { sed -n '2,32p' "$0"; exit 1; }
[[ "$DRY_RUN" == 1 || -n "${VLLM_API_KEY:-}" ]] || { echo "VLLM_API_KEY 가 없음" >&2; exit 1; }
for st in "$@"; do
  case "$st" in
    train | seq | calib | workloads | live | ksweep | analyze) "stage_$st" ;;
    *) echo "알 수 없는 stage: $st" >&2; exit 1 ;;
  esac
done
