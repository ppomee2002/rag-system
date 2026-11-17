#!/usr/bin/env bash
# Evidence preservation 정확도-latency-GPU 비용 검증 (evidence_cost.py / run_evidence_live.py).
# 사용자가 GPU 서버 (~/RAG) 에서 실행한다. Claude 는 실행하지 않는다.
# 기존 run (SV1 seq / live, evidence filter ablation) 은 읽기만 하고, 결과는 새 폴더에만 쓴다.
#
#   EVF=results/crag_system/pre/PRE-EVF-SV1-EVAL/<ts> bash experiments/crag_system/run_evidence_cost.sh <stage>
#
# stage
#   check        측정 단위 구성·재구성 검증만 (LLM 호출 없음)
#   estimate     [추정] 보고서 (LLM 호출 없음, 측정 전 바로 가능)
#   measure      (GPU) 조건별 generation sequential latency 측정 -> PRE-$TAG-SEQ
#   report       [측정] 반영 보고서 (LLM 호출 없음)
#   live         (GPU, 이번 단계에서는 실행하지 않음) LIVE_POLICIES 를 SV1 workload 로 실행
#   live-report  live 결과: 정확도 + 지연 + 처리량 + GPU (evidence_cost live-report, analyze_sched_validation)
#   diag         live 정합성·지연 분해·queueing·scheduling 진단 (diag_evc_live.py, LLM 호출 없음)
#
# 환경 변수 (기본값)
#   EVF               필수. evidence_filter_ablation 생성 run 폴더
#   VLLM_API_KEY      measure / live 에 필요
#   TAG=EVC-SV1       exp_id 접두사 (PRE-$TAG-*). 다시 측정할 때는 TAG 를 바꾼다 (덮어쓰기 금지)
#   SV_TAG=SV1        재사용할 SV workload / cost model / seq run 의 TAG
#   PREFIX_CACHE=reset  measure: 매 호출 전 prefix cache 초기화 (서버 --dev-mode 필요). none 이면 초기화 없음
#   REPEATS=1         measure 반복 (2 이상이면 repeat 마다 순서를 다시 섞음)
#   CONDITIONS=       measure 조건 (비우면 evf 의 조건 전부)
#   PROFILE=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
#   LIVE_POLICIES="crag_rule keep_all"   report 의 Live 후보를 보고 정한다
#   RHOS="0.8 0.95"   SEEDS="0 1"   K_MAIN=4   SCHED=fcfs_query
#   RESET_PREFIX_CACHE=0  live: 1 이면 run 시작 전 prefix cache 초기화 (--dev-mode 서버만)
#   GPU_LOG=1   DRY_RUN=0
set -euo pipefail

cd "$(dirname "$0")/../.."
: "${EVF:?EVF=<evidence_filter_ablation run 폴더> 를 지정}"
TAG=${TAG:-EVC-SV1}
SV_TAG=${SV_TAG:-SV1}
PREFIX_CACHE=${PREFIX_CACHE:-reset}
REPEATS=${REPEATS:-1}
CONDITIONS=${CONDITIONS:-}
PROFILE=${PROFILE:-experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml}
LIVE_POLICIES=${LIVE_POLICIES:-"crag_rule keep_all"}
RHOS=${RHOS:-"0.8 0.95"}
SEEDS=${SEEDS:-"0 1"}
K_MAIN=${K_MAIN:-4}
SCHED=${SCHED:-fcfs_query}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-0}
GPU_LOG=${GPU_LOG:-1}
DRY_RUN=${DRY_RUN:-0}

PRE=results/crag_system/pre
SEQ_DIR=$PRE/PRE-$SV_TAG-A/eval_seq
MEAS_ROOT=$PRE/PRE-$TAG-SEQ
LIVE_ROOT=$PRE/PRE-$TAG-LIVE
COST_MODEL=results/crag_system/cost_models/${SV_TAG}_train_wall.json
PY=experiments/crag_system/evidence_cost.py

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

need() { [[ -e "$1" || "$DRY_RUN" == 1 ]] || { echo "없음: $1 ($2)" >&2; exit 1; }; }

with_gpu_log() {  # $1 = 로그 경로 접두사
  local log="$1.gpu.csv"; shift
  if [[ "$DRY_RUN" == 1 || "$GPU_LOG" != 1 ]] || ! command -v nvidia-smi >/dev/null; then
    run "$@"; return
  fi
  [[ -e "$log" ]] && { echo "이미 있음 (덮어쓰지 않음): $log" >&2; exit 1; }
  mkdir -p "$(dirname "$log")"
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

latest_meas() { ls -td "$MEAS_ROOT"/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

stage_check() {
  need "$EVF/generations.jsonl" "EVF"
  # shellcheck disable=SC2086
  run uv run python $PY measure --evf "$EVF" --check-only ${CONDITIONS:+--conditions $CONDITIONS}
}

stage_estimate() {
  need "$SEQ_DIR/calls.jsonl" "SV1 seq"
  run uv run python $PY report --evf "$EVF" --seq "$SEQ_DIR" --name "$TAG-estimate"
}

stage_measure() {
  [[ "$DRY_RUN" == 1 || -n "${VLLM_API_KEY:-}" ]] || { echo "VLLM_API_KEY 가 없음" >&2; exit 1; }
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  # shellcheck disable=SC2086
  with_gpu_log "$MEAS_ROOT/gpu_$ts" uv run python $PY measure --evf "$EVF" \
    --exp-id "PRE-$TAG-SEQ" --prefix-cache "$PREFIX_CACHE" --repeats "$REPEATS" \
    ${CONDITIONS:+--conditions $CONDITIONS}
}

stage_report() {
  local m=${MEAS:-$(latest_meas)}
  [[ -n "$m" ]] || { echo "measure 결과 없음 (MEAS= 로 지정 가능)" >&2; exit 1; }
  run uv run python $PY report --evf "$EVF" --seq "$SEQ_DIR" --measured "$m" --name "$TAG"
}

rotated() {
  local shift=$1; shift
  local arr=("$@") n=$# i
  for ((i = 0; i < n; i++)); do echo "${arr[$(((i + shift) % n))]}"; done
}

stage_live() {
  [[ "$DRY_RUN" == 1 || -n "${VLLM_API_KEY:-}" ]] || { echo "VLLM_API_KEY 가 없음" >&2; exit 1; }
  need "$COST_MODEL" "SV train"
  local s rho pol order si=0
  for s in $SEEDS; do
    for rho in $RHOS; do
      local wl="results/crag_system/workloads/${SV_TAG}_rho${rho}_s${s}.json"
      need "$wl" "SV workloads (같은 arrival trace 재사용)"
      order=0
      # shellcheck disable=SC2086
      for pol in $(rotated "$si" $LIVE_POLICIES); do
        local run_id="rho${rho}_s${s}_${pol}_K${K_MAIN}" dir
        dir="$LIVE_ROOT/$run_id"
        if [[ -f "$dir/summary.json" ]]; then echo "완료됨, 건너뜀: $dir"; order=$((order + 1)); continue; fi
        if [[ -e "$dir" ]]; then
          echo "중단된 run 폴더가 있음 (덮어쓰지 않음): $dir -> 이름을 바꾸고 다시 실행" >&2; exit 1
        fi
        local extra=()
        [[ "$RESET_PREFIX_CACHE" == 1 ]] && extra+=(--set run.reset_prefix_cache=true)
        with_gpu_log "$dir" uv run python experiments/crag_system/run_evidence_live.py \
          --config experiments/crag_system/config.yaml --overlay "$PROFILE" \
          --set run.purpose=pre --set dataset.limit=503 \
          --exp-id "PRE-$TAG-LIVE" --run-id "$run_id" --workload "$wl" \
          --set "evidence_policy=$pol" \
          --set run.mode=live --set dataset.split.part=eval \
          --set crag.web_search.provider=replay \
          --set "crag.web_search.replay_from_runs=['$SEQ_DIR']" \
          --set crag.web_search.replay_key=query_id \
          --set crag.web_search.replay_on_miss=empty \
          --set crag.web_search.latency_mode=recorded \
          --set "scheduler.policy=$SCHED" --set "scheduler.slots=$K_MAIN" \
          --set "scheduler.cost_model={path: $COST_MODEL}" \
          --set run.sample_engine_metrics_sec=0.5 \
          --set "run.session={id: $TAG-live-rho$rho-s$s, order_index: $order}" "${extra[@]}"
        order=$((order + 1))
      done
    done
    si=$((si + 1))
  done
}

stage_live_report() {
  run uv run python $PY live-report --live-root "$LIVE_ROOT" --seq "$SEQ_DIR" --name "$TAG-live"
  local rho s pol args
  for rho in $RHOS; do
    for s in $SEEDS; do
      args=()
      for pol in $LIVE_POLICIES; do
        local d="$LIVE_ROOT/rho${rho}_s${s}_${pol}_K${K_MAIN}"
        [[ -d "$d" || "$DRY_RUN" == 1 ]] || continue
        args+=(--run "$pol=$d")
        [[ -f "$d.gpu.csv" ]] && args+=(--gpu-log "$pol=$d.gpu.csv")
      done
      ((${#args[@]} >= 4)) || continue
      run uv run python experiments/crag_system/analyze_sched_validation.py "${args[@]}" \
        --ref crag_rule --name "$TAG-live-rho${rho}-s${s}" --quiet
    done
  done
}

stage_diag() {
  local m=${MEAS:-$(latest_meas)} cal
  local args=(--live-root "$LIVE_ROOT" --seq "$SEQ_DIR" --evf "$EVF" --k "$K_MAIN" --name "$TAG-live-diag")
  [[ -n "$m" ]] && args+=(--evc-seq "$m")
  [[ -d "$PRE/PRE-$SV_TAG-LIVE" ]] && args+=(--sv-live-root "$PRE/PRE-$SV_TAG-LIVE")
  cal=$(ls -td "$PRE"/PRE-K/*_calibration 2>/dev/null | head -1 || true)
  [[ -n "$cal" && -f "$cal/calibration.json" ]] && args+=(--calib "$cal/calibration.json")
  run uv run python experiments/crag_system/diag_evc_live.py "${args[@]}" --quiet
}

[[ $# -ge 1 ]] || { sed -n '2,29p' "$0"; exit 1; }
for st in "$@"; do
  case "$st" in
    check | estimate | measure | report | live) "stage_$st" ;;
    live-report) stage_live_report ;;
    diag) stage_diag ;;
    *) echo "알 수 없는 stage: $st" >&2; exit 1 ;;
  esac
done
