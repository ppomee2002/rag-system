#!/usr/bin/env bash
# 상태 기반 SRPT-est (M2) GPU scheduling 검증 (state_sched.py / run_state_sched.py / state_sched_analysis.py).
# 사용자가 GPU 서버 (~/RAG) 에서 실행한다. Claude 는 실행하지 않는다.
# 기존 workload / cost model / EQL1 fcfs_query 결과는 읽기만 하고, 새 파일·새 폴더에만 쓴다.
#
#   bash experiments/crag_system/run_state_sched.sh fit expect check   # GPU 불필요
#   DRY_RUN=1 bash experiments/crag_system/run_state_sched.sh live      # 명령만 출력
#   bash experiments/crag_system/run_state_sched.sh live report
#
# stage
#   fit      train trace -> M2 표 (cost_models/${SV_TAG}_train_state_m2.json, 있으면 그대로 사용)
#   expect   (Part A) 기존 EQL1 fcfs_query live 로그로 결정 차이 + replay 시뮬레이션 + 검증 목표
#   check    기준 run (fcfs_query, keep_all, K, 같은 workload sha) / 누수 / 폴더 충돌 점검
#   live     (GPU) SCHEDS x SEEDS. seed 마다 정책 순서를 바꿈 (s0: srpt_est -> srpt_state, s1: 반대)
#   report   (Part D) fcfs_query (기준) vs srpt_est vs srpt_state
#
# 환경 변수 (기본값)
#   SCHEDS="srpt_est srpt_state"   (fcfs_query 를 넣으면 같은 runner 로 기준도 재측정 -> BASE_ROOT 자동 변경)
#   SEEDS="0 1"   K_MAIN=4   TAG=SST1   EQL_TAG=EQL1   TARGET_RHO=0.8   SV_TAG=SV1
#   EVC_TAG=EVC-SV1 (MEAS 기본값 탐색)   MEAS= (evc-seq 측정 폴더)   SPEEDUP= (예: 1:1,2:1.7,4:2.6)
#   EXPECT_JSON= (report 에 넣을 expect analysis.json, 비우면 최신 것)
#   PROFILE=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
#   STATE_SHADOW=1   RESET_PREFIX_CACHE=0   GPU_LOG=1   DRY_RUN=0
set -euo pipefail

cd "$(dirname "$0")/../.."
SCHEDS=${SCHEDS:-"srpt_est srpt_state"}
SEEDS=${SEEDS:-"0 1"}
K_MAIN=${K_MAIN:-4}
TAG=${TAG:-SST1}
EQL_TAG=${EQL_TAG:-EQL1}
TARGET_RHO=${TARGET_RHO:-0.8}
SV_TAG=${SV_TAG:-SV1}
EVC_TAG=${EVC_TAG:-EVC-SV1}
MEAS=${MEAS:-}
SPEEDUP=${SPEEDUP:-}
EXPECT_JSON=${EXPECT_JSON:-}
PROFILE=${PROFILE:-experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml}
STATE_SHADOW=${STATE_SHADOW:-1}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-0}
GPU_LOG=${GPU_LOG:-1}
DRY_RUN=${DRY_RUN:-0}

PRE=results/crag_system/pre
WLD=results/crag_system/workloads
ANA=results/crag_system/analysis/state_sched
TRAIN_DIR=$PRE/PRE-$SV_TAG-TRAIN/train
SEQ_DIR=$PRE/PRE-$SV_TAG-A/eval_seq
COST_MODEL=results/crag_system/cost_models/${SV_TAG}_train_wall.json
STATE_MODEL=results/crag_system/cost_models/${SV_TAG}_train_state_m2.json
BASE_ROOT=$PRE/PRE-$EQL_TAG-LIVE
[[ " $SCHEDS " == *" fcfs_query "* ]] && BASE_ROOT=$PRE/PRE-$TAG-LIVE-fcfs_query
PY=experiments/crag_system/state_sched.py
ANPY=experiments/crag_system/state_sched_analysis.py

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

need() { [[ -e "$1" || "$DRY_RUN" == 1 ]] || { echo "없음: $1 ($2)" >&2; exit 1; }; }

with_gpu_log() {  # $1 = run 폴더 (run_equal_load.sh 와 같은 형식: <run 폴더>.gpu.csv)
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

wl_of() { echo "$WLD/${EQL_TAG}_keep_all_rhoeff${TARGET_RHO}_s$1.json"; }
latest_meas() { ls -td "$PRE/PRE-$EVC_TAG-SEQ"/*/ 2>/dev/null | head -1 | sed 's:/$::'; }
latest_expect() { ls -td "$ANA/$TAG-expect"/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

stage_fit() {
  if [[ -f "$STATE_MODEL" ]]; then echo "있음, 그대로 사용: $STATE_MODEL"; return; fi
  need "$TRAIN_DIR/queries.jsonl" "SV train trace"
  need "$COST_MODEL" "SV cost model"
  run uv run python $PY fit --train "$TRAIN_DIR" --cost-model "$COST_MODEL" --out "$STATE_MODEL"
}

stage_expect() {
  need "$STATE_MODEL" "fit 단계"
  local m=${MEAS:-$(latest_meas)}
  local args=(expect --base-root "$PRE/PRE-$EQL_TAG-LIVE" --seq "$SEQ_DIR"
              --cost-model "$COST_MODEL" --state-model "$STATE_MODEL" --k "$K_MAIN"
              --name "$TAG-expect" --quiet)
  [[ -n "$m" ]] && args+=(--evc-seq "$m")
  [[ -n "$SPEEDUP" ]] && args+=(--speedup "$SPEEDUP")
  run uv run python $ANPY "${args[@]}"
}

stage_check() {
  need "$STATE_MODEL" "fit 단계"
  local s sch args=(check --state-model "$STATE_MODEL" --cost-model "$COST_MODEL"
                    --base-root "$PRE/PRE-$EQL_TAG-LIVE" --k "$K_MAIN")
  for s in $SEEDS; do need "$(wl_of "$s")" "EQL workload"; args+=(--workload "$(wl_of "$s")"); done
  for sch in $SCHEDS; do args+=(--new-root "$PRE/PRE-$TAG-LIVE-$sch"); done
  need "$SEQ_DIR/queries.jsonl" "SV seq (web replay)"
  run uv run python $PY "${args[@]}"
}

order_for() {  # seed 별 정책 순서 (실행 순서 효과 상쇄)
  local s=$1 list=($SCHEDS)
  if (( s % 2 == 1 )); then
    local i rev=()
    for (( i=${#list[@]}-1; i>=0; i-- )); do rev+=("${list[i]}"); done
    echo "${rev[@]}"
  else
    echo "${list[@]}"
  fi
}

stage_live() {
  [[ "$DRY_RUN" == 1 || -n "${VLLM_API_KEY:-}" ]] || { echo "VLLM_API_KEY 가 없음" >&2; exit 1; }
  stage_check
  local s sch wl run_id dir order
  for s in $SEEDS; do
    order=0
    for sch in $(order_for "$s"); do
      wl=$(wl_of "$s")
      if [[ -f "$wl" ]]; then
        run_id=$(uv run python experiments/crag_system/equal_load.py run-id --workload "$wl" --k "$K_MAIN")
      else
        run_id="rho<nominal>_s${s}_keep_all_K${K_MAIN}"  # DRY_RUN
      fi
      dir="$PRE/PRE-$TAG-LIVE-$sch/$run_id"
      if [[ -f "$dir/summary.json" ]]; then echo "완료됨, 건너뜀: $dir"; order=$((order + 1)); continue; fi
      if [[ -e "$dir" ]]; then
        echo "중단된 run 폴더가 있음 (덮어쓰지 않음): $dir -> 이름을 바꾸고 다시 실행" >&2; exit 1
      fi
      local extra=()
      [[ "$RESET_PREFIX_CACHE" == 1 ]] && extra+=(--set run.reset_prefix_cache=true)
      [[ "$STATE_SHADOW" == 1 ]] || extra+=(--set scheduler.state_shadow=false)
      # 아래 인자는 run_equal_load.sh live (EQL1 fcfs_query) 와 같고, runner / scheduler / state model 만 다르다
      with_gpu_log "$dir" uv run python experiments/crag_system/run_state_sched.py \
        --config experiments/crag_system/config.yaml --overlay "$PROFILE" \
        --set run.purpose=pre --set dataset.limit=503 \
        --exp-id "PRE-$TAG-LIVE-$sch" --run-id "$run_id" --workload "$wl" \
        --set evidence_policy=keep_all \
        --set run.mode=live --set dataset.split.part=eval \
        --set crag.web_search.provider=replay \
        --set "crag.web_search.replay_from_runs=['$SEQ_DIR']" \
        --set crag.web_search.replay_key=query_id \
        --set crag.web_search.replay_on_miss=empty \
        --set crag.web_search.latency_mode=recorded \
        --set "scheduler.policy=$sch" --set "scheduler.slots=$K_MAIN" \
        --set "scheduler.cost_model={path: $COST_MODEL}" \
        --set "scheduler.state_model={path: $STATE_MODEL}" \
        --set run.sample_engine_metrics_sec=0.5 \
        --set "run.session={id: $TAG-live-s$s, order_index: $order}" "${extra[@]}"
      order=$((order + 1))
    done
  done
}

stage_report() {
  local m=${MEAS:-$(latest_meas)} e=${EXPECT_JSON:-}
  [[ -z "$e" && -n "$(latest_expect)" ]] && e="$(latest_expect)/analysis.json"
  local args=(report --base-root "$BASE_ROOT" --m0-root "$PRE/PRE-$TAG-LIVE-srpt_est"
              --m2-root "$PRE/PRE-$TAG-LIVE-srpt_state" --seq "$SEQ_DIR" --k "$K_MAIN"
              --name "$TAG-report")
  [[ -n "$m" ]] && args+=(--evc-seq "$m")
  [[ -n "$e" ]] && args+=(--expect "$e")
  run uv run python $ANPY "${args[@]}"
}

[[ $# -ge 1 ]] || { sed -n '2,24p' "$0"; exit 1; }
for st in "$@"; do
  case "$st" in
    fit | expect | check | live | report) "stage_$st" ;;
    *) echo "알 수 없는 stage: $st" >&2; exit 1 ;;
  esac
done
