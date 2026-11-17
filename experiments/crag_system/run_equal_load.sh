#!/usr/bin/env bash
# 동일 추정 유효 부하 (equal effective load) live 실험 (equal_load.py / run_evidence_live.py).
# 사용자가 GPU 서버 (~/RAG) 에서 실행한다. Claude 는 실행하지 않는다.
# 기존 workload / EVC live / SV1 결과는 읽기만 하고, 새 workload 와 새 폴더 (PRE-$TAG-LIVE) 에만 쓴다.
#
#   WORK_RATIO=1.513 bash experiments/crag_system/run_equal_load.sh make check
#   DRY_RUN=1 bash experiments/crag_system/run_equal_load.sh live      # 명령만 출력
#   bash experiments/crag_system/run_equal_load.sh live report
#
# stage
#   make     기존 ${SV_TAG}_rho${SRC_RHO}_s<seed>.json 의 도착 시각을 정책 작업량 비로 늘린 workload 생성
#            (WORK_RATIO 또는 DIAG_JSON 의 work_ratio_vs_crag_rule)
#   check    질문·순서 동일 / 도착 시각만 일정 비율 / 같은 seed 재생성과 일치 / run 폴더 충돌 없음
#   live     (GPU) POLICIES 를 새 workload 로 실행 (scheduler·K·web replay·cost model 은 EVC live 와 같음)
#   report   새 run vs 기존 EVC live (crag_rule / 같은 정책, 명목 rho=SRC_RHO): E2E·대기·처리량·drain·유효 부하
#   diag     새 run 폴더에 diag_evc_live (지연 분해·queueing)
#
# 환경 변수 (기본값)
#   WORK_RATIO=       정책의 요청당 무부하 작업량 / crag_rule (예: 1.513). 비우면 DIAG_JSON 사용
#   DIAG_JSON=        diag_evc_live analysis.json (work_ratio_vs_crag_rule 의 SRC_RHO run 평균)
#   POLICIES="keep_all"   TARGET_RHO=0.8   SRC_RHO=0.8   SEEDS="0 1"   K_MAIN=4   SCHED=fcfs_query
#   TAG=EQL1          workload 이름 / exp_id 접두사 (PRE-$TAG-LIVE). 다시 할 때는 TAG 를 바꾼다
#   SV_TAG=SV1        원본 workload / cost model / seq run 의 TAG
#   EVC_TAG=EVC-SV1   기준 live (PRE-$EVC_TAG-LIVE)
#   EVF=  MEAS=       (선택) evf run / evc-seq 측정 폴더 -> check 의 evf 제외 질문 수, report 의 no-load
#   PROFILE=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
#   RESET_PREFIX_CACHE=0   GPU_LOG=1   DRY_RUN=0
set -euo pipefail

cd "$(dirname "$0")/../.."
WORK_RATIO=${WORK_RATIO:-}
DIAG_JSON=${DIAG_JSON:-}
POLICIES=${POLICIES:-keep_all}
TARGET_RHO=${TARGET_RHO:-0.8}
SRC_RHO=${SRC_RHO:-0.8}
SEEDS=${SEEDS:-"0 1"}
K_MAIN=${K_MAIN:-4}
SCHED=${SCHED:-fcfs_query}
TAG=${TAG:-EQL1}
SV_TAG=${SV_TAG:-SV1}
EVC_TAG=${EVC_TAG:-EVC-SV1}
EVF=${EVF:-}
MEAS=${MEAS:-}
PROFILE=${PROFILE:-experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-0}
GPU_LOG=${GPU_LOG:-1}
DRY_RUN=${DRY_RUN:-0}

PRE=results/crag_system/pre
WLD=results/crag_system/workloads
SEQ_DIR=$PRE/PRE-$SV_TAG-A/eval_seq
LIVE_ROOT=$PRE/PRE-$TAG-LIVE
REF_ROOT=$PRE/PRE-$EVC_TAG-LIVE
COST_MODEL=results/crag_system/cost_models/${SV_TAG}_train_wall.json
PY=experiments/crag_system/equal_load.py

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

need() { [[ -e "$1" || "$DRY_RUN" == 1 ]] || { echo "없음: $1 ($2)" >&2; exit 1; }; }

with_gpu_log() {  # $1 = run 폴더
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

src_wl() { echo "$WLD/${SV_TAG}_rho${SRC_RHO}_s$1.json"; }
new_wl() { echo "$WLD/${TAG}_$1_rhoeff${TARGET_RHO}_s$2.json"; }

stage_make() {
  local pol s args
  for pol in $POLICIES; do
    args=(make --policy "$pol" --target-rho-eff "$TARGET_RHO" --tag "$TAG" --k "$K_MAIN")
    if [[ -n "$WORK_RATIO" ]]; then
      args+=(--work-ratio "$WORK_RATIO")
    elif [[ -n "$DIAG_JSON" ]]; then
      args+=(--from-diag "$DIAG_JSON" --diag-rho "$SRC_RHO")
    else
      echo "WORK_RATIO= 또는 DIAG_JSON= 이 필요" >&2; exit 1
    fi
    for s in $SEEDS; do
      need "$(src_wl "$s")" "원본 workload"
      args+=(--source "$(src_wl "$s")")
    done
    run uv run python $PY "${args[@]}"
  done
}

stage_check() {
  local pol s args=(check --live-root "$LIVE_ROOT" --k "$K_MAIN")
  [[ -d "$SEQ_DIR" ]] && args+=(--seq "$SEQ_DIR")
  [[ -n "$EVF" ]] && args+=(--evf "$EVF")
  for pol in $POLICIES; do
    for s in $SEEDS; do
      need "$(new_wl "$pol" "$s")" "make 단계"
      args+=(--workload "$(new_wl "$pol" "$s")")
    done
  done
  need "$COST_MODEL" "SV train"
  need "$SEQ_DIR/queries.jsonl" "SV seq (web replay)"
  run uv run python $PY "${args[@]}"
}

stage_live() {
  [[ "$DRY_RUN" == 1 || -n "${VLLM_API_KEY:-}" ]] || { echo "VLLM_API_KEY 가 없음" >&2; exit 1; }
  need "$COST_MODEL" "SV train"
  stage_check
  local pol s wl run_id dir order
  for s in $SEEDS; do
    order=0
    for pol in $POLICIES; do
      wl=$(new_wl "$pol" "$s")
      if [[ -f "$wl" ]]; then
        run_id=$(uv run python $PY run-id --workload "$wl" --k "$K_MAIN")
      else
        run_id="rho<nominal>_s${s}_${pol}_K${K_MAIN}"  # DRY_RUN 에서 make 전
      fi
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
        --set "run.session={id: $TAG-live-s$s, order_index: $order}" "${extra[@]}"
      order=$((order + 1))
    done
  done
}

latest_meas() { ls -td "$PRE/PRE-$EVC_TAG-SEQ"/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

stage_report() {
  local m=${MEAS:-$(latest_meas)}
  local args=(report --new-root "$LIVE_ROOT" --ref-root "$REF_ROOT" --ref-rho "$SRC_RHO"
              --seq "$SEQ_DIR" --k "$K_MAIN" --sched-policy "$SCHED" --name "$TAG")
  [[ -n "$m" ]] && args+=(--evc-seq "$m")
  run uv run python $PY "${args[@]}"
}

stage_diag() {
  local m=${MEAS:-$(latest_meas)}
  local args=(--live-root "$LIVE_ROOT" --seq "$SEQ_DIR" --k "$K_MAIN" --name "$TAG-live-diag")
  [[ -n "$EVF" ]] && args+=(--evf "$EVF")
  [[ -n "$m" ]] && args+=(--evc-seq "$m")
  run uv run python experiments/crag_system/diag_evc_live.py "${args[@]}" --quiet
}

[[ $# -ge 1 ]] || { sed -n '2,27p' "$0"; exit 1; }
for st in "$@"; do
  case "$st" in
    make | check | live | report | diag) "stage_$st" ;;
    *) echo "알 수 없는 stage: $st" >&2; exit 1 ;;
  esac
done
