#!/usr/bin/env bash
# C2 gating — Controlled Replay smoke (서버, 사용자 실행. Claude 는 서버에 접속하지 않음)
#
# 기준 MODE A (정확도 상승 확인): LLaMA2-7B-hf fp16 TP=2, prompt v3_2024_10, 공식 T5
#   config  experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml
#   scores  results/official_crag/mode_a/score/20261126_072612
#   rag     results/official_crag/mode_a/generate_rag/20261126_093945   (705/1399)
#   crag    results/official_crag/mode_a/generate_crag/20261127_023204  (769/1399)
#
# 사용: bash experiments/c2_gating/server/smoke.sh <step>
#   prep      의존성 + MODE A 기준 확인 (정답 파일 자동 탐색, 705/769 재계산) + Correct 비율 집합   [GPU 불필요]
#   t5mem     T5 메모리 측정 -> vLLM gpu_memory_utilization 계산, refinement 중복 채점·memo 검사    [vLLM 중지 상태]
#   vllm      vLLM 실행 명령 (vLLM venv 창에서): 필요하면 계산된 gpu_memory_utilization 로        [별도 창]
#   probe     T5 ∥ vLLM 실제 중첩·동시 요청 점검                                                  [vLLM 실행 중]
#   web       smoke 질문 (판정별 4) + warmup 후보의 web 기록 (Tavily)                              [vLLM 실행 중]
#   smoke     B/C/Cbn x 동시 1/4, memo off + memo_verify 두 run, smoke_check                     [vLLM 실행 중]
#   reverify  (t5mem 에서 설정 변경이 필요했을 때만) MODE A rag/crag 1,399 재생성·비교               [vLLM 실행 중]
#   collect   공유할 작은 결과 파일 묶기
# TAG 를 바꾸면 결과 폴더 이름이 바뀐다 (기본 s1. 같은 TAG 로 다시 실행하면 덮어쓰지 않고 실패).
set -euo pipefail
cd "$(dirname "$0")/../../.."

TAG=${TAG:-s1}
MODE_A_CONFIG=experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml
SCORES=results/official_crag/mode_a/score/20261126_072612
RAG_PREDS=results/official_crag/mode_a/generate_rag/20261126_093945
CRAG_PREDS=results/official_crag/mode_a/generate_crag/20261127_023204
PROMPT_VERSION=v3_2024_10
T5_GPU=1                          # T5 를 올릴 물리 GPU (vLLM TP=2 는 0,1 모두 사용)
PER_ACTION=4
D=experiments/c2_gating
X=(uv run --extra crag --extra official python)
REF=results/c2_gating/mode_a_ref/$TAG
PLAN=results/c2_gating/probe/${TAG}_t5mem/gpu_plan.json
WEB=results/c2_gating/web_cache/${TAG}_smoke/web_cache.jsonl
SETS=results/c2_gating/workload_sets/${TAG}_ratio200

need_env() {
  local miss=0
  for v in "$@"; do [[ -n "${!v:-}" ]] || { echo "환경변수 $v 필요"; miss=1; }; done
  return $miss
}
answers() { source "$REF/ref.env"; echo "$ANSWERS"; }
plan_get() { python3 -c "import json,sys;print(json.load(open('$PLAN'))[sys.argv[1]])" "$1"; }
latest_launch() {
  ls -t ~/crag_logs/vllm_llama2_7b_fp16_tp2_fcfs_*_launch.json 2>/dev/null | head -1
}
vllm_up() {
  curl -sf -H "Authorization: Bearer ${VLLM_API_KEY:-}" http://127.0.0.1:8000/v1/models >/dev/null \
    || { echo "vLLM (127.0.0.1:8000) 응답 없음 -> 'smoke.sh vllm' 의 명령으로 먼저 실행"; exit 1; }
}

case "${1:-}" in
prep)
  uv sync --extra crag --extra official
  [[ -d third_party/CRAG ]] || { echo "third_party/CRAG 없음 (fetch_official.py repo)"; exit 1; }
  uv run python $D/resolve_mode_a.py resolve --run-id "$TAG" \
    --mode-a-config $MODE_A_CONFIG --scores $SCORES --rag-preds $RAG_PREDS \
    --crag-preds $CRAG_PREDS --prompt-version $PROMPT_VERSION
  echo "정답 파일: $(answers)"
  uv run python $D/make_workload_sets.py --scores $SCORES --name "${TAG}_ratio200" --n 200
  ;;
t5mem)
  need_env CRAG_EVALUATOR_PATH
  if curl -s -o /dev/null http://127.0.0.1:8000/v1/models; then
    echo "vLLM 이 실행 중. T5 단독 메모리 측정을 위해 vLLM 을 먼저 종료"; exit 1
  fi
  CUDA_VISIBLE_DEVICES=$T5_GPU "${X[@]}" $D/probe_serving.py t5-memory --run-id "${TAG}_t5mem" \
    --mode-a-config $MODE_A_CONFIG --crag-preds $CRAG_PREDS
  CUDA_VISIBLE_DEVICES=$T5_GPU "${X[@]}" $D/check_memo.py --mode-a-config $MODE_A_CONFIG \
    --forward --n 100
  echo "change_required=$(plan_get change_required) recommended=$(plan_get recommended_gpu_memory_utilization)"
  ;;
vllm)
  util=$(plan_get recommended_gpu_memory_utilization)
  extra=""
  [[ "$(plan_get change_required)" == "True" ]] && extra="--gpu-memory-utilization $util"
  cat <<EOF
# vLLM 전용 venv 창 (tmux) 에서:
cd $(pwd)
source ~/venvs/vllm/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 VLLM_API_KEY=\$(cat ~/.vllm_key)
python scripts/server/launch_vllm.py --profile $MODE_A_CONFIG $extra --dry-run
python scripts/server/launch_vllm.py --profile $MODE_A_CONFIG $extra
# 로그에서 KV cache 확인: grep -i "KV cache" \$(ls -t ~/crag_logs/vllm_llama2_7b_fp16_tp2_fcfs_*.log | head -1)
EOF
  ;;
probe)
  need_env CRAG_EVALUATOR_PATH VLLM_API_KEY
  vllm_up
  CUDA_VISIBLE_DEVICES=$T5_GPU "${X[@]}" $D/probe_serving.py concurrent --run-id "${TAG}_concurrent" \
    --mode-a-config $MODE_A_CONFIG
  ;;
web)
  need_env VLLM_API_KEY
  [[ -n "${TAVILY_API_KEY:-}" ]] || grep -q '^TAVILY_API_KEY=' .env 2>/dev/null \
    || { echo "TAVILY_API_KEY (.env 또는 환경변수) 필요"; exit 1; }
  vllm_up
  "${X[@]}" $D/prefetch_web.py --mode-a-config $MODE_A_CONFIG --reference-scores $SCORES \
    --per-action $PER_ACTION --name "${TAG}_smoke"
  ;;
smoke)
  need_env CRAG_EVALUATOR_PATH VLLM_API_KEY
  vllm_up
  L=$(latest_launch)
  echo "vLLM launch 기록: $L"
  common=(--mode-a-config $MODE_A_CONFIG --reference-scores $SCORES --reference-preds $CRAG_PREDS
    --answers "$(answers)" --web-cache $WEB --per-action $PER_ACTION --purpose smoke
    --modes B,C,Cbn --concurrency 1,4 --repeats 1 --vllm-launch-meta "$L")
  CUDA_VISIBLE_DEVICES=$T5_GPU "${X[@]}" $D/run.py "${common[@]}" --run-id "${TAG}_base"
  uv run python $D/smoke_check.py "results/c2_gating/controlled_replay/smoke/${TAG}_base" || true
  CUDA_VISIBLE_DEVICES=$T5_GPU "${X[@]}" $D/run.py "${common[@]}" --run-id "${TAG}_memo_verify" \
    --set t5_service.memo=true --set t5_service.memo_verify=true
  uv run python $D/smoke_check.py "results/c2_gating/controlled_replay/smoke/${TAG}_memo_verify" || true
  ;;
reverify)
  vllm_up
  [[ "$(plan_get change_required)" == "True" ]] || { echo "설정 변경 없음 -> 재검증 불필요"; exit 0; }
  M=experiments/official_crag/mode_a.py
  uv run python $M --run-id "c2_reverify_${TAG}" generate --method crag --scores $SCORES \
    --config $MODE_A_CONFIG --prompt-version $PROMPT_VERSION
  uv run python $D/resolve_mode_a.py compare-preds --run-id "${TAG}_crag" --answers "$(answers)" \
    --a $CRAG_PREDS --b "results/official_crag/mode_a/generate_crag/c2_reverify_${TAG}" || true
  uv run python $M --run-id "c2_reverify_${TAG}" generate --method rag \
    --config $MODE_A_CONFIG --prompt-version $PROMPT_VERSION
  uv run python $D/resolve_mode_a.py compare-preds --run-id "${TAG}_rag" --answers "$(answers)" \
    --a $RAG_PREDS --b "results/official_crag/mode_a/generate_rag/c2_reverify_${TAG}" || true
  ;;
collect)
  out=results/c2_gating/share_${TAG}_$(date +%Y%m%d_%H%M%S).tar.gz
  files=$(ls -d $REF/result.json $SETS/manifest.json results/c2_gating/probe/${TAG}_*/*.json \
    results/c2_gating/check_memo/*/summary.json \
    results/c2_gating/web_cache/${TAG}_smoke/manifest.json \
    results/c2_gating/controlled_replay/smoke/${TAG}_*/{summary.json,smoke_report.md,run_meta.json} \
    results/c2_gating/reverify/${TAG}_*/result.json 2>/dev/null || true)
  tar czf "$out" $files
  echo "$out"; tar tzf "$out"
  ;;
*)
  sed -n 2,21p "$0"; exit 1 ;;
esac
