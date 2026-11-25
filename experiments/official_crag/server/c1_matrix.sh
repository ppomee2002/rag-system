#!/usr/bin/env bash
# C1 조건 행렬 (MODE B) 을 순차 실행한다. 서버에서 사용자가 실행 (Claude 는 실행하지 않음).
#
# 필요한 환경변수
#   FROZEN               고정 입력 jsonl (freeze_inputs.py 결과, .manifest.json 동반)
#   WEB_CACHE            공유 correction 검색 캐시 (prefetch_web.py 결과 web_cache.jsonl, v2) — 권장
#     또는 REPLAY_RUN    기존 crag_system run 폴더 (web_cache.jsonl v1). 검색어 문자열이 같을 때만 재생
#                        (query_strict + accept_legacy_records). 검색어가 다르면 miss -> 오류 행
#   CRAG_EVALUATOR_PATH  공식 T5 가중치 폴더 (finetuned_t5_evaluator)
# 선택
#   EVAL_DEVICE=cpu      evaluator 위치 (cpu | cuda:0). vLLM 과 같은 GPU 면 README "GPU 메모리" 참고
#   EVAL_BACKEND=t5      t5 (in-process) | cache (precompute_scores.py 결과, EVAL_CACHE 필요)
#   PURPOSE=pre PREFIX=PRE-C1   (본실험: PURPOSE=paper PREFIX=C1 + PAPER=--paper-approved)
#   LIMIT=               smoke 면 숫자 (예: 10)
#   ONLY="T5-OFF LL-RULE"   일부 조건만
#
# 예)  FROZEN=results/official_crag/inputs/popqa_c1_common.jsonl \
#      WEB_CACHE=results/official_crag/web_cache/popqa_c1_common_legacy_rewrite/web_cache.jsonl \
#      CRAG_EVALUATOR_PATH=$HOME/models/crag_evaluator/finetuned_t5_evaluator \
#      LIMIT=10 PURPOSE=smoke PREFIX=SMOKE-C1 bash experiments/official_crag/server/c1_matrix.sh
set -euo pipefail
: "${FROZEN:?FROZEN 필요}"
if [[ -z "${WEB_CACHE:-}" && -z "${REPLAY_RUN:-}" ]]; then
  echo "WEB_CACHE 또는 REPLAY_RUN 필요" >&2
  exit 1
fi
EVAL_DEVICE=${EVAL_DEVICE:-cpu}
EVAL_BACKEND=${EVAL_BACKEND:-t5}
PURPOSE=${PURPOSE:-pre}
PREFIX=${PREFIX:-PRE-C1}
PAPER=${PAPER:-}
CFG=experiments/official_crag/config.yaml
ENGINE=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml

common=(--config "$CFG" --overlay "$ENGINE"
  --set experiment=official_crag
  --set "inputs.frozen_path=$FROZEN"
  --set web_search.replay_key=query_strict
  --set "run.purpose=$PURPOSE"
  --set "evaluator.device=$EVAL_DEVICE"
  --set "evaluator.backend=$EVAL_BACKEND")
if [[ "$EVAL_BACKEND" == cache ]]; then
  : "${EVAL_CACHE:?EVAL_BACKEND=cache 에는 EVAL_CACHE 필요}"
  common+=(--set "evaluator.cache_paths=[$EVAL_CACHE]")
fi
if [[ -n "${WEB_CACHE:-}" ]]; then
  common+=(--set "web_search.replay_paths=[$WEB_CACHE]")
else
  common+=(--set "web_search.replay_paths=[$REPLAY_RUN/web_cache.jsonl]"
    --set web_search.accept_legacy_records=true)
fi
[[ -n "${LIMIT:-}" ]] && common+=(--limit "$LIMIT")

declare -A COND=(
  [VAN]="--set evidence_policy=vanilla"
  [T5-OFF]="--set grader.type=official_t5 --set evidence_policy=official"
  [T5-OFF-OP]="--set grader.type=official_t5 --set evidence_policy=official --set generation.prompt=official"
  [T5-RULE]="--set grader.type=official_t5 --set evidence_policy=crag_rule"
  [T5-NCF]="--set grader.type=official_t5 --set evidence_policy=no_correct_filter"
  [T5-NID]="--set grader.type=official_t5 --set evidence_policy=no_incorrect_discard"
  [T5-KEEP]="--set grader.type=official_t5 --set evidence_policy=keep_all"
  [LL-RULE]="--set grader.type=llama_prompt --set evidence_policy=crag_rule"
  [LL-NCF]="--set grader.type=llama_prompt --set evidence_policy=no_correct_filter"
  [LL-NID]="--set grader.type=llama_prompt --set evidence_policy=no_incorrect_discard"
  [LL-KEEP]="--set grader.type=llama_prompt --set evidence_policy=keep_all"
  [LL-OFF]="--set grader.type=llama_prompt --set evidence_policy=official"
)
ORDER=(VAN T5-OFF T5-OFF-OP T5-RULE T5-NCF T5-NID T5-KEEP LL-RULE LL-NCF LL-NID LL-KEEP LL-OFF)
[[ -n "${ONLY:-}" ]] && read -r -a ORDER <<< "$ONLY"

mkdir -p results/official_crag/logs
LOG=results/official_crag/logs/c1_matrix_$(date +%Y%m%d_%H%M%S).txt
for id in "${ORDER[@]}"; do
  echo "=== $PREFIX-$id ($(date -Is))" | tee -a "$LOG"
  # shellcheck disable=SC2086
  uv run --extra crag --extra official python experiments/official_crag/run.py "${common[@]}" \
    --exp-id "$PREFIX-$id" ${COND[$id]} $PAPER 2>&1 | tee -a "$LOG" | tail -n 3
done
echo "done. 분석: uv run python experiments/official_crag/analyze.py results/official_crag/$PURPOSE/$PREFIX-*/* --name <이름>" | tee -a "$LOG"
