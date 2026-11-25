#!/usr/bin/env bash
# MODE A: 같은 generator 로 공식 RAG vs CRAG 를 PopQA 1,399문항 전체에서 비교 (공식 match 지표).
# 서버에서 사용자가 실행 (Claude 는 실행하지 않음). vLLM 은 미리 띄워 둔다.
#
# 순서
#   0) (vLLM 을 띄우기 전, 또는 vLLM 과 다른 GPU) 공식 T5 점수:
#        CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
#          python experiments/official_crag/mode_a.py --run-id t5_full score --device cuda:0
#      -> SCORES=results/official_crag/mode_a/score/t5_full
#      LLaMA2 TP=2 처럼 GPU 2장을 vLLM 이 모두 쓰면 반드시 vLLM 실행 전에 끝낸다.
#   1) vLLM 실행 (CONFIG 의 모델). 예: Llama 3.1 AWQ 1장
#        CUDA_VISIBLE_DEVICES=0 python scripts/server/launch_vllm.py \
#          --profile experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
#      LLaMA2 / SelfRAG fp16 TP=2:
#        CUDA_VISIBLE_DEVICES=0,1 python scripts/server/launch_vllm.py \
#          --profile experiments/official_crag/configs/mode_a_selfrag_7b_tp2.yaml
#   2) 이 스크립트
#
# 필요한 환경변수
#   SCORES   0) 의 score run 폴더
#   ANSWERS  Self-RAG eval_data/popqa_longtail_w_gs.jsonl (공식 eval.py 입력과 같은 정답 파일)
# 선택
#   CONFIG=experiments/official_crag/configs/mode_a_popqa.yaml   (generator 설정)
#   PROMPT_VERSION=v3_2024_10   (v1v2_selfrag_format = 논문 v1/v2 시기 prompt)
#   TAG=<이름>   run 폴더 이름 (기본: 모델·prompt·시각). 이미 있으면 실패 (덮어쓰지 않음)
#   LIMIT=       smoke 면 숫자 (예: 20). 본 비교는 비워 둔다 (1,399 전체)
#   SUBSET=      C1 공통 200문항 id 파일 (evaluate 의 subset 값도 함께 기록)
set -euo pipefail
: "${SCORES:?SCORES 필요 (mode_a.py score 결과 폴더)}" "${ANSWERS:?ANSWERS 필요}"
CONFIG=${CONFIG:-experiments/official_crag/configs/mode_a_popqa.yaml}
PROMPT_VERSION=${PROMPT_VERSION:-v3_2024_10}
TAG=${TAG:-$(basename "$CONFIG" .yaml)_${PROMPT_VERSION}_$(date +%Y%m%d_%H%M%S)}
lim=()
[[ -n "${LIMIT:-}" ]] && lim=(--limit "$LIMIT")
sub=()
[[ -n "${SUBSET:-}" ]] && sub=(--subset-ids "$SUBSET")
M=experiments/official_crag/mode_a.py
R=results/official_crag/mode_a

for method in rag crag; do
  extra=()
  [[ $method == crag ]] && extra=(--scores "$SCORES")
  echo "== generate $method ($TAG)"
  uv run python "$M" --run-id "$TAG" generate --method "$method" --config "$CONFIG" \
    --prompt-version "$PROMPT_VERSION" "${extra[@]}" "${lim[@]}"
  echo "== evaluate $method"
  uv run python "$M" --run-id "${TAG}_$method" evaluate \
    --preds "$R/generate_$method/$TAG" --answers "$ANSWERS" "${sub[@]}"
done
echo "== compare rag (A) vs crag (B)"
uv run python "$M" --run-id "$TAG" compare --a "$R/evaluate/${TAG}_rag" --b "$R/evaluate/${TAG}_crag"
echo "결과: $R/compare/$TAG/summary.json"
