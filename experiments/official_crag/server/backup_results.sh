#!/usr/bin/env bash
# 재현에 필요한 최소 결과만 /workspace/rag01 에 백업 (서버에서 사용자가 실행).
# 복사: config.yaml, run_meta.json, summary.json, queries.jsonl(.gz), 분석 csv/md, 고정 입력 manifest,
#       provenance (공식 commit·가중치 hash), smoke evaluator 보고서. calls.jsonl·web_cache 는 제외 (--with-calls).
# 원본은 지우거나 바꾸지 않는다. 대상 폴더가 이미 있으면 실패.
#
#   bash experiments/official_crag/server/backup_results.sh [--with-calls] <run 또는 analysis 폴더>...
set -euo pipefail
DEST_ROOT=${DEST_ROOT:-/workspace/rag01/official_crag}
WITH_CALLS=0
if [[ "${1:-}" == --with-calls ]]; then WITH_CALLS=1; shift; fi
[[ $# -ge 1 ]] || { echo "사용: $0 [--with-calls] <폴더>..."; exit 2; }
DEST="$DEST_ROOT/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST_ROOT"
mkdir "$DEST"
for src in "$@"; do
  [[ -d "$src" ]] || { echo "없음: $src"; exit 1; }
  rel=${src#results/}
  out="$DEST/$rel"
  mkdir -p "$out"
  for f in config.yaml run_meta.json summary.json run_end.json report.md conditions.csv paired.csv \
           agreement.csv per_question.csv scores.jsonl preds.txt; do
    if [[ -f "$src/$f" ]]; then cp -p "$src/$f" "$out/"; fi
  done
  for f in queries.jsonl preds.jsonl per_question.jsonl refinement.jsonl; do
    if [[ -f "$src/$f" ]]; then gzip -c "$src/$f" > "$out/$f.gz"; fi
  done
  if [[ $WITH_CALLS == 1 && -f "$src/calls.jsonl" ]]; then gzip -c "$src/calls.jsonl" > "$out/calls.jsonl.gz"; fi
done
# 공통 재현 자료
if [[ -d results/official_crag/provenance ]]; then cp -rp results/official_crag/provenance "$DEST/"; fi
if [[ -d results/official_crag/smoke/evaluator ]]; then cp -rp results/official_crag/smoke/evaluator "$DEST/smoke_evaluator"; fi
mkdir -p "$DEST/inputs"
for m in results/official_crag/inputs/*.manifest.json; do if [[ -f "$m" ]]; then cp -p "$m" "$DEST/inputs/"; fi; done
for m in results/official_crag/inputs/*.jsonl; do if [[ -f "$m" ]]; then gzip -c "$m" > "$DEST/inputs/$(basename "$m").gz"; fi; done
git rev-parse HEAD > "$DEST/git_commit.txt"
du -sh "$DEST"
echo "backup -> $DEST"
