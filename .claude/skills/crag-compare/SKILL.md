---
name: crag-compare
description: C-RAG latency 실험의 두 run 폴더(results/crag_latency/<run_id>)를 비교한다 — config 차이, pipeline 별 E2E·stage latency·품질 변화, 같은 query 끼리 짝지은 paired 차이, corrective action 분포 변화. threshold·모델·top_k·grader 등 설정 변경 효과를 볼 때, "두 run 비교", "이전 결과랑 비교", "설정 바꾼 효과" 요청 시 사용.
---

# 두 run 비교

`analyze.py` 는 run 하나를 분석한다. 이 스킬은 **설정을 바꾼 두 run 의 차이**를 본다.

## 1. 대상 run 정하기

- 사용자가 지정하지 않으면 `ls -t results/crag_latency/` 에서 최신 두 개를 쓰고, 어떤 두 개를 골랐는지 먼저 말한다.
- 순서: A = 기준(이전), B = 비교 대상(새 설정).

## 2. 실행

```bash
uv run python .claude/skills/crag-compare/compare_runs.py results/crag_latency/<A> results/crag_latency/<B>
```

출력(마크다운)은 B 폴더에 `compare_vs_<A>.md` 로도 저장된다. 포함 내용:

1. **config 차이**: 두 run 의 `config.yaml` 에서 값이 다른 키 (dotted path)
2. **pipeline 요약**: n, E2E mean/p50/p90, acc/em/f1 — A, B, Δ(B−A)
3. **stage 별 평균 latency** (C-RAG): A, B, Δ, Δ%
4. **paired 비교**: 두 run 모두에 있는 (query_id, pipeline) 만 짝지어 E2E 차이의 mean / median 과 B 가 더 빠른 query 비율.
   API latency 편차가 크므로 평균 차이보다 이 값을 더 신뢰한다.
5. **corrective action 분포** (C-RAG): 비율과 action 별 평균 E2E
6. **action 전이표**: 같은 query 의 action 이 A→B 에서 어떻게 바뀌었는지 (threshold·grader 변경 효과)

## 3. 해석 시 주의

- 공통 query 수가 적으면(예: 한쪽이 `--limit` smoke run) 경고가 출력된다 — 이때는 paired 결과만 의미 있다.
- config 차이에 `dataset.limit`, `run.repeats` 외 항목이 여러 개면 효과를 하나의 원인으로 단정하지 않는다.
- 네트워크 지연이 섞인 E2E 의 작은 차이(±5% 이내, paired median 기준)는 "차이 없음" 으로 보고한다.

## 4. 보고

config 차이 한 줄 + 핵심 변화 3~4줄(가장 크게 변한 stage, paired E2E 변화, 품질 변화, action 분포 변화)로 요약하고 저장된 파일 경로를 적는다.
그래프가 필요하면 `dataviz` 스킬 규칙을 따르고, 색은 `analyze.py` 의 `SERIES` 팔레트를 재사용한다.
