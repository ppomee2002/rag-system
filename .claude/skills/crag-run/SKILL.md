---
name: crag-run
description: C-RAG latency 실험을 처음부터 끝까지 실행한다 — 사전 점검(.env, 데이터, FAISS 인덱스), --limit smoke run, 본 실행, 오류 시 --resume, analyze.py 로 report.md 생성, 핵심 수치 요약. "실험 돌려줘", "crag 실행", "smoke run", "run 이어서", "결과 리포트 만들어줘" 같은 요청에 사용.
---

# C-RAG latency 실험 실행

`experiments/crag_latency/` 실험을 안전하게 끝까지 돌리고 결과를 요약한다.
API 비용(Anthropic, Tavily)이 드는 작업이므로 **본 실행 전에 반드시 smoke run 으로 확인**한다.

## 1. 사전 점검 (하나라도 실패하면 멈추고 사용자에게 알림)

- `.env` 존재 여부만 확인한다 (`test -f .env`). **내용은 읽지 않는다** (settings 에서 deny).
  필요한 키: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `WIKIMEDIA_CONTACT` — 없으면 `.env.example` 참고하라고 안내.
- config 확인: 기본 `experiments/crag_latency/config.yaml`. 사용자가 다른 config 를 지정하면 그것을 쓴다.
  `rag.config.load_config` 로 base 와 deep-merge 된 최종 값을 기준으로 아래 경로를 확인한다:
  - `dataset.path` 없음 → `uv run python experiments/crag_latency/prepare_data.py`
  - `retriever.index_path` / `retriever.docs_path` 없음 → `uv run python experiments/crag_latency/build_index.py`
  데이터/인덱스 생성은 오래 걸리므로 실행 전에 사용자에게 알린다.
- `uv run pytest -q` 통과 확인. 실패하면 실험을 돌리지 않는다.

## 2. Smoke run

```bash
uv run python experiments/crag_latency/run.py --config <config> --limit 10
```

- `--limit` 는 warmup(`run.warmup_queries`) 포함 개수다. limit ≤ warmup 이면 측정 행이 0 이 되므로 warmup + 5 이상으로 준다.
- 끝나면 출력된 run 폴더(`results/crag_latency/<timestamp>/`)에서 확인:
  - `results.jsonl` 행 수 = (limit − warmup) × pipeline 수 × repeats
  - `errors.jsonl` 이 비어 있는지
  - C-RAG 행의 `stage_latencies` 에 `retrieval/grading/rewrite/web_search/generation` 키가 모두 있는지
- 이상 없으면 본 실행 여부를 사용자에게 확인한다 (전체 1,399 문항 × 2 pipeline 은 시간·비용이 크다).

## 3. 본 실행

```bash
uv run python experiments/crag_latency/run.py --config <config> [--limit N]
```

- 오래 걸리므로 백그라운드로 실행하고 진행 상황(tqdm)을 지켜본다.
- 끝에 `N errors -> .../errors.jsonl` 이 출력되면 (rate limit 등) 같은 폴더로 재시도:
  ```bash
  uv run python experiments/crag_latency/run.py --config <config> --resume results/crag_latency/<run_id>
  ```
  `--resume` 은 완료된 (query_id, pipeline, repeat) 를 건너뛴다. 오류가 계속 반복되면 `errors.jsonl` 의 `error` 를 보고 원인을 보고한다.
  `errors.jsonl` 은 append 되므로 재시도 후에도 과거 오류가 남아 있다 — 완료 여부는 results.jsonl 행 수로 판단한다.

## 4. 분석

```bash
uv run python experiments/crag_latency/analyze.py results/crag_latency/<run_id>
```

`report.md`, `figures/components.png`, `figures/difficulty.png`, `results_flat.csv` 가 생성된다.

## 5. 요약 보고

report.md 를 읽고 다음만 짧게 정리한다 (전체 표를 그대로 붙이지 않는다):

1. Vanilla vs C-RAG: E2E mean / p50 / p90 과 `crag / vanilla` 배율, acc·f1 차이
2. C-RAG 에서 E2E 비중이 가장 큰 stage 와 `(untracked)` 비율 — untracked 가 10% 를 넘으면 측정 누락 의심으로 표시
3. corrective action 분포 (correct / ambiguous / incorrect 비율)와 action 별 E2E
4. 난이도(easy/hard)별 C-RAG − Vanilla latency 차이가 가장 큰 기준

마지막에 run 폴더 경로와 그림 경로를 적는다. 그림을 보여줄 때는 Read 로 png 를 열어 확인한다.

## 주의

- `results/`, `data/` 는 커밋 대상이 아니다.
- config 를 바꿔 여러 run 을 비교하려면 `crag-compare` 스킬을 사용한다.
- 공정 비교 조건(같은 retriever·LLM 인스턴스, temperature 0, warmup 제외, 순서 랜덤화)은 run.py 가 처리한다. run.py 를 수정했다면 실행 전에 `crag-check-timing` 으로 점검한다.
