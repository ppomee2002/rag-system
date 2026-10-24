# Correction-Aware Scheduling 실험

C-RAG 는 query 마다 correction 여부에 따라 작업량이 달라진다. 이 동적 작업량을 고려해 로컬 LLM 서버(단일 GPU)
위에서 여러 query 를 어떻게 스케줄링해야 지연이 줄어드는지 본다.

| # | 실험 | 방법 |
|---|---|---|
| 1 | Stage-Level Profiling | 부하 없이 query 를 하나씩 실행해 stage / LLM 호출 단위 trace (지연, 토큰, 서버 처리 시간) 수집 |
| 2 | Workload Analysis | correction 으로 늘어나는 작업량, retrieval 직후 feature 로 correction / 작업량을 예측할 수 있는지 (AUC) |
| 3 | Concurrent Query | open-loop Poisson 부하 (rho 0.3~0.95) 에서 E2E 평균 / p95 / 처리량. Ollama 병렬 slot vs client gate, 순차 vs 병렬 grading |
| 4 | Scheduling Baseline | FIFO (call / query) vs SJF-predicted vs SRPT-est (correction-aware) vs SRPT-oracle |

실험 3·4 는 실험 1 trace 를 재생하는 discrete-event simulation (`scheduling.py`) 으로 수행하고,
`run_live.py` 로 일부 조건을 실제 동시 부하로 돌려 simulation 을 검증한다.

## 실행

```bash
# Ollama 를 headless 로 (GUI 앱 대신) 아래 환경변수로 띄운 상태여야 한다
#   OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
uv run python experiments/correction_scheduling/run_profile.py            # 실험 1 (~1시간)
uv run python experiments/correction_scheduling/simulate.py <run_dir>     # 실험 3·4
uv run python experiments/correction_scheduling/run_live.py <run_dir>     # simulation 검증 (~50분)
uv run python experiments/correction_scheduling/analyze.py <run_dir>      # 실험 2 + report.md
```

`<run_dir>` = `results/correction_scheduling/<timestamp>`.

## 설계 결정

- **로컬 서빙 (Ollama llama3.1:8b)**: API 는 병목이 원격 서버 / rate limit 이라 스케줄링이 제어할 자원이 없다.
- **Ollama host 는 `127.0.0.1`**: Windows 에서 `localhost` 는 IPv6 시도 때문에 호출마다 ~2초가 붙는다.
- **web search 는 mock**: 실제 Tavily 지연 (crag_latency 실험 측정값) 을 query 별 고정 추출해 재생하고, 문서는
  rewrite 된 query 로 로컬 FAISS 검색. 지연 분포는 실제와 같지만 답변 품질 수치는 실제 web search 와 다르다.
- **pipeline 별 pass 로 profiling**: 같은 query 를 pipeline 끼리 연달아 돌리면 동일 prompt 가 Ollama prompt
  (KV) cache 에 걸려 두 번째 실행이 수 배 빨라진다.
- **GPU 모델**: 동시 호출이 처리량을 늘리지 않으므로 (probe: 4개 동시 8.7s vs 순차 ~7.6s) processor sharing
  으로 모델링. 정책 비교는 client gate 가 한 번에 1개씩 보내고 대기 호출 순서를 정하는 방식.
- 순서만 바꾸는 정책이라 답변 품질은 변하지 않는다 (조기 종료류는 품질 측정이 따로 필요).
