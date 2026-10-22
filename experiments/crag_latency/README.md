# C-RAG Latency 실험

Vanilla RAG 와 C-RAG 의 답변 품질 / latency 를 비교하고, C-RAG 의 추가 단계가 전체 latency 에 미치는 영향을 분석한다.

## 실행

```bash
uv sync --extra crag
# .env: ANTHROPIC_API_KEY, TAVILY_API_KEY, WIKIMEDIA_CONTACT

uv run python experiments/crag_latency/prepare_data.py   # PopQA long-tail + Wikipedia 문서 수집
uv run python experiments/crag_latency/build_index.py    # 청킹 -> bge 임베딩 -> FAISS
uv run python experiments/crag_latency/run.py --config experiments/crag_latency/config.yaml [--limit 20]
uv run python experiments/crag_latency/analyze.py results/crag_latency/<run_id>
```

`analyze.py` 는 run 폴더에 `report.md` (결과 정리 표), `figures/` (component 구성, 난이도별 latency),
`results_flat.csv` 를 만든다.

## 측정 항목 (query 단위, `results.jsonl`)

| 필드 | 설명 |
|---|---|
| `answer`, `em` / `f1` / `acc` | 답변 및 품질 metric |
| `e2e_latency` | 전체 실행 시간 (초) |
| `stage_latencies` | `retrieval` / `grading` / `rewrite` / `web_search` / `generation` |
| `corrective_action` | `correct` / `ambiguous` / `incorrect` |
| `grader_scores`, `n_corrective_steps` | 난이도 구분 기준 |
| `extra.retrieval_scores` | 최초 retrieval score (top-k) |

## 난이도 구분

`analyze.py` 의 `add_difficulty()` 에서 세 기준을 모두 계산해 각각 비교한다.

| 기준 | easy / hard |
|---|---|
| `diff_action` | corrective action 이 `correct` 이면 easy, 그 외 hard |
| `diff_grader` | 평균 grader score 3분위 (높을수록 easy) |
| `diff_retrieval` | 최초 top-1 retrieval score 3분위 (높을수록 easy) |

## Bad evidence 주입 실험 (`run_noise.py`)

gold 문서를 고정하고 bad 문서(distracting / random)의 종류와 개수만 바꿔 답변 품질을 비교한다.
latency 는 측정하지 않는다. 설정은 `config.yaml` 의 `noise:`.

```bash
uv run python experiments/crag_latency/run_noise.py --limit 203 --dry-run   # 문서 선택만 (LLM 호출 없음)
uv run python experiments/crag_latency/run_noise.py --limit 203 [--resume results/crag_noise/<run_id>]
```

결과는 `results/crag_noise/<run_id>/` 에 `results.jsonl` (query x condition), `selection.jsonl`
(query 별 gold / distracting / random 문서와 grader score), `report.md` 로 저장된다.

## 공정 비교 체크리스트

- [ ] 두 pipeline 이 같은 retriever / LLM 인스턴스 사용
- [ ] warmup query 제외 (`run.warmup_queries`)
- [ ] query 마다 pipeline 실행 순서 랜덤화 (run.py 에서 처리)
- [ ] API 기반 LLM 이면 네트워크 지연 편차 → `run.repeats` 로 반복 측정
- [ ] temperature 0
