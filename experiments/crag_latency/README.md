# C-RAG Latency 실험

Vanilla RAG 와 C-RAG 의 답변 품질 / latency 를 비교하고, C-RAG 의 추가 단계가 전체 latency 에 미치는 영향을 분석한다.

## 실행

```bash
uv run python experiments/crag_latency/run.py --config experiments/crag_latency/config.yaml
uv run python experiments/crag_latency/analyze.py results/crag_latency/<run_id>
```

## 측정 항목 (query 단위, `results.jsonl`)

| 필드 | 설명 |
|---|---|
| `answer`, `em` / `f1` / `acc` | 답변 및 품질 metric |
| `e2e_latency` | 전체 실행 시간 (초) |
| `stage_latencies` | `retrieval` / `grading` / `rewrite` / `web_search` / `generation` |
| `corrective_action` | `correct` / `ambiguous` / `incorrect` |
| `grader_scores`, `n_corrective_steps` | 난이도 구분 기준 |

## 난이도 구분

기본값: C-RAG `corrective_action` 이 `correct` 이면 easy, 그 외 hard.
`analyze.py` 의 `add_difficulty()` 에서 기준 변경 가능 (최대 grader score, 최초 retrieval score 등).

## 공정 비교 체크리스트

- [ ] 두 pipeline 이 같은 retriever / LLM 인스턴스 사용
- [ ] warmup query 제외 (`run.warmup_queries`)
- [ ] query 마다 pipeline 실행 순서 랜덤화 (run.py 에서 처리)
- [ ] API 기반 LLM 이면 네트워크 지연 편차 → `run.repeats` 로 반복 측정
- [ ] temperature 0
