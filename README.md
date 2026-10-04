# rag-system

RAG 실험 레포. 공통 코어(`src/rag`) 위에서 실험별 폴더로 나누어 진행한다.

| 실험 | 폴더 | 내용 |
|---|---|---|
| C-RAG latency | `experiments/crag_latency/` | Vanilla RAG vs C-RAG 품질/latency 비교, 단계별 latency 분석 |
| Chunking | `experiments/chunking/` | 청킹 전략별 retrieval/답변 품질 비교 |

## 구조

```
configs/base.yaml        # 공통 설정 (LLM, retriever, dataset) — 실험 config 가 덮어씀
src/rag/
  types.py               # Document, QAExample, PipelineResult
  config.py              # base + 실험 config deep-merge
  data/                  # 데이터 로더
  chunking/              # 청킹 전략 (Chunker 인터페이스)
  retrieval/             # Retriever, WebSearch 인터페이스 / 구현
  llm/                   # LLM 인터페이스 / 구현
  pipelines/             # vanilla.py, crag.py, prompts.py
  eval/metrics.py        # EM / F1 / Accuracy
  utils/timing.py        # StageTimer (단계별 latency)
experiments/<name>/      # 실험별 config, 실행/분석 스크립트
data/                    # 데이터 (커밋 X, data/README.md 참고)
results/                 # 실험 결과 (커밋 X)
tests/
```

## 세팅

```bash
# uv 설치: https://docs.astral.sh/uv/
uv sync
cp .env.example .env     # API 키 입력
uv run pytest
```

## 협업 규칙

- 자기 실험은 `experiments/<name>/` 안에서. 공통 코어(`src/rag/`) 변경은 PR 로 공유.
- `configs/base.yaml` 의 LLM / retriever 를 바꾸면 양쪽 실험에 모두 영향 → 팀원과 합의 후 변경.
- 브랜치: `exp/crag-latency`, `exp/chunking` 처럼 실험별로.
