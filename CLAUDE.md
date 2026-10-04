# CLAUDE.md

RAG 실험 레포. 두 실험이 공통 코어를 공유한다.

- **C-RAG latency** (`experiments/crag_latency/`): Vanilla RAG vs C-RAG 의 품질/E2E latency 비교, C-RAG 단계별(retrieval / grading / rewrite / web_search / generation) latency 분석, 난이도(corrective action)별 비교.
- **Chunking** (`experiments/chunking/`): 청킹 전략 비교. 다른 팀원 담당.

## 명령어

- 의존성: `uv sync`
- 테스트: `uv run pytest`
- 린트: `uv run ruff check . && uv run ruff format .`
- 실험 실행: `uv run python experiments/<name>/run.py --config experiments/<name>/config.yaml`

## 구조 / 규칙

- 공통 코드는 `src/rag/`, 실험 전용 코드는 `experiments/<name>/`. 실험 전용 로직을 코어에 넣지 않는다.
- 설정은 `configs/base.yaml` + 실험 config deep-merge (`rag.config.load_config`). 하드코딩 대신 config 사용.
- 컴포넌트는 `Protocol` 인터페이스(`Retriever`, `WebSearch`, `LLM`, `Chunker`)로 주입. 구현체 추가 시 인터페이스를 따른다.
- Latency 측정은 반드시 `rag.utils.timing.StageTimer` / `stopwatch` 사용 (`time.perf_counter` 기반). stage 이름은 `rag.pipelines.crag.STAGES` 와 일치시킨다.
- 공정 비교: Vanilla / C-RAG 는 같은 retriever·LLM 인스턴스 공유, temperature 0.
- 결과는 `results/<experiment>/<timestamp>/` 에 query 단위 jsonl + 사용한 config 저장. `data/`, `results/`, `.env` 는 커밋하지 않는다.

## 주의

- 다른 실험 폴더(`experiments/chunking/` 등)는 요청 없이 수정하지 않는다.
- `src/rag/` 의 인터페이스나 `configs/base.yaml` 변경은 양쪽 실험에 영향 → 변경 시 사용자에게 명시적으로 알린다.
- 개인 메모/설정은 `CLAUDE.local.md`, `.claude/settings.local.json` (gitignore 됨).
