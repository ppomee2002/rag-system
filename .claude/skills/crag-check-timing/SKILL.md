---
name: crag-check-timing
description: C-RAG latency 측정 규칙 위반을 점검한다 — StageTimer/stopwatch 사용, stage 이름과 rag.pipelines.crag.STAGES 일치, stopwatch 범위 밖 작업, 공정 비교 조건(공유 인스턴스, temperature 0, warmup, 순서 랜덤화). pipeline·run.py·retriever·LLM 코드를 수정한 뒤, 실험 실행 전, 또는 "측정 규칙 확인", "timing 체크", "latency 측정 맞는지 봐줘" 요청 시 사용.
---

# Latency 측정 규칙 점검

측정 코드가 하나만 어긋나도 실험 수치 전체를 믿을 수 없게 된다. 변경된 코드를 기준으로 아래 항목을 점검하고 위반만 보고한다.
**수정은 사용자가 요청할 때만** 한다. `experiments/chunking/` 은 점검·수정 대상이 아니다.

## 대상 범위

기본은 `git diff` + untracked 파일 중 `src/rag/`, `experiments/crag_latency/` 에 해당하는 것.
사용자가 "전체" 라고 하면 위 두 경로 전체를 본다.

## 점검 항목

### A. 측정 도구
1. latency 측정에 `time.time()`, `time.monotonic()`, `datetime.now()` 직접 사용 금지 → `rag.utils.timing.StageTimer` / `stopwatch` 만 사용.
   `grep -rnE "time\.(time|monotonic|perf_counter)\(|datetime\.now\(" src/rag experiments/crag_latency`
   (`timing.py` 내부의 perf_counter, run.py 의 run 폴더 timestamp 용 `datetime.now()` 는 정상)
2. `timer.stage("<name>")` 의 이름이 모두 `rag.pipelines.crag.STAGES` 에 있는지 — 오타가 있으면 `stage_latencies` 에서 해당 시간이 0 으로 빠지고 `(untracked)` 로 넘어간다.
   `grep -rnE "\.stage\(" src/rag experiments/crag_latency` 결과를 STAGES 와 대조.
3. 새 stage 를 추가했다면 `STAGES` 에도 추가됐는지, `analyze.py` 의 `SERIES` 팔레트 길이(5)를 넘지 않는지.

### B. 측정 범위
4. 각 pipeline 의 `run()` 에서 측정 대상 작업(retrieve, LLM 호출, web search)이 모두 `with stopwatch()` **안**에 있는지.
5. 측정과 무관한 작업(채점 `score_all`, 결과 직렬화, 파일 쓰기, 로깅)이 stopwatch **밖**에 있는지.
6. LLM/검색 호출이 어떤 `timer.stage` 에도 속하지 않으면 `(untracked)` 로 잡힌다 — 의도한 것인지 확인.
7. Vanilla 와 C-RAG 의 공통 단계(retrieval, generation)가 같은 방식으로 감싸져 있는지 (한쪽만 전처리 포함 등).

### C. 공정 비교 (run.py / config)
8. `build_pipelines` 에서 vanilla 와 crag 가 **같은** retriever·llm 인스턴스를 받는지 (pipeline 별로 새로 생성하지 않는지).
9. 최종 config 의 `llm.temperature == 0` (grader 를 따로 둘 경우 grader 도).
10. warmup query 가 결과 기록 전에 실행되고 기록에서 제외되는지.
11. query 마다 pipeline 실행 순서를 랜덤화하는지.
12. `stage_latencies` 를 STAGES 전체 키로 채우는지 (빠진 키는 analyze 에서 0 처리되지만 crag 쪽은 명시적으로 채우는 것이 규칙).

### D. 테스트
13. `uv run pytest -q tests/test_timing.py tests/test_crag.py` 통과 여부.

## 보고 형식

위반이 있으면 항목마다 한 줄씩: `파일:라인 — 항목 번호 — 무엇이 문제이고 수치에 어떤 영향을 주는지`.
위반이 없으면 "측정 규칙 위반 없음" 과 함께 확인한 파일 목록과 테스트 결과를 적는다.
`src/rag/` 인터페이스나 `configs/base.yaml` 변경이 필요한 수정이면 chunking 실험에도 영향이 있다는 점을 함께 알린다.
