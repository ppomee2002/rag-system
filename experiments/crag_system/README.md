# CRAG System (통합 실험)

LangGraph + vLLM + Custom Scheduler 기반 Corrective RAG 통합 시스템. Grading 최적화와 GPU 요청 스케줄링을
하나의 실행 환경에서 비교한다. 작업 브랜치: `exp/crag-system` (`main` 에서 분기, 팀원 브랜치는 수정하지 않음).

## 실험 명명 체계

기존 실험과 번호가 겹치지 않도록 prefix 로 구분한다. 결과 폴더는 `results/crag_system/<ID>/<timestamp>/`.

| ID | 의미 | 출처 |
|---|---|---|
| `L-CL` | Vanilla vs C-RAG latency (crag_latency) | `main` `experiments/crag_latency/` |
| `L-NZ` | bad evidence 주입 (crag_noise) | `main` `run_noise.py` |
| `L-CS1`~`L-CS4` | 기존 correction_scheduling 실험 1~4 (profiling / workload / 동시 부하 / 스케줄링 baseline) | `exp/correction-scheduling` |
| `L-GO` | 기존 Full / Lazy-hit / Lazy-rest grading 측정 | `exp/grading-optimization` |
| `L-SO` | 기존 SRPT-est / Aging / 대기 임계값 trace simulation | `exp/scheduling-optimization` |
| **`SYS-1`** | Baseline Integration (Vanilla, Full CRAG, LangGraph, vLLM, 기존 정확도 재현) | 신규 |
| **`SYS-2`** | Grading Optimization (Full / Lazy-hit / Batch / Adaptive) | 신규 |
| **`SYS-3`** | Scheduling Optimization (FCFS / SRPT-est / Aging / 대기 제한 / Correction-aware) | 신규 |
| **`SYS-4`** | Integrated Evaluation (2×2 + 후보) | 신규 |
| `SYS-P` | 엔진 / 서버 점검 (probe) | 신규, `scripts/server/probe_engine.py` |

세부 조건은 소문자 접미사: 예) `SYS-1a` Ollama 재현(203), `SYS-1b` vLLM Baseline(203), `SYS-1c` 엔진 bridge,
`SYS-2x` Continuous Batching × Batch Grading 분리 실험. 문항 수는 `n203` / `n1399` 태그로 기록한다.
mode 는 `sequential` / `live` / `trace_sim` 으로 결과에 반드시 표시하고 서로 섞지 않는다.

## 확정된 방향 (2026-11-09 사용자 결정)

1. 통합 브랜치 `exp/crag-system`. 기존 브랜치·결과 보존, 필요한 구현만 선별 이식 (출처 주석 기록).
2. 기존 `LLM.generate()` 유지 + `InferenceEngine` (Ollama / vLLM) 추상화 추가.
3. vLLM 이 최종 주 엔진. 서버 환경 미확정 → 로컬에서 검증 스크립트 준비, 실행은 사용자.
4. LangGraph 는 기존 구현 공통화, Grading 전략 교체 가능 구조.
5. Batch Grading 우선, Adaptive 는 이후.
6. Scheduler 는 `LLMGate` / SRPT-est / `CostModel` 재사용, Aging·대기 제한을 live 환경으로 확장.
7. Correction-aware 는 구현 전 가설부터 제안 (아래).
8. vLLM Continuous Batching 효과와 Batch Grading 효과를 분리하는 실험 설계 (아래).
9. 203문항 초기 재현 → 1,399문항 정확도 검증. 비열등성 ε 후보 3%p / 5%p (미확정).
10. 정확도는 실제 Tavily correction 기준. 성능 실험도 실제 correction 우선, 외부 지연은 분리 분석.
11. 주요 성능 비교는 동일 서버·엔진·GPU 할당·모델 조건.
12. 명명 체계 분리 (위 표).

## 기존 브랜치 코드 재사용 계획

이식 시 원본은 수정하지 않고, 이식한 파일 상단에 `Source: <branch>:<path>` 를 남긴다.

| 기존 코드 | 처리 | 이식 위치 / 단계 |
|---|---|---|
| `main` `src/rag/*` (CRAG, Vanilla, prompts, FAISS, Tavily, metrics, StageTimer, OllamaLLM) | 그대로 사용 | — |
| `main` `OllamaLLM` | `OllamaEngine` 이 내부에서 그대로 호출 (요청 payload 가 기존 실험과 동일) | P1 ✅ |
| `correction_scheduling/common.py` `QueryLLM` 의 호출 trace 필드 (`kind/idx/t_start/server_total/...`) | 필드명 유지한 `CallRecord` 로 일반화 → 기존 `job_from_row` 가 새 trace 를 읽을 수 있음 | P1 ✅ `rag/telemetry/calls.py` |
| `correction_scheduling/common.py` `call_kind` | `infer_call_kind` 로 이식 | P1 ✅ |
| `correction_scheduling/common.py` `LLMGate` | `EngineLLM(gate=...)` 이 같은 `acquire(priority)/release()` 인터페이스로 받음. 코드 이식 + K slots / 동적 우선순위 확장은 P4 | P4 |
| `correction_scheduling/scheduling.py` `features`, `CostModel`, `fit_cost_model`, OOF | 그대로 이식, 토큰 기반 추정 추가 | P4 |
| `scheduling_optimization/simulator.py` (Aging / 임계값) | 수정 없이 사용, 새 trace adapter 만 추가. 정책 로직은 live gate 정책으로 옮김 | P4 |
| `correction_scheduling/run_live.py` (open-loop Poisson 부하, prompt cache flush) | 구조 재사용 | P4 |
| `grading_optimization/run_live.py` LangGraph 그래프 | `src/rag/graph/` 로 공통화, grading 노드는 전략 객체 호출 | P2 |
| `grading_optimization/verify_local.py` | pytest 동등성 테스트로 이식 | P2 |
| `grading_optimization/gpu_monitor.py` | live 실행 중 GPU 샘플링용으로 이식 | P2~P4 |
| `grading_optimization/run.py` `Backend` (OpenAI 호환, `priority: 0` 하드코딩) | 대체됨 → `VLLMEngine` | P1 ✅ |
| `grading_optimization/start_local.py`, `setup_api.ps1`, `_deps/` | 개인 PC 경로 / vendored 의존성 → 이식하지 않음 | — |

## P0 / P1 변경 범위

공통 코어(`src/rag/`) 변경은 **추가만** 한다. 기존 파일·시그니처·`configs/base.yaml` 은 변경하지 않는다.

| 구분 | 파일 | 내용 |
|---|---|---|
| 신규 (core) | `src/rag/llm/engine.py` | `LLMRequest` / `LLMResponse` / `InferenceEngine` Protocol / `EngineError` / HTTP transport |
| 신규 (core) | `src/rag/llm/ollama_engine.py` | `OllamaEngine` (기존 `OllamaLLM` 재사용, 서버 측 시간·토큰·모델 digest 기록) |
| 신규 (core) | `src/rag/llm/vllm_engine.py` | `VLLMEngine` (OpenAI 호환 chat API, streaming TTFT, priority 전달 옵션, `/metrics`) |
| 신규 (core) | `src/rag/llm/engine_llm.py` | `EngineLLM`: 엔진을 기존 `LLM.generate()` 로 감싸는 adapter (+ 호출 기록, gate, priority hook) |
| 신규 (core) | `src/rag/llm/factory.py` | config → 엔진 생성 |
| 신규 (core) | `src/rag/telemetry/calls.py` | `CallRecord` / `CallRecorder` / `infer_call_kind` |
| 신규 (core) | `src/rag/telemetry/run_meta.py` | 실행 메타데이터 (git, 패키지, GPU, 엔진, seed, config hash), 덮어쓰기 금지 run 폴더, JSONL writer |
| 신규 | `experiments/crag_system/` | 이 README, 엔진 config |
| 신규 | `scripts/server/` | `check_env.sh`, `check_vllm_install.py`, `launch_vllm.sh`, `probe_engine.py` |
| 신규 | `tests/conftest.py`, `tests/test_engines.py`, `tests/test_run_meta.py`, `tests/test_probe.py` | 로컬 mock HTTP 서버 기반 테스트 |
| 수정 | `.env.example` | `VLLM_API_KEY` 항목 추가 |
| 수정 | `CLAUDE.md` | 실험 목록에 `crag_system` 한 줄 추가 |

의존성 추가 없음 (HTTP 는 표준 라이브러리 `urllib`). vLLM 은 서버에서 **별도 venv** 에 설치한다
(클라이언트 레포 환경과 CUDA/torch 의존성을 분리하고 `uv.lock` 에 Linux 전용 패키지를 넣지 않기 위해).

## 호출 단위 측정 구조

- 각 query 는 자기 `CallRecorder` 를 갖고, 모든 LLM 호출은 `EngineLLM` → `InferenceEngine.complete()` 를 지난다.
- `CallRecord` 필드는 기존 correction_scheduling trace 와 같은 이름을 쓴다
  (`kind, idx, priority, t_enqueue, t_start, t_end, gate_wait, wall, prompt_chars, prompt_tokens, out_tokens,
  server_total, server_load, server_prompt_eval, server_eval`) + 신규 (`t_first_token, ttft, engine_priority,
  finish_reason, engine, model, request_id, error`).
- 시각은 실험 `t0` 기준 초 (`time.perf_counter`; `stopwatch` 와 같은 시계). 동시 실행 시 모든 query 가 같은 `t0` 공유.
- `server_*` 는 Ollama 만 응답에 포함한다. vLLM 은 요청 단위 서버 시간을 응답에 주지 않으므로 `None` 이며,
  TTFT (streaming) 와 `/metrics` 집계 (queue / prefill / decode / preemption) 로 대신한다.
  → 기존 simulator 의 `server_total` 기반 서비스 시간은 vLLM trace 에 그대로 쓸 수 없다 (P4 adapter 에서 처리).
- query 행에 `calls` 를 포함해 저장한다 (기존 `profile.jsonl` 형식과 호환).
- run 폴더: `config.yaml`, `run_meta.json`, `queries.jsonl`. 폴더가 이미 있으면 실패 (덮어쓰기 없음).

## SYS-2x: Continuous Batching vs Batch Grading 분리 설계 (P3 에서 실행)

같은 vLLM 서버 설정·GPU 할당·모델에서 두 요인을 교차한다.

| 요인 | 수준 |
|---|---|
| Grading 방식 | `seq` (문서별 순차 호출, 기존 Full) / `par` (문서별 호출 k 개 동시 전송) / `batch` (한 호출에 k 문서) |
| 엔진 배칭 | `cb_on` (기본) / `cb_off` (`--max-num-seqs 1`: 요청 간 배칭 불가) |

- Continuous Batching 효과 = `par@cb_on` vs `par@cb_off` (그리고 동시 부하에서의 처리량 차이).
- Batch Grading 고유 효과 = `batch@cb_on` vs `par@cb_on` (엔진 배칭이 이미 있는 상태에서 추가 이득).
- 상호작용 = (`batch` − `par`)@`cb_off` vs (`batch` − `par`)@`cb_on`. 가설: CB 가 켜져 있으면 Batch 이득이 줄어든다.
- 단일 query (sequential) 와 동시 부하 (live, 같은 workload 파일) 두 조건 모두 측정. prefix caching on/off 는
  launch 설정으로 기록하고 주 비교에서는 고정한다.
- 품질 (grader 점수 일치, action 일치, accuracy) 은 `batch` vs `seq` 로 별도 비교 (`par` 와 `seq` 는 프롬프트 동일).

## Correction-aware 정책 가설 (구현 전 제안)

기존 SRPT-est 는 grading 이 끝나 correction 이 확정되면 남은 작업량 추정 (`rewrite_mean + gen_corrected_mean`) 이
커져서 correction query 의 호출이 오히려 뒤로 밀린다 (L-SO: ρ=0.8 Correction p95 484s vs 평균 75s).

| 후보 | 규칙 | SRPT-est 와 다른 점 | 예상 효과 / 반증 조건 |
|---|---|---|---|
| **CA-1 rewrite boost** | correction 확정 시 rewrite 호출만 최우선 | rewrite 는 짧은 호출(출력 수십 토큰)이지만 끝나야 외부 web search 가 시작된다. 먼저 보내면 web search 지연이 다른 query 의 GPU 작업과 겹쳐 숨겨진다 | Correction E2E 가 대략 rewrite 대기시간만큼 감소, 전체 평균 영향 작음. rewrite 대기가 원래 짧으면 효과 없음 |
| **CA-2 slowdown-aware** | 우선순위 = 예상 잔여 / (경과 + 예상 잔여) | 이미 오래 걸린 query 를 잔여량과 함께 고려 (Aging 은 큐 대기만, 이건 E2E 대비) | Correction p95 감소, 평균 증가폭이 Aging 보다 작을 것. Aging 과 차이 없으면 기각 |
| **CA-3 class slack** | 클래스별 목표 지연 (correction 목표는 더 크게) 대비 slack 최소 우선 | 꼬리 지연을 목표 기반으로 제한 | 목표 초과율 감소. 목표 설정에 민감하면 실용성 낮음 |

권장: CA-1 을 먼저 (구현 작고 가설이 명확), 효과 없으면 CA-2. 모두 dev workload 에서 파라미터 고정 후 평가.

## 외부 (Tavily) 지연 분리 설계 (P2 에서 구현)

- 정확도: 실제 Tavily 호출 결과로 측정. rewrite query 와 결과 문서·지연을 함께 기록 (record).
- 성능: 기본은 실제 Tavily (`live`). 같은 run 의 기록을 이용해 `replay_recorded` (같은 문서 + 기록된 지연) 와
  `replay_zero` (같은 문서 + 지연 0) 를 추가로 돌려 외부 지연의 기여를 분리한다.
- rewrite 결과가 달라 기록이 없는 경우 live 호출 후 cache miss 로 기록한다.

## 서버 점검 (사용자 실행)

`scripts/server/README.md` 참고.

## 구현된 구성요소 (GPU 없이 완성, mock 으로 기능 검증)

| 구성요소 | 위치 | 비고 |
|---|---|---|
| 추론 엔진 | `rag.llm.engine`, `ollama_engine`, `vllm_engine`, `engine_llm`, `factory` | 기존 `LLM.generate()` 유지 |
| LangGraph CRAG | `rag.graph.CRAGGraph` | 기존 `CRAG` 와 Full 동등성 테스트 (3분기 × 순차/병렬) |
| Grading 전략 | `rag.grading` (`full`, `lazy_hit`, `lazy_rest`, `batch`, `adaptive`) | 같은 인터페이스 `grade(question, docs, llm, upper, lower)` |
| Scheduler | `rag.scheduling` (`SchedulingGate`, `FCFS`, `FCFSQuery`, `SJFPred`, `SRPTEst`, `Aging`, `WaitThreshold`, `CostModel`, `OOFCostModel`, `LLMGate`) | K slots, 동적 우선순위, vLLM priority 전달 |
| 실행 상태 | `rag.scheduling.state` (`QueryState`, `PolicyView`) | stage / correction / 잔여 추정 / 누적 대기. 정책은 `PolicyView` 로 정책 입력만 읽음 |
| 사전 검증 | `rag.scheduling.preflight` | priority 설정 일치, launch 기록 대조, 서버 probe, cost model 누수 검사 |
| 비교 가능성 검사 | `experiments/crag_system/comparability.py` | run 간 질문·검색 결과·모델·replay 원본 일치, 교란 요인 기록 |
| Web 기록/재생 | `rag.retrieval.web_cache` | 실제 Tavily 기록 → 같은 문서로 지연 recorded / zero 재생 |
| Runner | `experiments/crag_system/runner.py`, `run.py` | sequential / live, 결과 `queries.jsonl`, `calls.jsonl`, `run_meta.json` 외 |
| Workload | `workload.py`, `make_workload.py` | 요청 순서·도착 시각 + sha256, 덮어쓰기 금지 |
| Cost model 학습 | `fit_cost_model.py` | sequential run trace → 모델 json (학습 질문 목록 포함, `--oof K`) |

테스트 (`uv run pytest`) 는 실제 시스템 코드의 기능 검증용이다. mock / 가짜 엔진 결과를 정확도·성능 결과로 쓰지 않는다.

## 실행 예 (서버, GPU 권한을 받은 뒤)

```bash
CFG=experiments/crag_system/config.yaml
# SYS-1: Baseline (vLLM, Full CRAG / Vanilla), 203문항 sequential
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-1b
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-1b --set pipeline=vanilla
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-1b --set pipeline=crag   # 기준 구현
# SYS-1c: 엔진 bridge (같은 조건, Ollama)
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-1c \
  --overlay experiments/crag_system/configs/engine_ollama.yaml

# SYS-2: grading 전략 (sequential). Full 은 실제 Tavily 를 기록하고, 나머지는 그 기록을 replay
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-2a --set grading.strategy=full
FULL=results/crag_system/SYS-2a/<full run>
for s in lazy_hit batch adaptive; do
  uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-2a \
    --set grading.strategy=$s --set crag.web_search.provider=replay \
    --set "crag.web_search.replay_from_runs=['$FULL']" --set crag.web_search.latency_mode=recorded
done
uv run python experiments/crag_system/comparability.py $FULL results/crag_system/SYS-2a/<other runs>

# SYS-3 / SYS-4: 동시 요청 (live). cost model 학습 질문과 평가 질문 분리
uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-3-train \
  --set dataset.split.part=train                               # cost model 학습용 trace
uv run python experiments/crag_system/fit_cost_model.py results/crag_system/SYS-3-train/<run> \
  --service-key wall --out results/crag_system/cost_models/train_wall.json
uv run python experiments/crag_system/make_workload.py --name eval_rho0.8_s0 --n 500 --rho 0.8 \
  --capacity-rps <측정값> --seed 0 --set dataset.split.part=eval
for pol in fcfs sjf_pred srpt_est; do
  uv run python experiments/crag_system/run.py --config $CFG --exp-id SYS-3a \
    --workload results/crag_system/workloads/eval_rho0.8_s0.json --set dataset.split.part=eval \
    --set run.mode=live --set scheduler.policy=$pol --set scheduler.slots=4 \
    --set "scheduler.cost_model={path: results/crag_system/cost_models/train_wall.json}"
done
# vLLM priority 까지 쓰는 조건: 서버를 SCHED_POLICY=priority 로 띄우고 세 설정을 함께
#   --set scheduler.engine_priority=key --set llm.send_priority=true \
#   --set llm.server_scheduling_policy=priority --set llm.launch_meta=~/crag_logs/vllm_<ts>_launch.json \
#   --set run.probe_server_policy=true
```

조건 비교 시 같은 workload 파일, 같은 vLLM launch 설정, 같은 GPU 할당을 쓰고 `comparability.py` 와
`run_meta.json` (`preflight`, `comparability`) 으로 확인한다. 같은 세션에서 조건을 번갈아 돌릴 때는
`--set "run.session={id: s1, order_index: 0}"` 처럼 실행 순서를 기록한다.

## 논문 타당성 관련 구현 (2026-11-10)

| 항목 | 구현 | 오류로 막는 경우 |
|---|---|---|
| 정적 SJF | `SJFPred`: 요청의 첫 호출 때 `pred_total` 을 예측해 요청 끝까지 고정 (request 단위) | cost model 없음 |
| 상태 갱신형 SRPT-est | `SRPTEst`: 호출마다 현재 stage 기준 남은 작업량. 추정치는 `QueryState.est_history` 에 기록 | cost model 없음 |
| 실행 상태 | `QueryState`: stage (`arrived → retrieval → grading → decided → [rewrite → web_search] → generation → done`), corrected, n_graded, cum_wait (정책 입력) / action, est_history, stage_history, retrieval_doc_ids (기록 전용) | 정책이 기록 전용 필드 접근 시 AttributeError |
| priority 일치 | `check_priority_config`, `check_launch_meta`, `probe_server_priority` | engine_priority=key ↔ send_priority 불일치, 서버 선언이 priority 가 아님, launch 인자·모델 불일치, 서버가 priority 거부 |
| cost model 누수 | 모델에 `train_query_ids` 저장, `check_cost_model_leakage`, `OOFCostModel`, `dataset.split` | 학습·평가 질문 중복 (OOF 아니고 `allow_cost_model_overlap` 아님), 학습 질문 목록 없는 모델 |
| 공정 비교 | `retrieval_doc_ids` 기록, `replay_from_runs`, run_meta `comparability` (질문 hash, 설정, replay 원본 sha256, prefix cache·session), run_end `prefix_cache_delta`, `comparability.py` | (검사 스크립트가 critical 차이 보고, exit 1) |

## Correction-aware: SRPT-est 와의 차이 (구현 보류, 정의 확정 후 구현)

- **SRPT-est (구현됨)**: 호출이 대기열에 들어갈 때 "이 query 의 남은 작업량" 추정으로 순서를 정한다.
  grading 이 끝나 correction 이 확정되면 rewrite 호출의 추정치가 `rewrite + correction 후 generation`
  으로 커져서, correction query 는 바로 그 시점부터 뒤로 밀린다. correction 상태를 "비용이 커진 것" 으로만 본다.
- **Correction-aware (제안)**: correction 상태를 비용이 아니라 별도 신호로 쓴다.
  - CA-1 rewrite boost: correction 이 확정된 query 의 rewrite 호출 (출력이 짧음) 만 최우선. web search
    (GPU 를 쓰지 않는 외부 지연) 를 빨리 시작해 다른 query 의 GPU 작업과 겹치게 한다.
  - CA-2 slowdown-aware: 경과시간 대비 남은 작업량으로 우선순위.
  - CA-3 class slack: correction / 비correction 별 목표 지연 대비 여유.
- 구현 위치: `rag.scheduling.policies` 에 Policy 하나 추가. 정책 입력 `PolicyView.stage`,
  `PolicyView.corrected` 가 이미 제공된다 (현재 SRPT-est / SJF 는 corrected 를 쓰지 않음).
  `build_policy({"policy": "correction_aware"})` 는 지금은 NotImplementedError.

## 실제 GPU 에서 검증해야 하는 것

| 항목 | 이유 | 확인 방법 |
|---|---|---|
| vLLM priority 필드 처리 / 낮을수록 먼저 / FCFS 서버의 거부 여부 | 버전마다 다름 | `probe_engine.py` priority, priority_order |
| `VLLM_API_KEY` 환경변수 인증 | 버전마다 다름 | probe auth |
| `/metrics` 지표 이름, `/reset_prefix_cache` 사용 가능 여부 | 버전·dev mode | probe metrics, `run.reset_prefix_cache=true` 결과 (`run_end.json`) |
| Llama 3.1 이 batch grading JSON 을 지키는 비율 | 모델 출력 | SYS-2 `extra.grading.info.batch_status`, fallback 수 |
| 같은 prompt 의 결정성 (temperature 0, batching) | continuous batching 비결정성 | probe determinism, 같은 조건 반복 run 의 답 일치율 |
| K 별 처리량, μ_max (ρ 기준) | 하드웨어 의존 | probe sweep, 포화 부하 live run |
| cost model 예측 정확도 (wall 기반) | vLLM 은 요청 단위 서버 시간 없음. 부하 시 batching 으로 서비스 시간이 달라짐 | train split trace 로 학습 → eval split 에서 예측 오차 확인 |
| FCFS 서버가 priority 요청을 거부하는지 | 버전마다 다름 (수락하면 probe 로 정책 판별 불가) | probe priority, `run.probe_server_policy` 결과 |
| vLLM priority 가 효과를 내는 K | K 가 서버 동시 처리 한도보다 작으면 vLLM 대기열이 비어 priority 무의미 | probe priority_order, K sweep |
| prefix cache 교란 크기 | 조건 순서에 따라 latency 가 달라질 수 있음 | `run_end.prefix_cache_delta`, ABBA 순서 비교 |
| Adaptive 임계값 | 데이터 기반 튜닝 필요 | dev split 에서 튜닝 후 고정 |
| 다른 사용자 GPU 간섭 | 공유 서버 | `run_meta.json` / `run_end.json` 의 `gpu`, `gpu_after` |
