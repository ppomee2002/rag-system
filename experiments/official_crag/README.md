# Official CRAG Baseline (C1 공통 연구 기반)

공식 CRAG ([HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG), 논문 [arXiv:2401.15884](https://arxiv.org/abs/2401.15884))
의 Retrieval Evaluator·판정·Knowledge Refinement 를 기준으로, 기존 serving 인프라 (vLLM, Llama 3.1 8B AWQ,
FCFS-query scheduler) 위에서 C1·C2·C3 연구를 수행하기 위한 공통 baseline.

- **MODE A — Official Reproduction**: 공식 데이터·evaluator·지식 파일·prompt 로 공식 경로를 단계별로 재현.
- **MODE B — Controlled Server Comparison**: 기존 PopQA 230문항 Top-5 를 고정하고 grader (공식 T5 / 기존 Llama) 와
  evidence 정책만 바꿔 정확도·비용을 비교. **공식 논문 수치의 재현이 아니다.**

> 이 문서의 모든 수치·자원 사용량은 서버 실행 전 추정치이다. 실제 GPU 결과는 아직 없다.

---

## 1. 공식 CRAG 분석 (기준 commit `de7c2961ae624a1483a138c5798e1f6d0c4fb0e0`)

| 항목 | 공식 구현 | 근거 |
|---|---|---|
| Evaluator 모델 | T5-large 를 회귀로 fine-tune 한 `T5ForSequenceClassification(num_labels=1)`. 학습 label = ±1 (MSE) | `train_evaluator.py:24-55` |
| 가중치 | Google Drive 폴더 `finetuned_t5_evaluator`: `config.json`, `model.safetensors` (2.75GB), `spiece.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` (2023-12-23). 버전 표기 없음 | README "Download" |
| 입력 | `"질문 [SEP] 문서"` 한 쌍, `padding="max_length", max_length=512`, **truncation 없음** | `CRAG_Inference.py:165` |
| 점수 | logit 하나 (범위 제한 없음). 빈 문서는 모델 없이 -1.0 | `CRAG_Inference.py:151-171` |
| **평가 단위** | **문서별 독립 점수** → 질문 단위 `any()` 집계 | `CRAG_Inference.py:174-195` |
| 판정 | flag: score≥upper→2, ≥lower→1, else 0. 질문: 2 가 있으면 Correct, 아니면 1 이 있으면 Ambiguous, 아니면 Incorrect | 같음 |
| Threshold (PopQA) | upper **0.592**, lower **-0.995** (`--lower_threshold 0.995` 를 코드에서 음수화) | `run_crag_inference.sh:15`, `CRAG_Inference.py:221` |
| Correct 지식 | 검색 문서 **전부**를 strip 으로 분해 (기본 `selection` = 문서 그대로) → evaluator 점수순 상위 3 (`excerption`/`fixed_num` 은 6) → `'; '` 결합. **threshold 로 거르지 않음** | `internal_knowledge_preparation.py`, `utils.py:274-299` |
| Incorrect 지식 | GPT-3.5 로 키워드 추출 → `"Wikipedia, "` 접두 → serper(Google) 상위 5 중 wikipedia 링크 페이지 → 문단 (`"제목: 문단"`, 10자 초과) → evaluator 상위 5 | `external_knowledge_preparation.py`, `utils.py:231-271` |
| Ambiguous 지식 | `"Knowledge1: <correct> [sep] Knowledge2: <incorrect>"` | `combined_knowledge_preparation.py:14` |
| 지식 준비 방식 | **모든 질문에 대해 세 지식을 오프라인으로 미리 생성**, inference 는 판정에 맞는 줄만 고름 (`ref/*` 파일 저장소에 포함) | `run_crag_inference.sh` |
| Generation | `"Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {지식}\n\nInstruction: Answer the question: {질문}"`, chat template 없음, temperature 0 / top_p 1 / max_tokens 100, 제어 토큰·개행 제거 | `CRAG_Inference.py:41-91,224` |
| Generator | LLaMA2-hf-7b, SelfRAG-LLaMA2-7b (vLLM 0.2.5 offline, fp16) | README, `requirements.txt` |
| 데이터 | Self-RAG PopQA long-tail 1,399문항 + Contriever 검색 상위 10 (`data/popqa/test_popqa.txt`, 빈 문서 1,321개 포함) | `data_process.py` |
| 평가 | `match`: 정답 문자열이 예측에 포함 (대소문자 구분, 정규화 없음) | `metrics.py:83-87`, `eval.py` |
| 라이선스 | **LICENSE 파일 없음, GitHub license 필드 null** → 코드를 이 레포에 복사하지 않음 | GitHub API (2026-11-23) |

### 공식 코드의 알려진 결함 (재구현에서 처리 방식)

| 결함 | 처리 |
|---|---|
| `combined_knowledge_preparation.py`: import 누락·미정의 변수·언팩 오류로 실행 불가 | 템플릿만 재현, ambiguous 는 저장소 제공 파일 사용 (MODE A) |
| `utils.py:231/240` keyword prompt 의 `{{question}}` 이 format 후 리터럴 `{question}` → 질문이 GPT 에 안 들어감 | 질문을 넣는다 (의도된 동작). 공개된 `ref/incorrect` 는 질문별 내용이라 다른 버전으로 생성된 것으로 보임 |
| `external_knowledge_preparation.py:154` `select_relevants(mode=...)` 인자 없음 → TypeError, `:180` 앞 10문항만 처리 | external 은 MODE A 에서 저자 파일 재사용 |
| `inference()` 가 질문당 10줄 가정 (`n % 10`) | `ndocs=10` 에서만 공식과 동일 (MODE A 는 10 고정) |
| `fixed_num` 분해에서 10단어 미만 문서는 IndexError | strip 하나로 둠 (차등 테스트에서 해당 경우만 제외 확인) |
| internal/external 지식 줄의 `'#'` 접두가 generation 입력에 남음 | MODE A 는 공식대로 유지, MODE B 는 직접 생성하므로 없음 |

### 차등 테스트 (공식 코드 vs 재구현)

`tests/test_official_differential.py` 는 공식 저장소 (`third_party/CRAG` 또는 `CRAG_OFFICIAL_REPO`) 의 함수를 직접
import 해서 실제 PopQA 데이터로 비교한다 (evaluator 는 양쪽에 같은 가짜 점수 함수). 로컬에서 **12개 모두 통과**:
`data_preprocess`, `process_flag`, `inference` 의 빈 문서 규칙, strip 분해 3모드 (300문항), `select_relevants`
순위·결합 결과 (selection / excerption, 150문항), generation prompt, 후처리, `match`.

---

## 2. 기존 프로젝트와의 차이 (Phase 0 감사)

### 2.1 기존 흐름 (exp/crag-system)

| 단계 | 파일 / 함수 | 내용 |
|---|---|---|
| Retrieval | `rag.retrieval.faiss_retriever.FaissRetriever` | bge-base-en-v1.5 + FAISS, corpus = PopQA subject/object Wikipedia 문서 chunk (`prepare_data.py`, `build_index.py`), top-5 |
| Grading | `rag.grading` (`FullGrading` 등) → `prompts.GRADE` + `parse_score` | Llama 가 문서마다 0~1 숫자 출력 |
| Correction | `rag.graph.CRAGGraph` `_decide/_rewrite/_web_search` | correct: score≥0.7 문서 / ambiguous: ≥0.3 문서 + web / incorrect: web 만. `prompts.REWRITE` + Tavily (기록·재생) |
| Generation | `prompts.GENERATE` + `format_context` | chat API (vLLM `/v1/chat/completions`) |
| 평가 | `rag.eval.metrics.accuracy` | 정규화 후 substring |
| 인프라 | `rag.llm.*_engine`, `EngineLLM`, `rag.telemetry`, `rag.scheduling`, `experiments/crag_system/runner.py` | 호출 기록, gate, run_meta, 덮어쓰기 금지 |

### 2.2 공식 vs 기존 비교

| | 공식 CRAG | 기존 (exp/crag-system) |
|---|---|---|
| Grader | T5-large 회귀 (전용 모델) | Llama 3.1 8B prompt |
| 점수 척도 | logit (대략 ±1, 무제한) | 0~1 (첫 숫자 파싱, clip) |
| Threshold | 0.592 / -0.995 | 0.7 / 0.3 |
| 평가 단위 | 문서별 독립 점수 → 질문 단위 any() | 문서별 독립 호출 → 질문 단위 any() |
| 판정 규칙 | 2 있으면 C, 1 있으면 A, 아니면 I | any≥upper → C, all<lower → I, 그 외 A — **논리적으로 동일** (테스트로 확인) |
| Correct evidence | 모든 문서 strip 중 점수 상위 3 (필터 없음) | score≥upper 문서만 |
| Ambiguous evidence | 지식 문자열 결합 (internal refine + external refine) | score≥lower 문서 + web 문서 원문 |
| Incorrect evidence | refine 된 web 문단 상위 5 | web 문서 원문 (Tavily content) |
| 검색어 | GPT-3.5 키워드 3개 + "Wikipedia, " | Llama REWRITE prompt |
| Web | serper(Google) + 페이지 HTML 문단 | Tavily (snippet content) |
| Prompt | 공식 PopQA prompt, chat template 없음, 100 토큰 | `GENERATE`, chat template, 512 토큰 |
| 데이터·검색 | Self-RAG PopQA + Contriever top-10 (공식 제공) | HF PopQA + 자체 Wikipedia corpus + bge top-5 |
| 평가 | 대소문자 구분 substring | 정규화 substring |

→ 핵심: **판정 규칙은 같고, 점수 모델·척도와 evidence 구성 (특히 Correct 의 필터 vs refinement) 이 다르다.**
기존 Full CRAG 의 정확도 하락 원인 후보 중 "Correct 에서 upper 미만 문서를 지우는 규칙" 은 공식 CRAG 에는 없다.

### 2.3 재사용 판단

| 구성요소 | 분류 | 비고 |
|---|---|---|
| `rag.llm` (VLLMEngine, EngineLLM, factory), `rag.telemetry` (CallRecorder, run_meta, JsonlWriter) | 그대로 재사용 | |
| `rag.scheduling` (`SchedulingGate`, `fcfs_query`, `QueryState`) | 그대로 재사용 | MODE B 기본 FCFS-query |
| `rag.retrieval.web_cache` (기록/재생, query_id replay) | 그대로 재사용 | 정책 간 같은 web 결과 |
| `rag.eval.metrics`, `rag.pipelines.prompts`, `rag.grading.base.parse_score` | 그대로 재사용 | legacy prompt / Llama grader |
| 기존 crag_system run 의 `retrieval_doc_ids` + chunk 파일 | Adapter (`freeze_inputs.py`) | Top-5 고정 |
| VLLMEngine (chat) | Adapter 추가 (`VLLMCompletionEngine`) | MODE A 는 chat template 없는 completions |
| Grading 판정·evidence 구성 | 공식 로직으로 교체 (새 경로) | `rag.crag_official`, `rag.evidence` |
| `rag.graph.CRAGGraph`, `rag.grading.*`, `ablation.py`, `rag.pipelines.crag.CRAG`, crag_system runner·분석 스크립트·결과 | Legacy 보존 (수정 없음) | crag_rule / ablation 정책이 같은 context 를 내는지 테스트로 확인 |
| T5 evaluator, refinement, 공식 데이터 로더, C2 상태, C3 budget, MODE A/B runner | 별도 구현 | |

**이번 작업에서 기존 파일 수정:** `.gitignore` (third_party/·models/·가중치 확장자), `pyproject.toml` + `uv.lock`
(선택 extra `official` = sentencepiece, protobuf 추가), `CLAUDE.md` (실험 목록 한 줄).
**`src/rag/` 는 새 패키지 추가만** (`rag.crag_official`, `rag.evidence`), 기존 모듈·인터페이스·`configs/base.yaml` 은 변경 없음.

### 2.4 확인이 필요한 사항

- **"230개 공통 문항" 은 이 레포의 어느 브랜치에도 없다** (crag_system config 는 `limit: 203`). 문항 ID 목록 파일은
  서버 결과에서 사용자가 지정한다 (`freeze_inputs.py --question-ids`). 이 집합은 공통 문항으로 선택된 것일 수 있어
  전체 PopQA 일반화 결과로 쓰지 않는다 (manifest `caveat` 에 기록).

---

## 3. 코드 구조

```
src/rag/crag_official/        # 공식 CRAG 재구현 (공통 코어, 추가만)
  source.py                   # 출처·commit·라이선스·상수·prompt·알려진 결함 (줄 번호 근거)
  evaluator.py                # T5 backend / 캐시 / mock, 공식 flag·판정, 가중치 검사, CRAG_EVALUATOR_PATH
  refinement.py               # strip 분해, select_relevants, internal / external / ambiguous 지식
  generation.py               # 공식 PopQA prompt, 후처리, official_match
  official_data.py            # 공식 test_popqa.txt / ref 파일 / 정답 파일 정렬
  completion_engine.py        # vLLM /v1/completions (chat template 없음)
src/rag/evidence/             # C1/C2/C3 공통 인터페이스 (MODE B)
  graders.py                  # DocumentGrader: OfficialT5Grader, LlamaPromptGrader
  state.py                    # C2: GradingTracker / GradingSnapshot
  budget.py                   # C3: stop_rule x retention
  policies.py                 # C1: evidence 정책 정의
  frozen.py                   # 고정 입력 (질문·정답·Top-k) + manifest
  pipeline.py                 # ControlledCRAG
  schema.py                   # 결과 필드
experiments/official_crag/
  config.yaml                 # MODE B 기본 설정
  configs/mode_a_popqa.yaml   # MODE A generator 설정 + deviations
  fetch_official.py           # 공식 저장소 고정 commit, 가중치 다운로드·검사·hash
  smoke_evaluator.py          # T5 단독 smoke test
  mode_a.py                   # MODE A: score / verify-refinement / generate / evaluate
  freeze_inputs.py            # MODE B 고정 입력
  precompute_scores.py        # T5 점수 캐시 (GPU 순차 실행)
  run.py, mode_b.py           # MODE B runner
  analyze.py                  # 조건 비교 (paired, McNemar, 판정 일치)
  server/c1_matrix.sh         # C1 조건 행렬 실행
  server/backup_results.sh    # /workspace/rag01 최소 백업
```

---

## 4. MODE B 조건

### 4.1 Grader

| `grader.type` | 점수 | threshold (기본) | 비고 |
|---|---|---|---|
| `official_t5` (기본) | T5 logit | `evaluator.task=popqa` → 0.592 / -0.995 | `evaluator.upper/lower` 로만 변경 |
| `llama_prompt` (legacy) | 0~1 | `grader.upper/lower` = 0.7 / 0.3 | 기존 GRADE prompt, 생성 엔진과 같은 vLLM |

두 threshold 는 서로 다른 config 키이며 통일하지 않는다. 결과의 `score_scale`, `thresholds` 에 기록된다.

### 4.2 Evidence 정책 (`evidence_policy`)

| 정책 | correct | ambiguous | incorrect | 출처 |
|---|---|---|---|---|
| `vanilla` | top-5 (grading 없음) | | | `VanillaRAG` |
| `crag_rule` | score≥upper | score≥lower + web | web | 기존 `CRAGGraph` (동등성 테스트) |
| `no_correct_filter` | top-5 | score≥lower + web | web | Ablation B (동등성 테스트) |
| `no_incorrect_discard` | score≥upper | score≥lower + web | top-5 + web | Ablation A (동등성 테스트) |
| `keep_all` | top-5 | top-5 + web | top-5 + web | 신규 |
| `official` | refine (상위 3) | Knowledge1/2 결합 | web refine (상위 5) | 공식 |

`policy_id = <정책>@<grader>[+budget]`. `official@official_t5` 만 `official_policy_semantics=true`.
`official@llama_prompt` 는 공식 정책 구조에 Llama 점수를 넣은 비공식 조합 (selection 모드에서 strip=문서라
grading 점수를 재사용, 추가 호출 없음).

### 4.3 MODE B 가 공식과 다른 점 (결과 해석 시 필수)

| 항목 | MODE B | 공식 |
|---|---|---|
| 검색 | 기존 bge + 자체 corpus top-**5** (고정) | Contriever top-10 |
| Generator | Llama 3.1 8B AWQ INT4, vLLM 0.33 | LLaMA2-7B fp16 |
| Prompt | `generation.prompt=legacy` (기본, 모든 정책 동일) 또는 `official` | 공식 PopQA prompt |
| 검색어 | `correction.query_mode=legacy_rewrite` (기본) / `official_keywords` (공식 few-shot 을 Llama 로) | GPT-3.5 키워드 |
| Web | Tavily 기록 재생 (query_id 기준, 정책 간 동일) | serper + 페이지 HTML |
| External refine | Tavily content 를 줄 단위 문단으로 나눠 상위 5 | 페이지 `<p>` 문단 상위 5 |
| 평가 | `is_correct` = 기존 정규화 acc, `metrics.official_match` 함께 기록 | match |

### 4.4 Web 실패 처리

`correction.status`: `ok` / `empty` / `error` / `replay_miss` / `replay_miss_empty`, `correction.valid` 는 ok 일 때만 true.
`on_web_error=fail` (기본) 이면 오류 행 (정확도 집계 제외, summary 에 개수). `continue` 면 web 없이 진행하되 valid=false.
Tavily 키가 없으면 엔진 생성 시점에 실패한다 (조용히 빈 결과로 진행하지 않음).

---

## 5. GPU 메모리 및 실행 방식 (추정 — 서버에서 확인 필요)

### 5.1 T5 evaluator 요구량

| 항목 | 추정 | 근거 |
|---|---|---|
| 파라미터 | 약 7.4억 (T5-large encoder+decoder + 분류 head) | t5-large 구조 |
| 가중치 (fp32, 공식) | **약 2.75 GiB** | safetensors 2.75GB ≈ 7.4억 x 4B |
| 추론 activation (batch 1, 512 토큰, encoder+decoder) | 수백 MB | 24+24 layer, d_model 1024 |
| CUDA context | 약 0.3~0.5 GiB | 프로세스당 |
| **합계 (batch 1)** | **약 3.2~3.8 GiB** | `smoke_evaluator.py` 의 `gpu_mem_after_load_gib`, `gpu_peak_gib` 로 확인 |
| fp16/bf16 | 약 절반 | **비공식** (T5 fp16 overflow 위험). `evaluator.dtype` 은 기본 float32 |

### 5.2 vLLM 과 같은 GPU 사용 여부

A2 15GB 에서 vLLM `gpu_memory_utilization=0.85` 는 총 메모리의 85% 를 예약한다 → 남는 양은 약 2 GiB 안팎으로
T5 (약 3.2~3.8 GiB) 보다 작을 가능성이 높다. **같은 GPU 동시 실행은 OOM 또는 vLLM 시작 실패가 예상된다 (미검증).**

| 방식 | 설정 | 장점 | 제약 / 측정 주의 |
|---|---|---|---|
| **A. 다른 GPU (권장)** | vLLM: `CUDA_VISIBLE_DEVICES=0`. runner: `CUDA_VISIBLE_DEVICES=1` + `evaluator.device=cuda:0` | 기존 vLLM 설정 그대로, 경합 없음 | 두 번째 A2 가 비어 있어야 함 (공유 서버: `nvidia-smi` 확인, run_meta `gpu` 기록). grading 시간은 "전용 GPU" 비용 |
| B. 같은 GPU + vLLM 예약 축소 | `launch_vllm.py --gpu-memory-utilization 0.6` (예시 값) | GPU 1장 | **KV cache·동시 처리량이 줄어 기존 0.85 결과와 성능 비교 불가**. T5·vLLM 연산 경합으로 latency 교란 |
| C. CPU evaluator | `evaluator.device=cpu` | GPU 영향 없음, 정확도는 동일 | 느림 (쌍당 수백 ms~수 초, 추정). grading latency 를 GPU 조건과 비교하지 말 것 |
| D. 순차 (점수 캐시) | vLLM 끄고 `precompute_scores.py` (GPU) → vLLM 켜고 `evaluator.backend=cache` | GPU 1장, 정확도 동일 | run 의 grading 시간 = 조회 시간. 비용은 `evaluator_compute_seconds` (단독 측정값) 로 따로 보고 |

정확도 비교는 어떤 방식이든 같은 점수를 쓰면 같다 (캐시 key = 입력 텍스트 hash). latency·GPU 비용 비교는 방식을 섞지 말고
run_meta 의 `evaluator.backend.device` 로 구분한다.

### 5.3 측정 시 주의

- `stage_timings` 는 wall-clock. `evaluator_compute_seconds` 는 토크나이즈+forward (CUDA synchronize 포함) 합. GPU 커널 시간이 아니다.
- vLLM 은 요청 단위 대기시간을 주지 않음 → `queue_wait.engine_queue=null`. client gate 대기만 `queue_wait.client_gate`.
- 공식 refinement 는 판정 점수 (truncation 없음) 와 별도로 strip 점수 (truncation=True) 를 다시 계산한다 (공식 그대로). official 정책의 evaluator 비용은 grading + refinement 합.

---

## 6. 서버 실행 안내 (사용자 실행, `/home/rag01/RAG`)

### 6.1 설치

```bash
cd /home/rag01/RAG
git fetch origin && git checkout feature/official-crag-baseline && git pull
uv sync --extra crag --extra official            # sentencepiece, protobuf 추가

# 공식 저장소 (고정 commit, third_party/CRAG — git 에 들어가지 않음)
uv run python experiments/official_crag/fetch_official.py repo

# 공식 evaluator 가중치 (Drive 폴더 전체, 2.75GB). 프로젝트 환경을 건드리지 않도록 uvx 사용
mkdir -p ~/models/crag_evaluator
uvx gdown --folder "https://drive.google.com/drive/folders/1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7" -O ~/models/crag_evaluator
#   (실패 시 브라우저로 6개 파일을 받아 ~/models/crag_evaluator/finetuned_t5_evaluator/ 에 복사)
export CRAG_EVALUATOR_PATH=~/models/crag_evaluator/finetuned_t5_evaluator   # 또는 config evaluator.model_path
uv run python experiments/official_crag/fetch_official.py verify --weights "$CRAG_EVALUATOR_PATH" --hash
```

공식 가중치를 받지 못하면 **다른 모델로 대체하지 않는다**: `evaluator.backend=t5` 는 가중치가 없으면
`EvaluatorWeightsError` 로 중단한다.

### 6.2 T5 단독 smoke test (vLLM 불필요)

```bash
nvidia-smi                                                   # GPU1 이 비어 있는지
CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
    python experiments/official_crag/smoke_evaluator.py --device cuda:0
uv run --extra crag --extra official python experiments/official_crag/smoke_evaluator.py --device cpu --n-queries 2
# 공식 코드와의 차등 테스트 (third_party/CRAG 사용)
uv run --extra crag --extra official pytest -q tests/test_official_differential.py
```

결과: `results/official_crag/smoke/evaluator/<ts>.json` — 버전, 가중치 구성, 로딩 시간, GPU 메모리
(`gpu_mem_after_load_gib`, `gpu_peak_gib`), 공식 PopQA 첫 N문항 점수·판정, 결정성, 토큰화 동등성.
확인할 것: `status=ok`, `tokenization_check.all_equal=true`, 빈 문서 -1.0, 점수가 대략 ±1 근처에 분포.

### 6.3 MODE A (공식 재현)

```bash
# 1) evaluator 점수 (13,990 쌍. GPU 권장)
CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
    python experiments/official_crag/mode_a.py score --device cuda:0
# 2) refinement 재현 확인 (재계산한 Correct 지식 vs 저자 ref/correct)
CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
    python experiments/official_crag/mode_a.py verify-refinement --device cuda:0 --limit 200
# 3) generation (vLLM 을 GPU0 에 띄운 뒤. chat template 없는 completions)
uv run python experiments/official_crag/mode_a.py generate --method crag \
    --scores results/official_crag/mode_a/score/<run_id>
uv run python experiments/official_crag/mode_a.py generate --method rag     # 공식 RAG baseline (top-10)
# 4) 평가 (정답: Self-RAG eval_data 의 popqa_longtail_w_gs.jsonl. Self-RAG README "Download data")
uv run python experiments/official_crag/mode_a.py evaluate \
    --preds results/official_crag/mode_a/generate_crag/<run_id> \
    --answers <popqa_longtail_w_gs.jsonl> [--subset-ids <230문항 id 파일>]
```

정답 파일 대신 `data/processed/popqa_longtail.jsonl` 을 줄 수도 있지만 위치·질문 문자열이 모두 같아야 하며,
다르면 실패한다. MODE A 의 generator 는 기본으로 Llama 3.1 8B AWQ (공식과 다름, `deviations` 에 기록).
서버에 LLaMA2-7B 가 있으면 `--set llm.model=...` 로 바꾼다.

### 6.4 MODE B (통제 비교)

```bash
# 1) 고정 입력 (230문항 ID 파일 + 기존 Full run 의 Top-5)
uv run python experiments/official_crag/freeze_inputs.py \
    --question-ids <230문항 id 파일> \
    --reference-run results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726 \
    --reference-run <같은 Top-5 를 써야 하는 다른 기존 run (선택)> \
    --live-check --config experiments/crag_system/config.yaml \
    --out results/official_crag/inputs/popqa_c1_230.jsonl
# 2) vLLM (기존 프로필, GPU0. scripts/server/README.md 의 vLLM 전용 venv 에서)
CUDA_VISIBLE_DEVICES=0 python scripts/server/launch_vllm.py \
    --profile experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
# 3) 공유 correction 검색 캐시 (모든 질문 1회 Tavily 검색, 9.3 절). 이후 C1/C2/C3 는 재생만
export FROZEN=results/official_crag/inputs/popqa_c1_230.jsonl
uv run --extra crag --extra official python experiments/official_crag/prefetch_web.py \
    --overlay experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml \
    --set "inputs.frozen_path=$FROZEN" --name popqa230_legacy_rewrite
export WEB_CACHE=results/official_crag/web_cache/popqa230_legacy_rewrite/web_cache.jsonl
#    (기존 crag_system 기록을 쓰려면 WEB_CACHE 대신 REPLAY_RUN=<run 폴더>: 검색어가 같을 때만 재생)
# 4) smoke (10문항, 공식 T5 + 공식 정책)
CUDA_VISIBLE_DEVICES=1 EVAL_DEVICE=cuda:0 LIMIT=10 PURPOSE=smoke PREFIX=SMOKE-C1 ONLY="T5-OFF LL-RULE VAN" \
    bash experiments/official_crag/server/c1_matrix.sh
# 5) 전체 C1 행렬 (pre) -> 분석 (C2 Correction 분석은 9.4 절)
CUDA_VISIBLE_DEVICES=1 EVAL_DEVICE=cuda:0 PURPOSE=pre PREFIX=PRE-C1 bash experiments/official_crag/server/c1_matrix.sh
uv run python experiments/official_crag/analyze.py results/official_crag/pre/PRE-C1-*/* \
    --baseline results/official_crag/pre/PRE-C1-VAN/<run_id> --name c1_pre
```

`launch_vllm.py` 의 venv·검사 옵션 (`--smoke`, `--dry-run`, launch 기록) 은 `scripts/server/README.md` 를 따른다. 본실험은 `PURPOSE=paper PREFIX=C1
PAPER=--paper-approved` (git clean, `--limit` 없음 필요). GPU 1장만 쓸 때는 5.2 의 D (캐시) 방식:

```bash
# vLLM 을 띄우기 전
uv run --extra crag --extra official python experiments/official_crag/precompute_scores.py \
    --set "inputs.frozen_path=$FROZEN" --set "web_search.replay_paths=[$WEB_CACHE]" \
    --set evaluator.device=cuda:0 --out results/official_crag/score_cache/popqa_c1_230_t5.jsonl
# vLLM 을 띄운 뒤
EVAL_BACKEND=cache EVAL_CACHE=results/official_crag/score_cache/popqa_c1_230_t5.jsonl \
    PURPOSE=pre PREFIX=PRE-C1C bash experiments/official_crag/server/c1_matrix.sh
```

### 6.5 결과·백업

- MODE A: `results/official_crag/mode_a/<step>/<run_id>/` (scores.jsonl, preds.jsonl, preds.txt, summary.json)
- MODE B: `results/official_crag/<purpose>/<exp_id>/<run_id>/` (config.yaml, run_meta.json, manifest.json, queries.jsonl,
  calls.jsonl, grading_docs.jsonl/.csv, summary.json [, web_cache.jsonl: tavily 기록 또는 replay_on_miss=live])
- 공유 검색 캐시: `results/official_crag/web_cache/<name>/` (web_cache.jsonl, calls.jsonl, manifest.json)
- 분석: `results/official_crag/analysis/<name>/`
- 출처 기록: `results/official_crag/provenance/` (공식 commit, 가중치 sha256)
- 백업 (핵심 CSV/JSON, 압축 jsonl, 고정 입력, provenance 만):
  `bash experiments/official_crag/server/backup_results.sh <run 또는 analysis 폴더>...` → `/workspace/rag01/official_crag/<ts>/`

### 6.6 오류 점검

| 증상 | 확인 | 조치 |
|---|---|---|
| `EvaluatorWeightsError: 경로가 없음` | `echo $CRAG_EVALUATOR_PATH` | 환경변수 또는 `evaluator.model_path` |
| `필수 파일 누락 [...]` | `ls $CRAG_EVALUATOR_PATH` | Drive 폴더 6개 파일 전부 (특히 `spiece.model`, `model.safetensors`) |
| `model_type=... (t5 가 아님)` | `config.json` | 다른 폴더를 가리킴 |
| sentencepiece / protobuf ImportError | `uv pip list | grep -i sentencepiece` | `uv sync --extra crag --extra official` |
| transformers 버전 경고 (unexpected/missing keys), 점수 이상 | smoke 결과 `versions`, `weights.transformers_version_saved` (가중치는 4.36 계열 추정), `tokenization_check` | 로컬 검증은 transformers 5.18 (무작위 가중치) 에서 로딩·토큰화 동등성 통과. 공식 가중치에서 키 불일치가 나면 smoke 보고서를 공유 |
| `CUDA out of memory` | `nvidia-smi` (vLLM·다른 사용자 점유), smoke `cuda_before` | 다른 GPU (`CUDA_VISIBLE_DEVICES`), `--device cpu`, 또는 캐시 방식 (5.2) |
| vLLM 시작 실패 (memory) | T5 가 같은 GPU 에 먼저 올라감 | T5 를 다른 GPU/CPU 로, 또는 vLLM 을 먼저 띄우고 T5 는 다른 장치 |
| `evaluator score cache miss` | 캐시 생성 때와 입력·replay 기록이 같은지 | 같은 `FROZEN`·`REPLAY_RUN` 으로 다시 precompute |
| `web search replay_miss` | 행의 `correction.replay.miss_reason` (`query_mismatch` = 같은 질문인데 검색어가 기록과 다름, `search_condition_mismatch` = provider/params/query_mode 다름, `not_recorded`) | `WEB_CACHE` 가 같은 `correction.query_mode`·generator 로 만든 것인지 확인. 필요 시 `web_search.replay_on_miss=live` (실제 검색 + 기록) 또는 `empty` (invalid 로 남음) |
| `고정 입력 검사 실패` | 메시지의 질문 id | ID 목록·기준 run·chunk 파일이 같은 데이터인지 |

---

## 7. 결과 로그 (MODE B `queries.jsonl`, `rag.evidence.schema`)

`run_id, request_id, question_id, experiment_mode, grader_type, score_scale, thresholds, evidence_policy, policy_id,
budget, official_policy_semantics, correction_path, decision, retrieval_doc_ids, grading_events, grading_snapshots,
refinement_events, correction, evidence{kept_retrieved_ids, removed_retrieved_ids, added_ids, selected_evidence_ids,
units, n_units, n_chars}, generation_input_tokens, generation_output_tokens, llm_calls, stage_timings
(retrieval/grading/rewrite/web_search/generation/refinement), evaluator_compute_seconds, queue_wait{client_gate,
engine_queue}, e2e_latency, prediction, ground_truth, is_correct, metrics{acc, em, f1, official_match}, error`.

- 측정하지 못한 값·실행하지 않은 stage 는 `null` (0 아님).
- `run_meta.json`: git commit/dirty, 패키지, GPU 상태, 엔진 정보, evaluator 설정·가중치 fingerprint, 공식 commit,
  고정 입력 sha256·manifest, seed, pipeline 설정 (`not_official_reproduction` 문구 포함).
- 텍스트 원문 (evidence, prompt) 은 `run.record_texts=true` 일 때만.

---

## 8. 팀원 인계

### C2 — 중간 Grading 상태

- 사용 모듈: `rag.evidence.state` (`GradingTracker`, `GradingSnapshot`), 결과의 `grading_events`, `grading_snapshots`.
  문서별 로그 (`grading_docs.jsonl/.csv`), manifest, 공유 검색 캐시, Correction 분석은 **9 절**.
- snapshot 필드: `n_evaluated, n_total, n_passed, n_partial, n_failed, cum_input_tokens, cum_output_tokens, cum_latency,
  passed_ids, partial_ids, evaluated_ids, provisional_action, action_final, first_pass_rank`.
- **공식 판정과의 관계**: T5 점수는 문서별로 독립이라 prefix 점수는 전체 평가 때와 같다. `provisional_action=correct`
  이면 최종 판정도 correct (`action_final=true`). ambiguous/incorrect 는 남은 문서에 따라 바뀔 수 있어 공식 판정이 아니다.
  공식 Correct 의 evidence 는 모든 문서의 refinement 점수가 필요하므로 snapshot 의 `passed_ids` 는 공식 evidence 가 아니다.
- 공식 evaluator 의 평가 방식은 바꾸지 않았다. 문서별 순차 기록은 같은 점수를 순서대로 남기는 것뿐이다.
- "중단 시 품질" 은 `budget.stop_rule=max_docs:N` 등으로 실제 중단 run 을 만들거나, full run 의 snapshot 으로 오프라인
  재구성한다 (후자는 generation 결과가 없음).

### C3 — Evidence Budget Control

- 사용 모듈: `rag.evidence.budget`. 두 변수를 분리: `budget.stop_rule` (full | first_pass | max_docs:N) 과
  `budget.retention` (policy | hit_only | hit_and_rest | all | ratio).
- preset: `full_grading`, `lazy_hit` (= 기존 LazyGrading hit, 동등성 테스트), `lazy_rest` (= 기존 rest, 동등성 테스트),
  `keep_all`.
- **"첫 통과 문서" vs 공식 Correct**: 통과 조건 (flag 2, score≥upper) 은 공식 Correct 의 충분조건과 같아 **판정은 같다**.
  그러나 공식 Correct 의 **evidence 는 다르다** (전체 refinement 상위 3). 그래서 `official` 정책 + lazy 조합은 거부된다.
  C3 실험은 `crag_rule` 등 legacy 정책 위에서, grader 는 `official_t5` 또는 `llama_prompt` 로 한다.
- `ratio` retention 은 **정의 미확정 placeholder** (후보 앞 ceil(ratio x 후보 수) 개). "Lazy-rest 25%" 의 정확한 정의는
  C3 담당자가 확정해 `RatioRetention` 을 수정하거나 새 `Retention` 을 추가한다.
- 동적 정책: `StopRule.should_stop(snapshot)` / `Retention.select(docs, hit_rank, action)` 을 구현해 `build_budget` 에 등록.

### 설정 바꾸기

`--set grader.type=llama_prompt`, `--set evidence_policy=keep_all`, `--set budget.preset=lazy_rest`,
`--set budget.stop_rule=max_docs:2 --set budget.retention=all`, `--set evaluator.device=cuda:0`,
`--set generation.prompt=official`, `--set correction.query_mode=official_keywords --set correction.wiki_prefix=true`.

### 아직 구현·검증되지 않은 것

- 실제 공식 가중치 추론, GPU 메모리, 처리 시간, MODE A/B 정확도 — **서버 실행 전 결과 없음**.
- MODE A 의 external 지식 재생성 (GPT-3.5 + serper + 페이지 수집): 저자 파일 재사용만 지원.
- 공식 refinement 재구현이 저자 `ref/correct` 를 실제 가중치로 재현하는지 (`verify-refinement` 로 서버에서 확인).
- Holdout / 확장 평가 split (230문항 외): `freeze_inputs.py` 로 다른 ID 목록을 고정하면 같은 runner 로 실행 가능하나
  split 정의는 미정.
- MODE B live 모드는 가짜 엔진 테스트만 통과 (vLLM 부하 검증은 서버).

---

## 9. C2 지원: 문서별 Grading 로그 · Manifest · 공유 검색 캐시 · Correction 분석

공식 CRAG 구현·공식 판정 로직은 바꾸지 않았다. 아래는 모두 기존 기록을 펼치거나 (로그), 기존 재생 기능을 확장하거나
(캐시), 결과를 읽기만 하는 (분석) 추가 기능이다.

### 9.1 문서별 Grading 로그 (`grading_docs.jsonl` / `.csv`, `rag.evidence.doc_log`)

grading 을 하는 MODE B run 마다 **질문 x 검색 문서 한 행**. 연결 키: `(request_id, eval_order)` →
`queries.jsonl` 의 `grading_events[eval_order]`, `snapshot_index` → `grading_snapshots[snapshot_index]`
(이 문서까지 평가한 뒤의 누적 상태). 검색 순서대로 평가 상태를 복원할 수 있다.

| 필드 | official_t5 | legacy_llama (`llama_prompt`) |
|---|---|---|
| `run_id, request_id, query_id, doc_rank, doc_id, retrieval_score` | 공통 | 공통 |
| `grader_type` | `official_t5` | `legacy_llama` (`grader_impl` = 기존 이름 `llama_prompt`) |
| `grader_score` (판정 점수) | T5 회귀 logit, 빈 문서는 공식 규칙 -1.0 | GRADE 출력에서 파싱한 0~1 |
| `t5_raw_logit` | 모델 (또는 캐시) logit. 규칙 점수면 null | null |
| `llama_parsed_score` | null | 0~1 파싱 점수 |
| `llama_yes_logprob` | null (`not_applicable`) | **null** (`not_extracted`): 기존 grader 는 logprobs 를 요청하지 않음. 0~1 점수로 대체하지 않음 |
| `grader_flag / grader_pass / grader_decision` | 2/1/0, flag 2 만 true, pass/partial/fail | 같음 |
| `decision_rule, threshold_upper/lower` | `score>=0.592 pass, >=-0.995 partial` + 질문 규칙 | 0.7 / 0.3 |
| `document_tokens` | T5 토크나이저로 문서 단독 토큰 수 (정규화 후, `</s>` 포함). 캐시 backend 면 null | null (Llama 토크나이저 로컬에 없음) |
| `grading_input_tokens` | T5 입력 (`질문 [SEP] 문서`) attention-mask 합. 판정 경로는 truncation 없음 → 실제 입력 전체 | 엔진 보고 prompt 토큰 |
| `grading_output_tokens` | null (`regression_scalar_no_generated_tokens`) | 엔진 보고 생성 토큰 |
| `grading_latency_ms` | 이번 run 의 문서 평가 wall | 같음 |
| `grading_compute_ms, timing_kind, is_cache_replay` | `live_model` / `cache_replay` (compute 는 기록 당시 값) / `rule_no_model` / `mock` | `live_llm` |
| `evaluated` | 평가 안 한 문서 (조기 중단) 는 false, **점수·시간·토큰 모두 null** | 같음 |
| `provisional_action_after, action_final_after` | 이 문서까지의 판정·확정 여부 | 같음 |

### 9.2 Experiment Manifest (`manifest.json`, `rag.evidence.manifest`)

MODE B run 마다: dataset 경로·sha256, question ID 목록, 고정 Top-5 입력 파일·sha256 (+ docs·retriever·기준 run),
prompt template 별 sha256 (사용한 것 + 전체, `version_sha256`), 공식 T5 가중치 fingerprint (파일 크기 + 작은 파일
sha256, `manifest.full_weight_hash=true` 면 safetensors 전체), generator (model, `llm.revision`, api, temperature,
top_p / `top_p_sent`, max_tokens, seed, generation prompt), grader threshold·판정 규칙, evidence 정책·budget,
correction (query_mode, 검색 조건, 캐시 파일 sha256), 평가 스크립트 sha256 (`metrics.py`, `generation.py`),
분석 스크립트 sha256, git commit/branch/dirty.

- `comparison` / `comparison_sha256`: **정책 간에 같아야 하는 조건** (고정 입력, dataset, 질문 집합, generator,
  web top_k, seed, `experiment_mode`). 정책·grader·budget 은 넣지 않는다. 같으면 같은 비교 조건.
- `experiment_mode=B_controlled` 이고 `not_comparable_with=A_official_reproduction`. MODE A 는 공식 데이터·저자
  지식 파일·completions API 를 쓰는 별도 설정이며 분석 스크립트가 섞지 않는다 (MODE A manifest 는 추가하지 않음,
  `run_meta.json` 참고).
- `llm.revision` 은 서버에서 직접 적어야 한다 (vLLM API 가 revision 을 주지 않음): `--set llm.revision=<snapshot hash>`.
  chat API 엔진은 top_p 를 보내지 않으므로 `top_p_sent=null`.

### 9.3 Correction 검색 캐시 공유 (`rag.retrieval.web_cache` v2 + `prefetch_web.py`)

기존 Tavily Record/Replay (`RecordingWebSearch` / `ReplayWebSearch`) 를 확장했다 (별도 시스템 없음).

- 기록 (`web_cache.v2`, v1 필드에 추가만): `query_id, query (실제 외부 검색어), provider, params (search_depth,
  include_raw_content, top_k), query_mode, searched_at_utc, cache_version, docs, raw_response (Tavily 원본 응답;
  `web_search.include_raw_content=true` 면 페이지 원문 포함), latency`.
- 재생 `replay_key=query_strict` (official_crag 기본값으로 변경): `(query_id, 실제 검색어, top_k)` + 검색 조건
  (provider, params, query_mode) 이 모두 같을 때만 hit. **같은 query_id 라도 검색어가 다르면 miss**
  (`miss_reason=query_mismatch`, 기록된 검색어 함께 남김). 결과 행 `correction.replay.compatibility` = `exact` | `legacy_assumed`.
- 공식 CRAG vs Legacy 검색 방식: 공식 원본은 GPT-3.5 keyword 추출 + serper (Google) + 페이지 수집, MODE B 는
  Llama 로 검색어 생성 + Tavily. MODE B 안에서도 `correction.query_mode` (legacy_rewrite / official_keywords) 가
  다르면 검색어가 다르므로 **캐시가 호환되지 않는다** (`search_condition_mismatch`). query_mode 마다 캐시를 따로 만든다.
- v1 기록 (기존 crag_system `web_cache.jsonl`, 조건 정보 없음) 은 `accept_legacy_records=true` 일 때만, 검색어 문자열과
  top_k 가 같고 query_mode=legacy_rewrite 일 때 쓴다.
- 비밀정보: API key 는 환경변수 (`TAVILY_API_KEY`) 로만 읽고, params·raw_response 의 key/token/secret/password/auth
  이름 필드는 `<redacted>` 로 지운다. `results/` 는 git 에 들어가지 않는다.
- `replay_on_miss=live`: miss 면 실제 검색하고 그 run 의 `web_cache.jsonl` 에 v2 로 기록 (다음 run 에 추가 가능).

```bash
# query_mode 별 공유 캐시 (vLLM 실행 중, 모든 질문 1회 검색 = 230 Tavily 호출)
uv run --extra crag --extra official python experiments/official_crag/prefetch_web.py \
    --overlay experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml \
    --set "inputs.frozen_path=$FROZEN" --name popqa230_legacy_rewrite
uv run --extra crag --extra official python experiments/official_crag/prefetch_web.py \
    --overlay experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml \
    --set "inputs.frozen_path=$FROZEN" --set correction.query_mode=official_keywords \
    --name popqa230_official_keywords
# C1/C2/C3 run 에서
--set web_search.replay_key=query_strict --set "web_search.replay_paths=[$WEB_CACHE]"
```

생성 검색어는 temperature 0 순차 실행 기준으로 재현된다. live 모드 (동시 실행) 에서 검색어가 달라지면 miss 로 드러난다
(조용히 다른 결과를 재사용하지 않음).

### 9.4 Correction 발생 질문 분석 (`analyze_correction.py`)

Correction 발생 범위 (논문 CRAG / 공식 `CRAG_Inference.py`): **Correct** = 내부 refinement 만 (외부 검색 없음, Correction
아님), **Incorrect** = 내부 문서 버리고 web search, **Ambiguous** = 내부 refinement + web search. 기본 Correction 집합
= 공식 T5 판정이 Ambiguous ∪ Incorrect. strict (Incorrect 만) 결과도 같이 낸다.

같은 고정 입력·질문·generator·평가 기준인지 `manifest.json` (`comparison`) 으로 확인하고 다르면 실패한다
(official/vanilla), 또는 해당 비교 run 을 제외하고 사유를 남긴다 (legacy/keep_all). 질문 단위 paired 비교.

```bash
# 서버: C1 행렬 (T5-OFF, VAN, LL-RULE, T5-KEEP 등) 이 끝난 뒤
R=results/official_crag/pre
uv run python experiments/official_crag/analyze_correction.py \
    --official $R/PRE-C1-T5-OFF/<run_id> --vanilla $R/PRE-C1-VAN/<run_id> \
    --legacy $R/PRE-C1-LL-RULE/<run_id> --keep-all $R/PRE-C1-T5-KEEP/<run_id> \
    --question-ids <230문항 id 파일> --name c2_correction_pre
# (T5 grader 의 legacy 정책과도 비교:  --extra t5_rule=$R/PRE-C1-T5-RULE/<run_id>)
```

출력 `results/official_crag/analysis/<name>/`:

| 파일 | 내용 |
|---|---|
| `summary.json` / `report.md` | 전체 질문 수, Correction 수·비율 (+ strict), 경로 C/A/I 수, web 무효 수, 집합 (correction / strict / all / 경로별) x 비교 (official·legacy·keep_all vs vanilla, vs official) 의 기준·비교 정확도, 개선 (Vanilla 오답→정답) / 악화 (정답→오답) / 둘 다 정답 / 둘 다 오답, McNemar p. `acc` 와 `official_match` 각각. C2 분류, 경로 교차표, 제외 run·사유 |
| `per_question.csv` | 질문별 T5 경로, correction 실행·유효·검색어, 정책별 정답 (acc / official_match)·경로·e2e, Vanilla 대비 변화, `c2_category` |
| `path_breakdown.csv` | T5 경로 x 정책: 정확도 + 비용 (e2e, grading / refinement / rewrite / web_search / generation 초, evaluator 계산, generation 토큰, LLM 호출 수, web 검색 비율, evidence 수) |
| `early_stop_summary.csv` | 중단 규칙 (`first_pass`, `max_docs:k`) 별 조기 중단·판정 변화 (전이)·evidence 변화 수, 절감 문서·grading latency (비율)·토큰, 품질 영향 **상한** |
| `early_stop_per_question.csv` | 질문 x 규칙 상세 |
| `verification_candidates.csv` | evidence 가 바뀌어 generation 재실행으로 확인해야 하는 (질문, 규칙) 과 확인 방법 |

C2 분류 (Correction 질문, acc): `correction_improved` (기여), `correction_no_change_correct` / `_wrong` (실행됐지만
정답 변화 없음), `correction_degraded` (Correction 후 오답). Accuracy 가 Vanilla 보다 높지 않아도 모든 표가 나온다.
Vanilla 대비 차이는 official 정책에서는 refinement + correction 결합 효과다. `keep_all@official_t5` 는 같은 질문에서
correction 을 하면서 내부 문서를 유지하므로 keep_all − vanilla 가 "web 추가" 효과에 가깝다.

Grading 중단 분석: full run 의 문서별 기록으로 판정·evidence 변화와 절감 비용은 **정확히** 계산된다 (공식 T5 점수는 문서별
독립). `first_pass` 는 공식 판정을 바꾸지 않는다 (flag 2 는 확정). 정답 변화는 **확정하지 않는다**: evidence 가 바뀐
질문 중 full run 정답 수 = 손실 상한, 오답 수 = 이득 상한 (evidence 가 같으면 temperature 0 에서 답이 같다는 가정).
official 정책은 refinement 를 모든 문서에 대해 한다고 가정한다. 실제 확인: legacy 정책은
`--set budget.stop_rule=max_docs:k` run 으로 가능, official 정책 + 부분 grading 은 runner 가 거부하므로 판정 강제
재실행 기능이 필요하다 (미구현).

**주의**: 230문항은 기존 공통 성공 문항 (선택된 집합) 이다. 이 집합의 정확도·Correction 비율을 전체 PopQA 의 일반적인
수치로 표현하지 않는다 (report 에 자동 표기). mock evaluator run 이면 report 맨 위에 MOCK 경고가 붙는다.
