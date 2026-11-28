# C2 Bottleneck-aware Gating — MODE A 위의 B / C / Cbn

브랜치 `exp/c2-gating-mode-a` (기준 `feature/official-crag-baseline` @ 743096e). MODE A 재현 코드·결과는 수정하지
않는다 (`experiments/official_crag/`, 기존 `rag.crag_official` 모듈, `results/official_crag/`). main·기존 브랜치에 merge 하지 않는다.

> **상태**: 구현과 로컬 mock 검증까지. 서버 smoke / 성능 결과는 아직 없다. MODE B 예비 실험 수치가 MODE A 에서
> 나온다고 가정하지 않는다. 성능 개선·신규성은 입증 전이다.

## 0. 기준 MODE A (정확도 상승 확인 설정)

| 항목 | 값 |
|---|---|
| config | `experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml` (변경 금지, run.py 가 llm 블록 일치 검사) |
| generator | `meta-llama/Llama-2-7b-hf`, fp16, TP=2, vLLM completions, temperature 0 / top_p 1 / max_tokens 100 |
| 데이터·검색 | PopQA long-tail 1,399문항, 공식 Contriever top-10 |
| evaluator | 공식 T5 (fp32), PopQA threshold 0.592 / -0.995 |
| prompt | `v3_2024_10` |
| scores | `results/official_crag/mode_a/score/20261126_072612` |
| Vanilla | `results/official_crag/mode_a/generate_rag/20261126_093945` — 705/1399 (50.393%) |
| CRAG | `results/official_crag/mode_a/generate_crag/20261127_023204` — 769/1399 (54.968%), +4.575%p, McNemar p=0.000789 (정규화 acc +2.001%p, p=0.1201) |
| 정답 파일 | 서버에서 `resolve_mode_a.py resolve` 가 evaluate run 의 config 에서 찾고 705/769 재계산으로 확정 |

## 1. 두 실험 모드 (결과 폴더·주장 범위가 다르다)

| | **Controlled Replay** (`gating.experiment_mode=controlled_replay`, 기본) | **Runtime End-to-End** (`runtime_e2e`) |
|---|---|---|
| generation evidence | 저자 사전 생성 지식 `ref/{correct,incorrect,ambiguous}` (MODE A 와 같은 줄, 고정) | 실행 중 계산한 지식 (internal refine / external refine / Knowledge1-2) |
| correction workload | 실제 실행 (internal·external refinement T5, rewrite, web 재생) | 실제 실행 |
| 비교 대상 | B/C/Cbn 의 **serving 비용** | B/C/Cbn 의 action·evidence·prompt·**정확도**·성능 |
| 정확도 주장 | MODE A 와의 일치는 evidence 고정의 결과. **End-to-End CRAG 정확도 보존의 증거가 아님** | B (같은 runtime 경로) 기준. 저자 지식과 evidence 가 달라 MODE A 정확도와는 참고 비교만 |
| 결과 | `results/c2_gating/controlled_replay/<purpose>/<run>/` | `results/c2_gating/runtime_e2e/<purpose>/<run>/` |

두 모드 모두 **그 판정에 필요한 correction 작업이 끝난 뒤에만 generation 을 시작**한다 (Correct: internal refine,
Ambiguous: internal + external, Incorrect: external). 요청마다 `generation_precondition` (필요 작업, 완료 시각,
generation 시작 시각, 위반 사유) 을 기록하고 smoke 가 전 요청 위반 0 을 확인한다. B/C/Cbn 의 차이는 correction
경로의 **시작·submit 시점과 취소** (실행 정책) 뿐이다: evidence 구성, prompt, generation 은 같은 코드.

Runtime E2E 준비 상태: 구현·mock 테스트 완료 (`test_runtime_e2e_mock`). 실행 전 필요: 대상 질문 전체의 web 기록,
Controlled Replay smoke 통과.

## 2. 조건

| | correction 시작 | Correct 판정 시 |
|---|---|---|
| **B** | 판정 후 필요한 경로만 (internal_refine ∥ rewrite→web→external_refine) | 해당 없음 |
| **C** | grading ∥ rewrite→web→external_refine (web 직후 T5 submit) | queue 대기 job 제거, 실행 중 job 은 다음 batch 에서 정지, rewrite/web 폐기 |
| **Cbn** | grading ∥ rewrite→web. external_refine 은 판정 (Ambiguous/Incorrect) 확정 후에만 submit | rewrite/web 취소·폐기 (T5 submit 없음) |

공통: T5Service (단일 worker FIFO queue, batch_size, memo), generator 서버, 질문·web 기록, internal_refine (판정 후).

## 3. 공식 CRAG 와 다른 실험 환경 (config `deviations`, run_meta·summary 에 기록)

| 항목 | 공식 | 이 실험 | 영향 |
|---|---|---|---|
| keyword 추출 | gpt-3.5-turbo-16k | MODE A generator (LLaMA2-7B-hf, 같은 vLLM), 공식 PopQA few-shot, 첫 줄, max_tokens 32 | rewrite 비용·검색어 |
| web 검색 | serper.dev (Google) | Tavily (basic, top_k 5, include_raw_content), 질문당 1회 기록 → query_id 재생, 기록 지연 재현 | web 지연, 결과 페이지 |
| 페이지 텍스트 | HTML `<p>` 문단 | Tavily raw_content 의 줄 (10자 초과, "제목: 문단"), wikipedia.org 결과만 | external refinement 문단 수 (T5 작업량) |
| internal refinement | (MODE A 는 ref/correct 사용) | 공식 selection refinement 실제 실행 | Controlled Replay 는 비용만 |

Controlled Replay 에서는 위 차이가 generation 입력에 들어가지 않는다 (비용에만 영향). Runtime E2E 에서는 evidence 에 들어간다.

## 4. GPU 배치 (LLaMA2 fp16 TP=2 + T5 fp32)

- vLLM TP=2 가 GPU 0,1 을 모두 쓰므로 T5 는 vLLM rank 와 같은 GPU (기본 물리 GPU1) 를 공유한다. generator·precision·
  T5 dtype 은 바꾸지 않는다.
- 프로필 `gpu_memory_utilization=0.85` 는 GPU 당 약 15% 만 남겨 T5 (가중치 2.75GB + activation + CUDA context, 추정
  3~4 GiB) 가 들어가지 않을 가능성이 높다 (미검증). `probe_serving.py t5-memory` 가 vLLM 을 띄우기 전에 T5 최대
  메모리를 재고 `gpu_plan.json` 에 (a) 현재 설정으로 들어가는지, (b) 들어가지 않으면 권장 `gpu_memory_utilization`,
  (c) 그때의 KV cache 추정 (max_num_seqs 8, CRAG prompt p95 길이 기준) 을 적는다.
- 바꾸는 것은 `launch_vllm.py --gpu-memory-utilization` (launch 기록 `overrides` 에 남고 run_meta `vllm_launch` 로 연결).
  프로필 파일은 수정하지 않는다. max_num_seqs (8)·max_model_len·TP 는 그대로.
- **바꾸면 정확도 재검증 범위**: KV cache 크기만 바뀌고 가중치·dtype·kernel 은 같아 순차 greedy 출력은 같을 것으로
  예상하지만 검증 전이다. 필요: MODE A `generate --method crag` 1,399 순차 재실행 → `20261127_023204` 와 문항별 prediction
  비교·769 재현, `generate --method rag` → `20261126_093945`·705 재현 (Vanilla 대비 +4.575%p 를 새 serving 설정에서
  주장하려면). T5 score run 은 재실행 불필요 (smoke 의 판정 대조로 확인). `smoke.sh reverify`.
- T5 와 vLLM rank 가 같은 GPU 에서 연산을 경합한다. B/C/Cbn 공통 조건이며 `probe_serving.py concurrent` 로 크기를 잰다.

## 5. Memo / batching

- memo 는 같은 요청 안에서 입력 텐서 (padding 포함 input_ids) 가 같은 쌍만. 적용 조건: `check_memo.py` 의
  `memo_decision.memo_allowed=true` (같은 텐서 문서의 grading/refinement 점수 비트 일치, memo 적용 refinement 결과
  전 문항 동일) **그리고** memo_verify smoke run 의 mismatch 0. 적용하면 B/C/Cbn 모두 (`t5_service.memo` 는 run 단위 설정).
- batch_size 1 (공식). 요청 간 batching 은 `T5Service(policy=BatchPolicy)` 인터페이스 유지 (팀원 협업).

## 6. Workload Sensitivity (Correct 비율)

새 데이터셋 일반화가 아니라 같은 1,399문항 안에서 Correct 비율만 바꾼 workload 분석이다. `make_workload_sets.py` 가 MODE A
score run 의 실제 분포로 Low / Mid / High 를 각 200문항 (가능하면 서로 겹치지 않게, 비-Correct 는 전체의 Ambiguous:Incorrect
비율 유지) 으로 만든다. 비율은 분포에서 자동 결정 (mid = 전체 비율의 0.05 반올림, low/high = mid ∓ d) 후 manifest 에
기록하고, 결과를 보고 확정한다. 집합별 입력 통계 (문서 길이, 빈 문서) 를 함께 기록한다. Mid 를 Go/No-go 집합으로 쓴다.

## 7. Go/No-go (제안 — smoke 결과를 본 뒤, 성능 실험 전에 확정)

Controlled Replay, Mid 200, B/C/Cbn x 동시 1/4/16, 반복 2, 조건 순서 셔플.

1. 전제: smoke 모든 항목 PASS (오류 0, 동일성, Correct external T5 0 (B/Cbn), Cbn 판정 전 submit 0, precondition 위반 0, 누수 0).
2. 병목 존재 (MODE A 에서 다시 확인, MODE B 수치 가정 안 함):
   (a) 직렬 대기: B c=1 에서 Ambiguous/Incorrect 요청의 correction 경로 (rewrite+web+external) 가 E2E 의 상당 부분,
   (b) 고부하 T5: 동시 요청이 늘 때 T5 utilization 상승, c=16 grading queue wait 가 요청당 grading busy 보다 큼.
3. Cbn: c=1 에서 C 의 E2E 이득 (B 대비, repetition 범위 밖) 의 상당 부분 유지, c=16 에서 B 대비 throughput 손실이 C 보다
   작음, 낭비 external T5 0.

2 가 성립하지 않으면 No-go 로 보고하고 원인을 분석한다 (부정적 결과도 그대로 보고). 수치 기준 (예: "상당 부분") 은 smoke·probe
결과로 분산을 본 뒤 정한다.

## 8. 파일

```
src/rag/crag_official/t5_service.py   T5Service (단일 worker queue, 취소, memo, BatchPolicy), OfficialRules, JobScorer  [신규]
experiments/c2_gating/
  c2_pipeline.py / c2_data.py / c2_web.py / c2_metrics.py / c2_components.py
  run.py                closed-loop runner (조건·요청 순서 셔플, warmup, 누수 점검, vLLM launch 기록)
  resolve_mode_a.py     MODE A 기준 확인 (정답 파일 탐색, 705/769 재계산) + prediction 재검증 비교
  probe_serving.py      t5-memory (GPU 계획) / concurrent (실제 중첩)
  check_memo.py         refinement 중복 채점·memo 성립
  prefetch_web.py       web 기록 (여러 index 파일, 이미 기록된 질문 건너뜀)
  make_workload_sets.py Low/Mid/High Correct 비율 집합
  smoke_check.py        정합성 판정
  analyze.py            조건별 표 + B 대비
  server/smoke.sh       서버 단계별 명령 (prep / t5mem / vllm / probe / web / smoke / reverify / collect)
```

## 9. 측정 항목 위치

| 항목 | 조건 `summary.json` |
|---|---|
| throughput, 평균/P95, action 별 | `throughput_rps`, `e2e`, `e2e_by_action` |
| 요청당 T5 busy (grading / internal / 유효 ext / 낭비 ext), queue wait | `t5.busy_per_request_mean_s`, `t5.queue_wait_s` |
| rewrite / web 낭비 | `waste` |
| GPU utilization | `gpu` (nvidia-smi 200ms) |
| 실행 중첩 | `overlap.stage_overlap_frac`, `overlap.exec_overlap_frac` |
| Correct external refine / Cbn 판정 전 submit | `correct_external_refine`, `external_submit_before_decision` |
| generation 선행 조건 | `generation_precondition` |
| 누수 | `leak` |
| 동일성·MODE A 대조 | run `summary.json` `identity`, `reference` |

## 10. 알려진 한계

- 버린 rewrite 는 HTTP 호출을 끊지 않고 끝날 때까지 기다린 뒤 폐기 (vLLM 연산은 실제로 쓰임 = 낭비로 기록).
- 실행 중 T5 job 취소는 batch 경계에서만.
- 동시 요청에서는 evidence·prompt 가 같아도 vLLM batch 구성 차이로 greedy 출력이 달라질 수 있다 (`identity.prediction`).
- vLLM `max_num_seqs=8` (MODE A 설정 유지) 이므로 동시 16/32 에서는 vLLM 내부 대기가 생긴다 (병목 해석 시 구분).
