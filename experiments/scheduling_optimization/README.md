# Scheduling Optimization

기존 SRPT-est + Aging 실험에 이어 **대기 임계값 T의 성능 절충**을 분석합니다. 최종 후속 실험일은 2026-11-07입니다.

## 실험 목적

SRPT-est는 예상 잔여 작업시간이 짧은 요청을 먼저 처리합니다. 기존 Aging 실험에서는 Correction 요청의 지연이 줄었지만 전체 평균 E2E가 증가했습니다. 이번에는 대기가 T를 넘은 요청만 우선 처리하여 **Aging보다 전체 평균 E2E를 낮추면서 Correction p95를 유지하거나 줄일 수 있는지** 확인했습니다.

## 비교 정책

| 정책 | 다음 LLM 호출 선택 규칙 |
|---|---|
| SRPT-est | 예상 잔여 작업시간이 가장 짧은 요청 |
| SRPT-est + Aging | `예상 잔여 작업시간 − α × 누적 큐 대기시간`이 가장 작은 요청 |
| SRPT-est + 임계값 | 누적 큐 대기가 T를 넘은 요청 중 가장 오래 기다린 요청. 없으면 SRPT-est |

- 임계값 정책은 Aging과 결합하지 않습니다. T 초과 후에도 누적 대기를 초기화하지 않습니다.
- 누적 대기는 큐에서 기다린 시간이며 실행·검색시간을 제외합니다. 동률이면 예상 잔여 작업시간, 큐 진입 순서로 결정합니다.
- 실행 중인 호출은 중단하지 않고 호출 사이에서 순서를 정합니다. T는 응답 완료시간 보장이 아닙니다.
- Correction 분기 시 잔여 작업량 갱신은 기존 구현에도 있던 기능입니다.

## 실험 조건

| 항목 | 최종 설정 |
|---|---|
| 입력 | 기존 Full Trace 50문항, 원본 Correction 사례 5개 |
| 부하 ρ | 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 |
| 도착·추출 시드 | 30, 31, 32, 33, 34 |
| 요청 수 | 조건별 3,000개 반복 추출, 초기 10% 제외 |
| T | 30, 60, 120, 240, 480초 |
| Aging α | 주 비교 기준 0.5, 참고 기준 0.1·1.0 |
| 비교 통제 | 정책 간 같은 Trace 표본·도착 시각·예측 모델 |

모든 T를 공개하는 탐색적 분석입니다. 종합 점수로 후보를 탈락시키거나 최적 T를 선택하지 않았습니다. 기본 설정의 시드 0~4 대신 `tradeoff_study.seeds`를 사용합니다.

E2E는 도착부터 완료까지의 모델링된 검색·LLM 실행·큐 대기를 포함합니다. Correction 지표는 보정 단계만의 시간이 아니라 보정 요청의 전체 E2E입니다. `Non-correction`은 보정 없이 완료된 요청입니다.

## 최종 결과

대표 부하 **ρ=0.8**, 시드별 통계의 평균입니다.

| 정책 | 전체 평균 E2E | Non-correction 평균 E2E | Correction p95 |
|---|---:|---:|---:|
| SRPT-est | 75.18초 | 69.32초 | 483.94초 |
| SRPT-est + Aging (α=0.5) | 80.16초 | 79.01초 | 240.97초 |
| T=30초 | 84.45초 | 83.66초 | 236.23초 |
| T=60초 | 83.23초 | 82.00초 | 236.23초 |
| T=120초 | 81.44초 | 79.28초 | 240.38초 |
| T=240초 | 79.12초 | 75.62초 | 303.85초 |
| T=480초 | 77.06초 | 72.70초 | 461.49초 |

- 작은 T는 장기 대기를 완화하지만 짧은 요청의 처리를 늦춰 전체 평균이 증가했습니다.
- 큰 T는 SRPT의 이점을 유지하지만 Correction 장기 지연 완화 효과가 약해졌습니다.
- **6개 부하 × 5개 T에서 Aging α=0.5보다 전체 평균 E2E가 낮고 Correction p95도 같거나 낮은 조합은 시드 평균 기준으로 없었습니다.** 목표한 동시 개선을 확인하지 못했습니다.
- 대기시간이 임계값을 넘으면 잔여 작업량보다 대기를 우선하는 규칙이 SRPT의 짧은 작업 우선 효과를 약화시키는 것으로 해석됩니다. 통계적 유의성 주장은 아닙니다.

## 재현 방법

프로젝트 최상위 폴더에서 실행합니다. 기존 입력 Trace와 Python 의존성이 필요하며 Ollama·외부 API는 호출하지 않습니다.

```powershell
.venv\Scripts\python.exe -B experiments/scheduling_optimization/verify.py
.venv\Scripts\python.exe -B experiments/scheduling_optimization/run.py --tradeoff-study --records "results/grading_optimization/grading 최종결과/results.jsonl" --config experiments/scheduling_optimization/config.local.json
.venv\Scripts\python.exe -B experiments/scheduling_optimization/plot.py results/scheduling_optimization/20261107_130122_tradeoff
```

새 실행 결과는 별도 날짜 폴더에 저장됩니다. 그래프 명령의 마지막 경로는 새 실행 폴더로 바꿉니다. 조건 변경은 `config.local.json`의 공통 부하·요청 수 설정과 `tradeoff_study`에서 합니다.

## 결과 위치와 파일

최종 결과: `results/scheduling_optimization/20261107_130122_tradeoff/`

| 파일 | 내용 |
|---|---|
| `report.md`, `analysis_ko.md` | 전체 부하 결과와 해석 |
| `evaluation_summary.csv` | 시드별 통계를 평균한 정책 비교 |
| `sim_results.csv` | 개별 부하·시드·정책 결과 |
| `paired_comparisons.csv` | 기준 정책 대비 변화율과 동시 개선 여부 |
| `config.json`, `metadata.json`, `source_snapshot/` | 실행 설정·출처·코드 |
| `workloads/`, `verification.json` | 공유 요청 표본·도착 시각과 검증 |
| `figures/` | E2E·Correction·대기시간·절충 그래프 |

평균·최대 누적 큐 대기, 임계값 초과 비율, 처리량도 CSV에 기록합니다. `mean_gpu_wait` 등은 기존 열 이름이며 실제 의미는 모델 서버 큐 대기입니다. p95·최대 대기 요약은 시드별 통계의 평균이고 전체 시드 최대 대기는 `worst_seed_max_wait`입니다. 서로 다른 T의 자체 초과 비율은 같은 기준으로 해석하지 않습니다.

## 해석 범위와 이전 실험

- 3,000개 요청은 원본 50문항을 재사용한 것입니다. 독립 Correction 사례는 5개뿐입니다.
- 답변과 작업시간을 고정한 Trace 시뮬레이션이며 새 정확도·GPU 사용률·실제 동시 서비스 성능을 측정하지 않습니다. ρ는 GPU 사용률이 아닙니다.
- 질문 단위 예측은 OOF이지만 호출 비용 모델은 전체 원본 풀로 학습합니다. 독립 질문에 대한 일반화 검증은 아닙니다.
- 이전 Aging 결과(`20261101_101541`)는 배경 자료입니다. 시드가 다른 이전 수치를 최종 표와 섞지 않습니다.
- 중간 임계값 선택·4종 구성 실험(`20261107_114509_threshold`)과 호출 1회 우선권 실험(`20261107_122027_rescue`)은 보존하지만 최종 절충 분석에 포함하지 않습니다. 해당 실행의 설정·코드 스냅샷을 참고합니다.
- CRAG 4종 통합 후속 부하 평가는 이번 최종 실험 범위에 포함하지 않습니다.
- `simulator.py`는 팀원 원본을 실험 폴더 내에서 확장한 코드입니다. 공통 `src/rag/`, `configs/base.yaml`, 팀원 원본은 변경하지 않습니다.
