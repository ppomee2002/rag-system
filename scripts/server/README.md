# 서버 점검 절차 (사용자가 직접 실행)

Claude 는 서버에 접속하지 않는다. 아래를 순서대로 실행하고, 결과 파일(`~/crag_logs/*.txt|json`,
`results/crag_system/SYS-P/<timestamp>/`)을 확인 후 공유한다. 키·토큰 값은 공유하지 않는다.

| 스크립트 | 실행 환경 | GPU 사용 | 하는 일 |
|---|---|---|---|
| `check_env.sh` | 시스템 셸 | 없음 | OS·GPU·CUDA·다른 사용자 점유·디스크·네트워크·포트 조회 (읽기 전용) |
| `check_vllm_install.py` | vLLM venv | CUDA 초기화만 (모델 로드 없음) | torch/CUDA/장치 capability, vLLM 기능 플래그 존재 여부 |
| `launch_vllm.sh` | vLLM venv, tmux | 모델 로드 | vLLM 서버 실행 + launch 메타데이터 기록 |
| `probe_engine.py` | 레포 uv 환경 | 짧은 요청 | SYS-P 점검 (smoke, 결정성, grade 출력 형식, auth, priority, metrics, 선택: 순서·처리량) |

## 1단계: 환경 확인 (GPU 사용 없음)

```bash
cd ~/RAG && git fetch && git checkout exp/crag-system      # 사용자가 push 한 뒤
bash scripts/server/check_env.sh                            # -> ~/crag_logs/env_<ts>.txt
```

여기서 결과를 공유해 주시면 GPU 모델·메모리에 맞춰 모델 정밀도 / TP / 양자화 설정을 제안한다.
특히 확인할 것: GPU 모델·VRAM·compute capability, 드라이버 버전, 다른 사용자 프로세스, `$HOME` 여유 공간
(모델 BF16 ≈ 16GB), Hugging Face 접근 (Llama 3.1 은 gated 모델: 라이선스 동의 + 토큰 필요).

## 2단계: vLLM 설치 (레포 환경과 분리된 venv)

```bash
uv venv ~/venvs/vllm --python 3.12          # uv 가 없으면: python3 -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
uv pip install vllm                          # 버전은 1단계 결과 (드라이버/CUDA) 를 보고 고정 예정
export HF_HOME=~/hf_cache                    # 용량 충분한 위치
export HF_TOKEN=...                          # 화면/로그에 남기지 않기

export CUDA_VISIBLE_DEVICES=<비어 있는 GPU 번호>
python ~/RAG/scripts/server/check_vllm_install.py --out ~/crag_logs/vllm_install_$(date +%Y%m%d_%H%M%S).json
```

## 3단계: vLLM 서버 실행 (tmux)

```bash
tmux new -s vllm
source ~/venvs/vllm/bin/activate
export CUDA_VISIBLE_DEVICES=<GPU>  HF_HOME=~/hf_cache  HF_TOKEN=...
export VLLM_API_KEY=$(openssl rand -hex 16); echo "$VLLM_API_KEY" > ~/.vllm_key; chmod 600 ~/.vllm_key
MODEL=meta-llama/Llama-3.1-8B-Instruct TP=1 GPU_MEM_UTIL=0.85 SCHED_POLICY=fcfs \
  bash ~/RAG/scripts/server/launch_vllm.sh
# 분리: Ctrl-b d
```

- GPU 1장에 BF16 이 안 들어가면 (`check_vllm_install.py` 의 `weights_fit_hint`) 실행하지 말고 결과를 공유.
  TP / 양자화는 정확도·성능 비교 조건을 바꾸므로 함께 정한다.
- priority 점검까지 하려면 `SCHED_POLICY=priority` 로 한 번 더 띄운다 (`PORT=8001` 로 따로 띄워도 됨).

## 4단계: 점검 (SYS-P)

```bash
cd ~/RAG && uv sync --extra crag
export VLLM_API_KEY=$(cat ~/.vllm_key)
uv run python scripts/server/probe_engine.py --config experiments/crag_system/configs/engine_vllm.yaml
# priority 서버에서만 (GPU 부하 수십 초):
uv run python scripts/server/probe_engine.py --config experiments/crag_system/configs/engine_vllm.yaml \
  --priority-load 8 --sweep 1,2,4,8
```

## 정리

```bash
tmux attach -t vllm   # Ctrl-C 로 서버 종료
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # 내 프로세스가 남지 않았는지
```

## 공유 서버 수칙

- `CUDA_VISIBLE_DEVICES` 와 `GPU_MEM_UTIL` 을 항상 명시. 실험 전후 `nvidia-smi` 스냅샷은 `run_meta.json` 에 자동 기록됨.
- 서버는 `127.0.0.1` 에만 바인딩, API key 는 환경변수로만 (명령줄 인자는 `ps` 로 다른 사용자에게 보임).
- 다른 사용자의 프로세스가 같은 GPU 에 있으면 측정 run 은 시작하지 않는다 (성능 비교 무효).
