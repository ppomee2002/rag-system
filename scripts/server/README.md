# 서버 실행 절차 (사용자가 직접 실행)

Claude 는 서버에 접속하지 않는다. 아래 명령은 사용자가 순서대로 실행한다.
기존 vLLM 설치 환경 (venv) 은 바꾸지 않는다 — launch / verify 스크립트는 import 만 한다.

| 구분 | 결과 위치 | 논문 사용 |
|---|---|---|
| smoke (`SMOKE-*`) | `results/crag_system/smoke/` | 사용 안 함 |
| 예비 측정 (`PRE-*`) | `results/crag_system/pre/` | 설정 결정 근거 (부록 수준) |
| 본실험 (`SYS-1~4`) | `results/crag_system/paper/` | 별도 승인 후 `--paper-approved` |

## 공통 준비 (터미널마다)

```bash
cd ~/RAG
AWQ=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
BF16=experiments/crag_system/configs/engines/vllm_bf16_tp2.yaml
export VLLM_API_KEY=$(cat ~/.vllm_key)
```

vLLM 서버 창에서는 추가로 (venv 경로는 설치한 위치로):

```bash
source ~/venvs/vllm/bin/activate
source ~/.hf_env && export HF_HOME=~/hf_cache
```

## 0. 코드 반영과 사전 확인 (GPU 사용 없음)

```bash
cd ~/RAG && git fetch && git checkout exp/crag-system && git pull && git log --oneline -1
uv sync --extra crag && uv run pytest -q
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
test -f ~/.vllm_key || (umask 077 && openssl rand -hex 16 > ~/.vllm_key)
```

모델 다운로드 (vLLM venv, 약 21GB):

```bash
hf download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 || huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
hf download meta-llama/Llama-3.1-8B-Instruct || huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

## 1. AWQ INT4 (GPU 1장) smoke

창 1 (tmux `vllm`, vLLM venv):

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/server/launch_vllm.py --profile $AWQ --smoke --dry-run   # 검사만
python scripts/server/launch_vllm.py --profile $AWQ --smoke             # --enforce-eager 로 실행
```

`Application startup complete` 가 보이면 창 2 (레포 uv 환경):

```bash
M=$(ls -t ~/crag_logs/vllm_awq_int4_1gpu_fcfs_*_launch.json | head -1); echo $M
uv run python scripts/server/verify_launch.py --launch-meta $M
uv run python scripts/server/probe_engine.py --config $AWQ --set llm.launch_meta=$M
```

확인: verify 의 `checks` 가 모두 true, `compare` 의 실제 값 (모델 메모리, KV cache 토큰, 최대 동시 처리),
probe 의 smoke / determinism / grade_format / auth / launch_meta 오류 없음.
종료: 창 1 에서 Ctrl-C → `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 로 해제 확인.

## 2. BF16 TP=2 (GPU 2장) smoke

창 1:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/server/launch_vllm.py --profile $BF16 --smoke --dry-run
python scripts/server/launch_vllm.py --profile $BF16 --smoke
```

창 2:

```bash
M=$(ls -t ~/crag_logs/vllm_bf16_tp2_fcfs_*_launch.json | head -1); echo $M
uv run python scripts/server/verify_launch.py --launch-meta $M
uv run python scripts/server/probe_engine.py --config $BF16 --set llm.launch_meta=$M
```

TP=2 가 시작 단계에서 멈추면: Ctrl-C 후 `export NCCL_DEBUG=INFO` 로 다시 실행해 로그를 공유.
(`NCCL_P2P_DISABLE=1` 재시도는 결과를 보고 결정. 설정한 NCCL_* 는 launch 기록에 남는다.)

## 3. 두 설정 비교 (1·2 통과 후)

### 3-1. 속도 (CRAG 패턴 부하, `--smoke` 없이 띄운 서버)

각 프로필마다 창 1 에서 `--smoke` 없이 실행한 뒤 (AWQ 는 `CUDA_VISIBLE_DEVICES=0`, BF16 은 `0,1`):

```bash
python scripts/server/launch_vllm.py --profile $AWQ          # 또는 $BF16
```

창 2:

```bash
M=$(ls -t ~/crag_logs/*_fcfs_*_launch.json | head -1); echo $M
uv run python scripts/server/verify_launch.py --launch-meta $M
uv run python scripts/server/probe_engine.py --config $AWQ --set llm.launch_meta=$M \
  --sweep 1,2,4,8 --sweep-requests 2          # BF16 서버면 --config $BF16
```

### 3-2. 품질 (같은 질문·prompt, 데이터 파일과 TAVILY_API_KEY 필요)

AWQ 서버 실행 중 (A: 실제 Tavily 검색 기록):

```bash
uv run python experiments/crag_system/run.py --overlay $AWQ --set llm.launch_meta=$M \
  --exp-id PRE-Q --set run.purpose=pre --set run.record_texts=true --limit 23
A=$(ls -td results/crag_system/pre/PRE-Q/*/ | head -1); echo $A
```

BF16 서버로 바꾼 뒤 (B: A 의 검색 기록 재생, rewrite 가 달라 기록이 없으면 실제 검색):

```bash
uv run python experiments/crag_system/run.py --overlay $BF16 --set llm.launch_meta=$M \
  --exp-id PRE-Q --set run.purpose=pre --set run.record_texts=true --limit 23 \
  --set crag.web_search.provider=replay --set "crag.web_search.replay_from_runs=['$A']" \
  --set crag.web_search.replay_fallback=true --set crag.web_search.latency_mode=recorded
B=$(ls -td results/crag_system/pre/PRE-Q/*/ | head -1); echo $B
uv run python experiments/crag_system/compare_quality.py $A $B --out ~/crag_logs/pre_q_awq_vs_bf16.json
```

`--limit 23` = warmup 3 + 측정 20문항. 소규모 비교이며 비열등성 판단에 쓰지 않는다.

## 4. (엔진 확정 후) priority 처리 순서 검증

창 1: `--sched-policy priority --max-num-seqs 1` 로 실행. 창 2:

```bash
M=$(ls -t ~/crag_logs/*_priority_*_launch.json | head -1)
uv run python scripts/server/probe_engine.py --config <확정 프로필> --set llm.launch_meta=$M \
  --set llm.server_scheduling_policy=priority --priority-order 6 --priority-blockers 1
```

대조군: 같은 명령을 `--sched-policy fcfs --max-num-seqs 1` 서버와 `server_scheduling_policy=fcfs` 로.
`priority_order.verdict` 가 각각 `priority_order` / `fcfs_order` 이면 정상.

## 5. (엔진 확정 후) K calibration — PRE-K

```bash
uv run python experiments/crag_system/calibrate_k.py --overlay <확정 프로필> \
  --set llm.launch_meta=$M --set crag.web_search.provider=replay \
  --set "crag.web_search.replay_from_runs=['$A']" --ks 1,2,4,6,8,12 --n-requests 40
```

## 공유할 파일 (키·토큰 없음)

`~/crag_logs/*_launch.json`, `*_verify.json`, `results/crag_system/smoke/SMOKE-P/*/report.json`,
`~/crag_logs/pre_q_awq_vs_bf16.json`, `nvidia-smi topo -m` 출력.

## 공유 서버 수칙

- `CUDA_VISIBLE_DEVICES` 를 항상 지정. launch_vllm.py 는 선택한 GPU 에 다른 프로세스가 있으면 실행하지 않는다.
- API key 는 환경변수로만. vLLM 은 `/v1` 경로만 key 로 보호하고 `/metrics`, `/version` 은 인증이 없다.
  `--dev-mode` (`/reset_prefix_cache` 사용) 는 인증 없는 endpoint 를 더 열므로 필요할 때만.
