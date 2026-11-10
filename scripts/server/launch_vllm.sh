#!/usr/bin/env bash
# (이전 방식) 환경변수 기반 launch 는 launch_vllm.py 로 대체되었다.
# 엔진 프로필 YAML 이 모델 / TP / GPU 수 / 옵션의 단일 기준이다.
#
#   python scripts/server/launch_vllm.py --profile experiments/crag_system/configs/engines/<profile>.yaml [--smoke]
set -euo pipefail
exec python3 "$(dirname "$0")/launch_vllm.py" "$@"
