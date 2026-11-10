"""엔진 프로필 YAML 로 vLLM OpenAI 호환 서버를 띄운다 (vLLM 전용 venv 에서 실행).

    source ~/venvs/vllm/bin/activate
    export CUDA_VISIBLE_DEVICES=0 VLLM_API_KEY=$(cat ~/.vllm_key)
    P=experiments/crag_system/configs/engines/vllm_awq_int4_1gpu.yaml
    python scripts/server/launch_vllm.py --profile $P --smoke
    python scripts/server/launch_vllm.py --profile $P --dry-run   # 검사와 명령만 출력

실행 전 검사 (하나라도 실패하면 띄우지 않는다):
  1. CUDA_VISIBLE_DEVICES 개수 == server.n_gpus == server.tensor_parallel_size
  2. VLLM_API_KEY 환경변수 (명령줄 인자로 넘기지 않음 — 공유 서버에서 ps 로 보임)
  3. 설치된 vLLM 버전 == server.vllm_version (--allow-version-mismatch 로 무시)
  4. 넘길 옵션이 설치된 vLLM 의 EngineArgs / FrontendArgs 에 실제로 있는지
  5. 선택한 GPU 에 다른 프로세스가 없는지 (--allow-busy-gpu 로 무시)
  6. 포트가 비어 있는지

실행 기록: ~/crag_logs/<profile>_<policy>_<ts>.log 와 *_launch.json
(argv, 프로필 sha256, 버전, GPU 상태). *_launch.json 은 runner / probe 의 llm.launch_meta 로
넘겨 client 설정과 대조한다 (rag.scheduling.preflight).

vLLM 0.33.0 소스에서 확인한 사항 (2026-11-10):
  - priority 는 정수, 낮을수록 먼저. FCFS 서버가 0 이 아닌 priority 를 거부하는 코드는 찾지 못함
    → 수락 여부로는 정책을 판별할 수 없으므로 처리 순서로 검증한다 (probe_engine.py priority_order).
  - VLLM_API_KEY 는 /v1 등만 보호. /metrics, /version, (dev mode) /reset_prefix_cache 는 인증 없음.
  - /reset_prefix_cache 는 VLLM_SERVER_DEV_MODE=1 일 때만 등록되고 {"success": bool} 을 돌려준다.
  - generation_config 기본값 auto 는 체크포인트의 generation_config.json 을
    기본 sampling 값으로 쓴다
    → 프로필에서 vllm 으로 고정 (AWQ / BF16 체크포인트 간 기본값 차이 제거).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# 프로필 server 키 -> (CLI flag, EngineArgs / FrontendArgs 필드)
FLAG_MAP = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_num_seqs": "--max-num-seqs",
    "generation_config": "--generation-config",
    "seed": "--seed",
}
FRONTEND_FIELDS = ("host", "port")
SCHED_POLICIES = ("fcfs", "priority")


class LaunchError(RuntimeError):
    pass


def load_profile(path: str | Path) -> dict:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("engine_profile", "server", "llm"):
        if key not in data:
            raise LaunchError(f"프로필에 {key} 가 없음: {path}")
    return data


def effective_server(profile: dict, overrides: dict) -> dict:
    server = dict(profile["server"])
    for k, v in overrides.items():
        if v is not None:
            server[k] = v
    return server


def build_argv(server: dict, sched_policy: str, smoke: bool) -> list[str]:
    if sched_policy not in SCHED_POLICIES:
        raise LaunchError(f"--sched-policy: {SCHED_POLICIES}")
    argv = ["vllm", "serve", server["model"], "--host", "127.0.0.1", "--port", str(server["port"])]
    for key, flag in FLAG_MAP.items():
        value = server.get(key)
        if value is not None:
            argv += [flag, str(value)]
    argv += ["--scheduling-policy", sched_policy]
    if server.get("enable_prefix_caching") is True:
        argv.append("--enable-prefix-caching")
    elif server.get("enable_prefix_caching") is False:
        argv.append("--no-enable-prefix-caching")
    if smoke:
        argv.append(
            "--enforce-eager"
        )  # 첫 실행: CUDA graph 캡처 메모리 제외 (성능 측정에는 사용 금지)
    return argv


def used_fields(argv: list[str]) -> set[str]:
    """argv 의 --flag 들을 EngineArgs / FrontendArgs 필드 이름으로."""
    out = set()
    for tok in argv[3:]:
        if tok.startswith("--"):
            name = tok[2:]
            if name.startswith("no-"):
                name = name[3:]
            out.add(name.replace("-", "_"))
    return out


def check_gpu_assignment(server: dict, cuda_visible: str | None) -> list[str]:
    errors = []
    if not cuda_visible:
        return ["CUDA_VISIBLE_DEVICES 미설정: 사용할 GPU 를 명시할 것 (예: 0 또는 0,1)"]
    devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
    n, tp = server.get("n_gpus"), server.get("tensor_parallel_size", 1)
    if len(set(devices)) != len(devices):
        errors.append(f"CUDA_VISIBLE_DEVICES 에 중복: {cuda_visible}")
    if n != tp:
        errors.append(f"프로필 n_gpus ({n}) != tensor_parallel_size ({tp})")
    if len(devices) != tp:
        errors.append(
            f"CUDA_VISIBLE_DEVICES={cuda_visible} ({len(devices)}개) != tensor_parallel_size ({tp})"
        )
    return errors


def check_supported(
    argv: list[str], engine_fields: set[str], frontend_fields: set[str]
) -> list[str]:
    known = engine_fields | frontend_fields
    return [
        f"설치된 vLLM 에 없는 옵션: --{f.replace('_', '-')}"
        for f in sorted(used_fields(argv))
        if f not in known
    ]


def installed_vllm() -> dict:
    """vLLM venv 에서만 동작. 버전과 옵션 필드 목록."""
    import vllm
    from vllm.engine.arg_utils import EngineArgs

    try:
        from vllm.entrypoints.common.cli_args import FrontendArgs

        frontend = {f.name for f in dataclasses.fields(FrontendArgs)}
    except Exception:  # noqa: BLE001 - 경로가 바뀐 버전
        frontend = set(FRONTEND_FIELDS)
    info = {
        "vllm": vllm.__version__,
        "engine_fields": {f.name for f in dataclasses.fields(EngineArgs)},
        "frontend_fields": frontend,
    }
    try:
        import torch

        info.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
    except Exception:  # noqa: BLE001
        pass
    return info


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_status(cuda_visible: str | None) -> dict:
    """선택한 GPU 의 상태와 그 GPU 를 쓰는 프로세스 (CUDA_DEVICE_ORDER=PCI_BUS_ID 기준 index)."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    rows = (
        _run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        or ""
    )
    gpus = []
    for line in rows.splitlines():
        idx, uuid, name, bus, total, used = (x.strip() for x in line.split(","))
        gpus.append(
            {
                "index": idx,
                "uuid": uuid,
                "name": name,
                "pci_bus_id": bus,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    wanted = {d.strip() for d in (cuda_visible or "").split(",") if d.strip()}
    selected = [g for g in gpus if g["index"] in wanted or g["uuid"] in wanted]
    apps = (
        _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        or ""
    )
    uuids = {g["uuid"] for g in selected}
    busy = [line.strip() for line in apps.splitlines() if line.split(",")[0].strip() in uuids]
    return {"available": True, "all": gpus, "selected": selected, "busy_processes": busy}


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def preflight(
    profile: dict,
    server: dict,
    argv: list[str],
    env: dict,
    installed: dict | None,
    gpus: dict | None,
    args,
) -> list[str]:
    errors = check_gpu_assignment(server, env.get("CUDA_VISIBLE_DEVICES"))
    if not env.get("VLLM_API_KEY"):
        errors.append("VLLM_API_KEY 미설정")
    if profile["llm"].get("model") != server["model"]:
        errors.append(f"프로필 llm.model ({profile['llm'].get('model')}) != server.model")
    if installed is not None:
        want = str(server.get("vllm_version", ""))
        if want and installed["vllm"] != want and not args.allow_version_mismatch:
            errors.append(f"설치된 vLLM {installed['vllm']} != 프로필 {want}")
        errors += check_supported(argv, installed["engine_fields"], installed["frontend_fields"])
    if gpus and gpus.get("available"):
        if len(gpus["selected"]) != server.get("tensor_parallel_size", 1):
            errors.append(
                f"CUDA_VISIBLE_DEVICES 와 일치하는 GPU 를 nvidia-smi 에서 찾지 못함: "
                f"{env.get('CUDA_VISIBLE_DEVICES')}"
            )
        if gpus["busy_processes"] and not args.allow_busy_gpu:
            errors.append(f"선택한 GPU 를 쓰는 프로세스가 있음: {gpus['busy_processes']}")
    return errors


def main(argv=None, env=None, installed=None, gpus=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument(
        "--smoke", action="store_true", help="--enforce-eager (첫 실행용, 성능 측정 금지)"
    )
    ap.add_argument("--sched-policy", default="fcfs", choices=SCHED_POLICIES)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument(
        "--dev-mode",
        action="store_true",
        help="VLLM_SERVER_DEV_MODE=1 (/reset_prefix_cache 사용. 인증 없는 endpoint 가 열림)",
    )
    ap.add_argument("--allow-version-mismatch", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir", default=str(Path.home() / "crag_logs"))
    a = ap.parse_args(argv)

    env = dict(os.environ if env is None else env)
    profile_path = Path(a.profile)
    profile = load_profile(profile_path)
    overrides = {
        "max_num_seqs": a.max_num_seqs,
        "port": a.port,
        "gpu_memory_utilization": a.gpu_memory_utilization,
    }
    server = effective_server(profile, overrides)
    cmd = build_argv(server, a.sched_policy, a.smoke)
    if installed is None:
        try:
            installed = installed_vllm()
        except ImportError:
            if not a.dry_run:
                raise LaunchError("vllm import 실패: vLLM 전용 venv 에서 실행할 것") from None
            # dry-run 을 vLLM 없는 환경에서 돌리면 버전·옵션 검사는 생략된다
    if gpus is None:
        gpus = gpu_status(env.get("CUDA_VISIBLE_DEVICES"))
    errors = preflight(profile, server, cmd, env, installed, gpus, a)
    if not a.dry_run and not port_free(int(server["port"])):
        errors.append(f"포트 {server['port']} 사용 중")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{profile['engine_profile']}_{a.sched_policy}_{ts}"
    log_dir = Path(a.log_dir)
    meta = {
        "created_utc": datetime.now(UTC).isoformat(),
        "argv": cmd,
        "profile": {
            "name": profile["engine_profile"],
            "path": str(profile_path),
            "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "server": server,
            "estimates": profile.get("estimates"),
        },
        "overrides": {k: v for k, v in overrides.items() if v is not None},
        "scheduling_policy": a.sched_policy,
        "smoke": a.smoke,
        "dev_mode": a.dev_mode,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "env": {
            k: ("<set>" if any(m in k for m in ("KEY", "TOKEN")) else v)
            for k, v in env.items()
            if k.startswith(("CUDA_", "VLLM_", "HF_", "NCCL_"))
        },
        "installed": None
        if installed is None
        else {k: v for k, v in installed.items() if not k.endswith("_fields")},
        "gpus": gpus,
        "log": str(log_dir / f"{stem}.log"),
        "preflight_errors": errors,
    }
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "argv",
                    "scheduling_policy",
                    "smoke",
                    "cuda_visible_devices",
                    "preflight_errors",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if errors:
        print("실행 전 검사 실패. 서버를 띄우지 않습니다.", file=sys.stderr)
        return 2
    if a.dry_run:
        print("dry-run: 검사 통과 (실행하지 않음)")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / f"{stem}_launch.json"
    with open(meta_path, "x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"launch metadata: {meta_path}\nlog: {meta['log']}")
    print(
        "준비 완료 표시: 로그의 'Application startup complete'. 다른 창에서 verify_launch.py 실행."
    )

    child_env = {**env, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if a.dev_mode:
        child_env["VLLM_SERVER_DEV_MODE"] = "1"
    with open(meta["log"], "x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
