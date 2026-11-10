"""실험 시작 전 설정 검증 (논문 결과의 타당성에 직접 영향을 주는 조합을 조용히 허용하지 않는다).

1. priority 설정 일치
   - scheduler.engine_priority (client 가 priority 를 만드는지)
   - llm.send_priority         (VLLMEngine 이 priority 를 요청에 넣는지)
   - llm.server_scheduling_policy (vLLM 서버를 어떤 --scheduling-policy 로 띄웠다고 선언했는지)
   - llm.launch_meta           (launch_vllm.sh 가 남긴 실행 기록. 있으면 실제 인자와 대조)
   - 서버 probe                (priority=1 요청을 보내 수락 여부 확인. 요청 1회, 선택)
2. cost model 데이터 누수
   - 학습 질문 (train_query_ids) 과 평가 질문이 겹치면 오류. OOF 모델이면 자기 fold 제외로 허용.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.scheduling.cost_model import OOFCostModel

SERVER_POLICIES = ("fcfs", "priority")


class PreflightError(ValueError):
    pass


def check_priority_config(scheduler_cfg: dict | None, llm_cfg: dict) -> dict:
    s = scheduler_cfg or {}
    policy = s.get("policy", "none") or "none"
    engine_priority = s.get("engine_priority", "none") or "none"
    send = bool(llm_cfg.get("send_priority", False))
    provider = llm_cfg.get("provider")
    server = llm_cfg.get("server_scheduling_policy")
    if server is not None and server not in SERVER_POLICIES:
        raise PreflightError(f"llm.server_scheduling_policy: {SERVER_POLICIES} (got {server})")
    if engine_priority == "key":
        if policy == "none":
            raise PreflightError("scheduler.engine_priority=key 에는 scheduler.policy 가 필요")
        if provider != "vllm":
            raise PreflightError(f"engine priority 는 vLLM 만 지원 (provider={provider})")
        if not send:
            raise PreflightError(
                "scheduler.engine_priority=key 인데 llm.send_priority=false: priority 가 전송되지 "
                "않음. 둘 다 켜거나 둘 다 끌 것"
            )
        if server != "priority":
            raise PreflightError(
                "engine priority 를 보내려면 llm.server_scheduling_policy=priority 로 선언하고 "
                "서버를 --scheduling-policy priority 로 띄울 것"
            )
    elif send:
        raise PreflightError(
            "llm.send_priority=true 인데 scheduler.engine_priority 가 key 가 아님: "
            "보낼 priority 가 없음. 의도한 조건인지 확인 후 둘 다 맞출 것"
        )
    return {
        "scheduler_policy": policy,
        "engine_priority": engine_priority,
        "send_priority": send,
        "declared_server_policy": server,
    }


def launch_meta_policy(path: str | Path) -> dict:
    """launch_vllm.sh 의 *_launch.json 에서 모델과 --scheduling-policy 를 읽는다."""
    path = Path(path).expanduser()
    meta = json.loads(path.read_text(encoding="utf-8"))
    argv = meta["argv"]

    def arg(name):
        return argv[argv.index(name) + 1] if name in argv else None

    model = argv[2] if len(argv) > 2 and argv[1] == "serve" else None
    tp = arg("--tensor-parallel-size")
    return {
        "path": str(path),
        "model": model,
        "scheduling_policy": arg("--scheduling-policy"),
        "max_num_seqs": arg("--max-num-seqs"),
        "max_model_len": arg("--max-model-len"),
        "tensor_parallel_size": int(tp) if tp else 1,
        "generation_config": arg("--generation-config"),
        "enforce_eager": "--enforce-eager" in argv,
        "cuda_visible_devices": meta.get("cuda_visible_devices"),
        "profile_name": (meta.get("profile") or {}).get("name"),
        "profile_sha256": (meta.get("profile") or {}).get("sha256"),
        "smoke": meta.get("smoke"),
        "argv": argv,
    }


def check_launch_meta(
    path: str | Path,
    llm_cfg: dict,
    server_cfg: dict | None = None,
    profile_name: str | None = None,
) -> dict:
    """launch 기록 (launch_vllm.py *_launch.json) 과 client config / 엔진 프로필을 대조."""
    info = launch_meta_policy(path)
    declared = llm_cfg.get("server_scheduling_policy")
    actual = info["scheduling_policy"] or "fcfs"  # 인자 없으면 vLLM 기본 fcfs
    if declared is not None and declared != actual:
        raise PreflightError(
            f"서버 launch 의 --scheduling-policy={actual} 와 config 선언 {declared} 불일치 ({path})"
        )
    if info["model"] and info["model"] != llm_cfg.get("model"):
        raise PreflightError(
            f"서버 모델 {info['model']} 와 llm.model {llm_cfg.get('model')} 불일치"
        )
    visible = info["cuda_visible_devices"]
    if visible is not None:
        n_visible = len([d for d in str(visible).split(",") if d.strip()])
        if n_visible != info["tensor_parallel_size"]:
            raise PreflightError(
                f"launch 기록의 GPU 수 ({n_visible}) != tensor_parallel_size "
                f"({info['tensor_parallel_size']})"
            )
    if server_cfg:
        expect = {
            "tensor_parallel_size": server_cfg.get("tensor_parallel_size"),
            "max_model_len": None
            if server_cfg.get("max_model_len") is None
            else str(server_cfg["max_model_len"]),
            "generation_config": server_cfg.get("generation_config"),
        }
        for key, want in expect.items():
            if want is not None and info[key] != want:
                raise PreflightError(
                    f"launch 기록의 {key}={info[key]} 와 엔진 프로필 {want} 불일치"
                )
    if profile_name and info["profile_name"] and info["profile_name"] != profile_name:
        raise PreflightError(
            f"launch 프로필 {info['profile_name']} 와 config engine_profile {profile_name} 불일치"
        )
    return {**info, "actual_scheduling_policy": actual}


def probe_server_priority(engine, declared: str | None) -> dict:
    """priority=1 짧은 요청으로 서버가 priority 를 받는지 확인 (요청 1회).

    수락 여부로 알 수 있는 것은 제한적이다:
      - 선언 priority 인데 거부 -> 확실한 불일치 (오류)
      - 선언 fcfs 인데 수락     -> vLLM 0.33.0 소스에는 FCFS 서버가 priority 를 거부하는 코드가 없어
                                   수락이 정상일 수 있다. 정책 판별은 처리 순서 검증
                                   (scripts/server/probe_engine.py priority_order) 과
                                   launch 기록으로 한다.
    """
    from rag.llm.engine import EngineError, LLMRequest
    from rag.llm.vllm_engine import VLLMEngine

    if not isinstance(engine, VLLMEngine):
        return {"skipped": f"engine {engine.name} 는 priority 미지원"}
    probe = VLLMEngine(
        engine.model,
        base_url=engine.base_url,
        api_key_env=engine.api_key_env,
        stream=False,
        send_priority=True,
        seed=engine.seed,
        transport=engine.transport,
    )
    try:
        probe.complete(LLMRequest("ping", max_tokens=1, priority=1))
        accepted, detail = True, None
    except EngineError as e:
        if e.status in (401, 403, None):
            raise PreflightError(f"서버 접근 실패로 priority 확인 불가: {e}") from None
        accepted, detail = False, f"HTTP {e.status}: {(e.body or '')[:200]}"
    out = {"nonzero_priority_accepted": accepted, "detail": detail, "declared": declared}
    if declared == "priority" and not accepted:
        raise PreflightError(f"서버가 priority 요청을 거부함 (선언: priority): {detail}")
    if declared == "fcfs" and accepted:
        out["note"] = "FCFS 서버도 priority 를 수락할 수 있음 (0.33.0). 수락 여부로 정책 판별 불가"
    return out


def check_cost_model_leakage(model, eval_query_ids, allow_overlap: bool = False) -> dict:
    """학습 질문과 평가 질문이 겹치는지 검사.

    - 단일 모델: 겹치면 오류 (allow_overlap=true 로 명시해야 허용, 결과에 기록)
    - OOF 모델 : 겹친 질문은 자기 fold 를 제외한 모델로 예측되므로 허용
    - 학습 질문 목록이 없는 모델 (이전 형식) : 검증 불가 -> allow_overlap 없으면 오류
    """
    if model is None:
        return {"status": "no_cost_model"}
    eval_ids = set(eval_query_ids)
    train = model.train_query_ids
    if train is None:
        if not allow_overlap:
            raise PreflightError("cost model 에 train_query_ids 가 없어 누수 검증 불가")
        return {"status": "unverified_allowed"}
    overlap = sorted(eval_ids & set(train))
    base = {"n_train": len(train), "n_eval": len(eval_ids), "n_overlap": len(overlap)}
    if isinstance(model, OOFCostModel):
        return {"status": "oof", **base}
    if overlap and not allow_overlap:
        raise PreflightError(
            f"cost model 학습 질문과 평가 질문이 {len(overlap)}개 겹침 (예: {overlap[:3]}). "
            "dataset.split 으로 분리하거나 OOF (scheduler.cost_model.oof_folds) 사용"
        )
    return {"status": "overlap_allowed" if overlap else "disjoint", **base}
