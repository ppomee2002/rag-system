"""MODE A serving 점검 (LLaMA2-7B fp16 TP=2 + 공식 T5 fp32).

generator / precision / evaluator dtype 은 바꾸지 않는다. 바꿀 수 있는 것은 GPU 배치와 vLLM 메모리
예약 (gpu_memory_utilization) 뿐이며, 바꾸면 정확도 재검증 범위를 gpu_plan.json 에 적는다.

  t5-memory   (vLLM 을 띄우기 전에) T5 를 GPU 에 올려 최악 입력
              (가장 긴 grading 쌍, truncation 없음)
              으로 최대 메모리를 잰다 -> T5 와 vLLM TP=2 가 같은 GPU 에 같이 들어가는 vLLM
              gpu_memory_utilization 계산 + KV cache 추정.
  concurrent  (vLLM 을 띄운 뒤) T5 단독 / generator 단독 (동시 1·4·8·16) / 동시 실행 -> 실제 실행
              중첩, 서로의 지연 증가, vLLM 동시 처리 여부.

    CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official \
        python experiments/c2_gating/probe_serving.py t5-memory \
        --mode-a-config experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml \
        --crag-preds results/official_crag/mode_a/generate_crag/20261127_023204
-> results/c2_gating/probe/<run_id>/{gpu_plan.json | probe.json}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_components import build_engine, build_runner  # noqa: E402
from c2_data import load_items, read_jsonl  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official import source  # noqa: E402
from rag.crag_official.evaluator import normalize_passage, pair_text  # noqa: E402
from rag.crag_official.t5_service import T5Service  # noqa: E402
from rag.llm.engine import LLMRequest  # noqa: E402
from rag.telemetry.run_meta import create_run_dir, gpu_snapshot, write_new  # noqa: E402

ROOT = HERE.parents[1]
GIB = 1024**3
DEFAULT_UTIL = 0.85


def own_gpu_mib() -> float | None:
    """nvidia-smi 의 이 프로세스 GPU 메모리 (CUDA context 포함)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            return float(parts[1])
    return None


def recommend_util(total_gib: float, need_gib: float, util_now: float) -> tuple[bool, float]:
    """(현재 설정으로 들어가는지, 권장 gpu_memory_utilization). 들어가면 현재 값 유지."""
    fits = total_gib * (1 - util_now) >= need_gib
    if fits:
        return True, util_now
    return False, math.floor((total_gib - need_gib) / total_gib * 100) / 100


def cmd_t5_memory(a, cfg) -> dict:
    import torch

    items, _ = load_items(a.repo)
    free0, total = torch.cuda.mem_get_info(0)
    runner = build_runner(cfg)
    svc = T5Service(runner, (cfg.get("t5_service") or {}).get("batch_size", 1))
    t0 = time.perf_counter()
    lens = []
    for it in items:
        for p in it.passages:
            pp = normalize_passage(p)
            if pp.strip():
                t = pair_text(it.question, pp)
                lens.append((runner.encode(t, False).length, t))
    lens.sort(key=lambda x: -x[0])
    tok_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats(0)
    worst = [t for _, t in lens[:5]]
    svc.submit("mem", "grading", worst, False).wait()  # 가장 긴 입력 (truncation 없음)
    svc.submit("mem2", "external_refine", worst, True).wait()  # refinement (512 truncation)
    svc.close()
    peak_reserved = torch.cuda.max_memory_reserved(0)
    proc_mib = own_gpu_mib()
    proc_gib = proc_mib / 1024 if proc_mib is not None else peak_reserved / GIB + 0.5
    total_gib = total / GIB
    need = proc_gib + a.margin_gib
    srv = cfg.get("server") or {}
    est = cfg.get("estimates") or {}
    tp = int(srv.get("tensor_parallel_size", 1))
    util_now = float(srv.get("gpu_memory_utilization", DEFAULT_UTIL))
    fits_now, rec = recommend_util(total_gib, need, util_now)
    kv_gib = rec * total_gib - est.get("weights_gib_per_gpu", 6.3) - a.vllm_overhead_gib
    kv_tokens = max(0, int(kv_gib * GIB / (est.get("kv_bytes_per_token", 524288) / tp)))
    plen = None
    if a.crag_preds:
        pt = sorted(
            r["prompt_tokens"]
            for r in read_jsonl(Path(a.crag_preds) / "preds.jsonl")
            if r.get("prompt_tokens")
        )
        plen = {"mean": sum(pt) / len(pt), "p95": pt[int(0.95 * (len(pt) - 1))], "max": pt[-1]}
    need_seq = None if plen is None else plen["p95"] + cfg["llm"].get("max_tokens", 100)
    return {
        "gpu_total_gib": total_gib,
        "gpu_free_before_t5_gib": free0 / GIB,
        "t5_process_gpu_gib": proc_gib,
        "t5_process_gpu_source": "nvidia-smi" if proc_mib is not None else "torch reserved + 0.5",
        "t5_peak_reserved_gib": peak_reserved / GIB,
        "longest_grading_tokens": lens[0][0] if lens else None,
        "n_grading_pairs_over_512": sum(1 for n, _ in lens if n > source.EVAL_MAX_LENGTH),
        "tokenize_all_seconds": tok_s,
        "margin_gib": a.margin_gib,
        "profile_gpu_memory_utilization": util_now,
        "fits_with_profile_setting": fits_now,
        "recommended_gpu_memory_utilization": rec,
        "change_required": not fits_now,
        "kv_estimate_at_recommended": {
            "kv_gib_per_gpu": kv_gib,
            "kv_tokens": kv_tokens,
            "vllm_overhead_gib_assumed": a.vllm_overhead_gib,
            "max_model_len": srv.get("max_model_len"),
            "max_num_seqs": srv.get("max_num_seqs"),
            "crag_prompt_tokens": plen,
            "seqs_at_p95_len": (kv_tokens // need_seq) if need_seq else None,
            "note": "추정치. 실제 KV 는 vLLM 로그의 'GPU KV cache size' 로 확인",
        },
        "placement": (
            "vLLM TP=2: CUDA_VISIBLE_DEVICES=0,1 (두 GPU 모두 사용). T5: 물리 GPU "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')} (vLLM rank 와 같은 GPU 공유 -> 연산 경합, "
            "B/C/Cbn 공통 조건)"
        ),
        "unchanged": "generator meta-llama/Llama-2-7b-hf, fp16, TP=2, max_model_len, "
        "max_num_seqs, prompt, sampling, T5 fp32",
        "accuracy_reverification": None
        if fits_now
        else {
            "why": "vLLM 메모리 예약을 줄이면 KV cache 크기가 바뀐다. 가중치·dtype·kernel 은 같아 "
            "순차 greedy 출력은 같을 것으로 예상하지만 검증 전이다.",
            "scope_required": [
                "MODE A generate --method crag (1,399문항, 순차) 를 새 설정으로 다시 실행 -> "
                "20261127_023204 와 문항별 prediction 비교 + official_match (769/1399 재현 여부)",
                "MODE A generate --method rag (1,399, 순차) -> 20261126_093945 와 비교 "
                "(705/1399). Vanilla 대비 +4.575%p 를 새 serving 설정에서도 주장하려면 필요",
            ],
            "not_required": "T5 score run (별도 프로세스·fp32·같은 가중치. "
            "smoke 의 판정 대조로 확인)",
        },
    }


def gen_burst(engine, prompts, n):
    out = [None] * n

    def one(k):
        t0 = time.perf_counter()
        engine.complete(LLMRequest(prompts[k % len(prompts)], kind="rewrite", max_tokens=32))
        out[k] = (t0, time.perf_counter())

    ths = [threading.Thread(target=one, args=(k,)) for k in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


def cmd_concurrent(a, cfg) -> dict:
    items, _ = load_items(a.repo)
    runner = build_runner(cfg)
    engine = build_engine(cfg)
    svc = T5Service(runner)
    qs = items[: a.n_rep]
    texts = [
        [pair_text(q.question, normalize_passage(p)) for p in q.passages if p.strip()] for q in qs
    ]
    prompts = [source.POPQA_KEYWORD_PROMPT.replace("{question}", q.question) for q in qs]
    res: dict = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_before": gpu_snapshot(),
        "t5_runner": runner.describe(),
        "engine": engine.describe(),
    }
    svc.submit("warm", "grading", texts[0], False).wait()
    engine.complete(LLMRequest(prompts[0], kind="rewrite", max_tokens=32))
    solo = []
    for k, t in enumerate(texts):
        j = svc.submit(f"s{k}", "grading", t, False)
        j.wait()
        solo.append(j.busy_s)
    res["t5_solo_grading_s"] = solo
    res["gen_solo_wall_s"] = {
        n: [e - s for s, e in gen_burst(engine, prompts, n)] for n in (1, 4, 8, 16)
    }
    both = []
    for k, t in enumerate(texts):
        j = svc.submit(f"b{k}", "grading", t, False)
        spans = gen_burst(engine, prompts, 4)
        j.wait()
        ov = [max(0.0, min(j.t_end, e) - max(j.t_start, s)) for s, e in spans]
        both.append(
            {"t5_busy_s": j.busy_s, "gen_wall_s": [e - s for s, e in spans], "overlap_s": ov}
        )
    res["concurrent"] = both
    res["gpu_after"] = gpu_snapshot()
    svc.close()

    def m(x):
        return sum(x) / len(x) if x else None

    t5_both = m([b["t5_busy_s"] for b in both])
    res["summary"] = {
        "t5_solo_mean_s": m(solo),
        "t5_with_gen_mean_s": t5_both,
        "t5_slowdown": (t5_both or 0) / (m(solo) or 1),
        "gen_solo_mean_s": {n: m(v) for n, v in res["gen_solo_wall_s"].items()},
        "gen_with_t5_mean_s": m([x for b in both for x in b["gen_wall_s"]]),
        "exec_overlap_frac": m([float(any(o > 0 for o in b["overlap_s"])) for b in both]),
        "vllm_concurrent_ok": (m(res["gen_solo_wall_s"][4]) or 0)
        < 2.5 * (m(res["gen_solo_wall_s"][1]) or 0),
        "note": "vllm_concurrent_ok: 동시 4 의 요청당 wall 이 단독의 2.5배 미만이면 동시 처리로 봄 "
        "(대략 판단). max_num_seqs 를 넘는 요청은 vLLM 안에서 대기",
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["t5-memory", "concurrent"])
    ap.add_argument("--mode-a-config", required=True)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "CRAG"))
    ap.add_argument("--crag-preds", default=None, help="MODE A generate_crag run (prompt 길이)")
    ap.add_argument("--margin-gib", type=float, default=0.75)
    ap.add_argument(
        "--vllm-overhead-gib",
        type=float,
        default=1.5,
        help="vLLM activation·CUDA graph 등 (KV 추정용 가정)",
    )
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    cfg = load_layered_config(a.mode_a_config, [a.config], a.set)
    res = cmd_t5_memory(a, cfg) if a.cmd == "t5-memory" else cmd_concurrent(a, cfg)
    d = create_run_dir("c2_gating/probe", "results", a.run_id)
    name = "gpu_plan.json" if a.cmd == "t5-memory" else "probe.json"
    write_new(d / name, json.dumps(res, indent=2, ensure_ascii=False, default=str))
    show = res if a.cmd == "t5-memory" else res["summary"]
    print(json.dumps(show, indent=2, ensure_ascii=False, default=str))
    print(f"saved -> {d / name}")


if __name__ == "__main__":
    main()
