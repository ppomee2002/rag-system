"""C2 Bottleneck-aware Gating 실험 runner (MODE A 위의 B / C / Cbn).

    # smoke (판정별 4문항, B/C/Cbn x 동시 1/4)
    uv run --extra crag --extra official python experiments/c2_gating/run.py \
        --mode-a-config experiments/official_crag/configs/mode_a_popqa.yaml \
        --reference-scores results/official_crag/mode_a/score/<run> --per-action 4 \
        --modes B,C,Cbn --concurrency 1,4 --repeats 1 --purpose smoke

설정 계층: configs/base.yaml + <MODE A generator 설정 (정확도 상승을 확인한 그 파일)> +
experiments/c2_gating/config.yaml + --overlay + --set. MODE A 설정의 llm 블록은 c2 설정이 덮어쓰지
못한다 (덮어쓰면 실패).

같은 run 안의 모든 조건은 같은 T5 가중치 (한 번 로드), 같은 generator 서버, 같은 질문 집합, 같은 web
기록을 쓴다. 조건 실행 순서는 repetition 마다 seed 로 섞고 (plan.json), 요청 순서도 repetition 마다
섞되 같은 repetition 의 조건끼리는 같다.

결과: results/c2_gating/<gating.experiment_mode>/<purpose>/<run_id>/
  (controlled_replay 와 runtime_e2e 는 폴더부터 분리한다)
  config.yaml, run_meta.json, plan.json, summary.json
  conds/<NN>_<mode>_c<k>_r<rep>/{queries,t5_jobs,chains,gpu}.jsonl + summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_components import GpuSampler, build_engine, build_runner  # noqa: E402
from c2_data import (  # noqa: E402
    load_items,
    reference_actions,
    reference_preds,
    select_indices,
    warmup_pool,
)
from c2_metrics import condition_summary, identity_check  # noqa: E402
from c2_pipeline import GatedOfficialCRAG, GatingConfig  # noqa: E402
from c2_web import WebReplay  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official.t5_service import T5Service  # noqa: E402
from rag.telemetry.run_meta import (  # noqa: E402
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    write_new,
    write_run_files,
)

ROOT = HERE.parents[1]

CLAIM_SCOPE = {
    "controlled_replay": (
        "Controlled Replay: generation evidence = 저자 사전 생성 지식 (고정). "
        "correction workload 는 "
        "실제 실행. B/C/Cbn serving 비용 비교만 주장한다. 정확도 동일성은 evidence 고정의 결과이며 "
        "End-to-End CRAG 정확도 보존의 증거가 아니다."
    ),
    "runtime_e2e": (
        "Runtime End-to-End: 실행 중 correction 결과를 generation 에 사용. "
        "B/C/Cbn 의 action·evidence·"
        "prompt·정확도·성능 비교. 저자 지식 (MODE A) 과 evidence 가 달라 MODE A 정확도와 직접 "
        "동일하다고 주장하지 않는다 (참고 비교만)."
    ),
}


# ---------------------------------------------------------------- plan
def make_plan(modes, concs, repeats, seed, order) -> list[dict]:
    plan = []
    for rep in range(repeats):
        cells = [(m, c) for c in concs for m in modes]
        if order == "shuffled":
            random.Random(seed * 1000 + rep).shuffle(cells)
        elif order == "rotated":  # 동시 요청 수는 오름차순, mode 순서는 repetition 마다 회전
            k = rep % len(modes)
            rot = modes[k:] + modes[:k]
            cells = [(m, c) for c in concs for m in rot]
        plan += [{"mode": m, "concurrency": c, "rep": rep} for m, c in cells]
    for i, p in enumerate(plan):
        p["cond_id"] = f"{i:02d}_{p['mode']}_c{p['concurrency']}_r{p['rep']}"
    return plan


# ---------------------------------------------------------------- condition
def run_condition(cond, items, order, warmup, cfg, runner, engine, web, out_dir: Path) -> dict:
    t5c = cfg["t5_service"]
    service = T5Service(
        runner, t5c.get("batch_size", 1), t5c.get("memo", False), t5c.get("memo_verify", False)
    )
    g = cfg["gating"]
    gcfg = GatingConfig(
        mode=cond["mode"],
        experiment_mode=g["experiment_mode"],
        internal_refinement=g.get("internal_refinement", "run"),
        prompt_version=g.get("prompt_version", "v3_2024_10"),
        decompose_mode=(cfg.get("refinement") or {}).get("decompose_mode", "selection"),
        rewrite_max_tokens=g.get("rewrite_max_tokens", 32),
        wiki_prefix=g.get("wiki_prefix", True),
        generation_max_tokens=cfg["llm"].get("max_tokens", 100),
        record_texts=g.get("record_texts", False),
    )
    pipe = GatedOfficialCRAG(gcfg, service, engine, web)
    d = out_dir / cond["cond_id"]
    d.mkdir(parents=True)
    qw = JsonlWriter(d / "queries.jsonl")
    threads_before = threading.active_count()

    t0 = time.perf_counter()
    for k, i in enumerate(warmup):  # 순차 warmup (집계 제외)
        r = pipe.run(items[i], f"w{k}", t0)
        qw.write({**r, "warmup": True})
    for c in pipe.all_chains:
        c.join(60)
    service.idle(60)
    n_warm_jobs, n_warm_chains = len(service.jobs), len(pipe.all_chains)

    gpu = GpuSampler(d / "gpu.csv", cfg["run"].get("gpu_sample_ms", 200)).start()
    rows: list[dict] = []
    lock = threading.Lock()
    it = iter(order)
    t0 = time.perf_counter()

    def worker():
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            rid = f"q{i}"
            r = pipe.run(items[i], rid, t0)
            r["concurrency"] = cond["concurrency"]
            with lock:
                rows.append(r)
            qw.write(r)

    ws = [threading.Thread(target=worker, name=f"client-{k}") for k in range(cond["concurrency"])]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    makespan = time.perf_counter() - t0
    gpu_stats = gpu.stop()

    # ---- 누수 점검: 남은 chain (버린 rewrite 등) 을 기다린 뒤 queue / 실행 / memo
    joined = all(c.join(cfg["run"].get("chain_join_timeout_s", 120)) for c in pipe.all_chains)
    idle = service.idle(60)
    snap = service.snapshot()
    leak = {
        "chains_all_joined": joined,
        "open_chains": len(pipe.open_chains),
        "t5_idle": idle,
        "t5_queue_len": snap["queue_len"],
        "t5_running": snap["running"],
        "memo_requests_left": snap["memo_requests"],
        "unfinished_jobs": sum(1 for j in service.jobs if not j.done.is_set()),
        "thread_delta": threading.active_count() - threads_before,
    }
    service.close()
    leak["thread_delta_after_close"] = threading.active_count() - threads_before
    leak["ok"] = (
        joined
        and idle
        and not leak["open_chains"]
        and not snap["queue_len"]
        and snap["running"] is None
        and not snap["memo_requests"]
        and not leak["unfinished_jobs"]
        and leak["thread_delta_after_close"] <= 0
    )

    action_of = {r["request_id"]: r.get("action") for r in rows}
    jobs = []
    for j in service.jobs[n_warm_jobs:]:
        jd = j.to_dict(t0)
        if j.kind == "external_refine":
            jd["usage"] = "wasted" if action_of.get(j.request_id) == "correct" else "required"
        else:
            jd["usage"] = "required"
        jobs.append(jd)
    chains = [c.to_dict(t0) for c in pipe.all_chains[n_warm_chains:]]
    for name, data in (("t5_jobs", jobs), ("chains", chains)):
        w = JsonlWriter(d / f"{name}.jsonl")
        for x in data:
            w.write(x)
    summ = condition_summary(rows, jobs, chains, makespan, gpu_stats)
    summ.update(
        cond=cond,
        leak=leak,
        t5_service=service.describe(),
        t5_max_queue_len=service.max_queue_len,
        web=web.stats(),
        engine_max_active=getattr(engine, "max_active", None),
    )
    write_new(d / "summary.json", json.dumps(summ, indent=2, ensure_ascii=False, default=str))
    return {"summary": summ, "rows": rows}


# ---------------------------------------------------------------- main
def load_cfg(a) -> dict:
    mode_a = load_layered_config(a.mode_a_config, [], [])
    cfg = load_layered_config(a.mode_a_config, [a.config, *a.overlay], a.set)
    if a.mock:
        cfg["llm"] = {**cfg["llm"], "provider": "mock"}
        cfg["evaluator"] = {**cfg["evaluator"], "backend": "mock"}
    elif cfg["llm"] != mode_a["llm"]:
        diff = {
            k: (mode_a["llm"].get(k), v) for k, v in cfg["llm"].items() if mode_a["llm"].get(k) != v
        }
        raise SystemExit(f"c2 설정이 MODE A llm 블록을 바꿈 (MODE A 설정을 그대로 써야 함): {diff}")
    cfg["mode_a_config"] = a.mode_a_config
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode-a-config", required=True, help="정확도 상승을 확인한 MODE A generator 설정"
    )
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "CRAG"))
    ap.add_argument("--answers", default=None, help="Self-RAG popqa_longtail_w_gs.jsonl")
    ap.add_argument("--web-cache", action="append", default=[], help="prefetch_web.py 결과")
    ap.add_argument("--reference-scores", default=None, help="MODE A score run (판정 대조·층화)")
    ap.add_argument("--reference-preds", default=None, help="MODE A generate_crag run (예측 대조)")
    ap.add_argument("--indices-file", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-action", type=int, default=None)
    ap.add_argument("--modes", default="B,C,Cbn")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--order", default="shuffled", choices=["shuffled", "rotated", "fixed"])
    ap.add_argument(
        "--purpose", default="pre", choices=["smoke", "gonogo", "main", "sensitivity", "pre"]
    )
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--mock", action="store_true", help="T5·generator 를 mock 으로 (로컬 점검)")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument(
        "--vllm-launch-meta", default=None, help="launch_vllm.py 의 *_launch.json (서빙 설정 기록)"
    )
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    cfg = load_cfg(a)
    seed = int(cfg.get("seed", 0))
    items, data_info = load_items(a.repo, a.answers)
    ref_a = reference_actions(a.reference_scores) if a.reference_scores else None
    idx, sel = select_indices(
        items,
        indices_file=a.indices_file,
        limit=a.limit,
        per_action=a.per_action,
        ref_actions=ref_a,
        seed=seed,
    )
    wcfg = cfg.get("web") or {}
    if not a.web_cache:
        raise SystemExit("--web-cache (prefetch_web.py 결과) 필요")
    web = WebReplay(
        a.web_cache,
        wcfg.get("latency_mode", "recorded"),
        wcfg.get("wikipedia_only", True),
        wcfg.get("top_k", 5),
        wcfg.get("latency_scale", 1.0),
    )
    missing = [i for i in idx if web.recorded_latency(str(i)) is None]
    if missing and not cfg["run"].get("allow_web_miss", False):
        raise SystemExit(f"web 기록 없는 질문 {len(missing)}개: {missing[:10]}")
    modes = a.modes.split(",")
    concs = [int(x) for x in a.concurrency.split(",")]
    plan = make_plan(modes, concs, a.repeats, seed, a.order)
    n_warm = int(cfg["run"].get("warmup_queries", 3))
    sel_set = set(idx)
    warmup = [
        i
        for i in warmup_pool(len(items), seed)
        if i not in sel_set and web.recorded_latency(str(i)) is not None
    ][:n_warm]
    if len(warmup) < n_warm:
        raise SystemExit(
            f"warmup 질문의 web 기록 부족 ({len(warmup)}/{n_warm}): prefetch_web.py 확인"
        )
    orders = {}
    for rep in range(a.repeats):
        o = list(idx)
        if a.order != "fixed":
            random.Random(seed * 7919 + rep).shuffle(o)
        orders[rep] = o

    runner = build_runner(cfg)
    engine = build_engine(cfg)
    emode = cfg["gating"]["experiment_mode"]
    run_dir = create_run_dir(f"c2_gating/{emode}/{a.purpose}", a.output_dir, a.run_id)
    out = run_dir / "conds"
    out.mkdir()
    saved = copy.deepcopy(cfg)
    saved["cli"] = vars(a)
    write_run_files(
        run_dir,
        saved,
        collect_run_metadata(
            saved,
            {"generator": engine},
            {
                "experiment_mode": "A_official_reproduction + C2 gating",
                "gating_experiment_mode": emode,
                "claim_scope": CLAIM_SCOPE[emode],
                "deviations": cfg.get("deviations"),
                "selection": sel,
                "data": data_info,
                "web": web.describe(),
                "t5_runner": runner.describe(),
                "vllm_launch": vllm_launch_view(a.vllm_launch_meta),
                "warmup_indices": warmup,
            },
        ),
    )
    write_new(
        run_dir / "plan.json",
        json.dumps({"plan": plan, "orders": orders, "indices": idx}, indent=2),
    )
    results = {}
    for cond in plan:
        print(f"== {cond['cond_id']} (n={len(idx)})", flush=True)
        res = run_condition(cond, items, orders[cond["rep"]], warmup, cfg, runner, engine, web, out)
        s = res["summary"]
        print(
            f"   thr={s['throughput_rps']:.3f}/s e2e_mean={s['e2e']['mean']:.3f}s "
            f"p95={s['e2e']['p95']:.3f}s errors={s['n_errors']} leak_ok={s['leak']['ok']}",
            flush=True,
        )
        results[cond["cond_id"]] = res
    summary = {
        "experiment_mode": emode,
        "claim_scope": CLAIM_SCOPE[emode],
        "deviations": cfg.get("deviations"),
        "conditions": {k: v["summary"] for k, v in results.items()},
        "identity": identity_check({k: v["rows"] for k, v in results.items()}),
        "reference": reference_check(
            results, ref_a, reference_preds(a.reference_preds) if a.reference_preds else None, items
        ),
    }
    write_new(
        run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"saved -> {run_dir}")
    return run_dir


def vllm_launch_view(path: str | None) -> dict | None:
    if not path:
        return None
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    srv = (m.get("profile") or {}).get("server") or {}
    return {
        "path": str(path),
        "profile": (m.get("profile") or {}).get("name"),
        "profile_sha256": (m.get("profile") or {}).get("sha256"),
        "overrides": m.get("overrides"),
        "gpu_memory_utilization": srv.get("gpu_memory_utilization"),
        "max_num_seqs": srv.get("max_num_seqs"),
        "tensor_parallel_size": srv.get("tensor_parallel_size"),
        "dtype": srv.get("dtype"),
        "smoke_enforce_eager": m.get("smoke"),
        "cuda_visible_devices": m.get("cuda_visible_devices"),
        "argv": m.get("argv"),
    }


def reference_check(results: dict, ref_a: dict | None, ref_p: dict | None, items=None) -> dict:
    """MODE A 결과와 대조: 판정 (score run), 예측·정확도 (generate run, 같은 문항)."""
    from rag.crag_official.generation import official_match

    out = {}
    for cid, res in results.items():
        rows = [r for r in res["rows"] if not r.get("error")]
        d: dict = {"n": len(rows)}
        if ref_a:
            diff = [r["index"] for r in rows if ref_a[r["index"]]["action"] != r["action"]]
            sd = [
                max(
                    abs(x - y)
                    for x, y in zip(r["scores"], ref_a[r["index"]]["scores"], strict=True)
                )
                for r in rows
            ]
            d.update(
                action_mismatch=len(diff),
                action_mismatch_indices=diff[:20],
                max_score_abs_diff=max(sd) if sd else None,
            )
        if ref_p:
            pm = [
                r["index"]
                for r in rows
                if r["index"] in ref_p and ref_p[r["index"]]["prediction"] != r["prediction"]
            ]
            d.update(
                n_with_ref_pred=sum(r["index"] in ref_p for r in rows),
                prediction_mismatch=len(pm),
                prediction_mismatch_indices=pm[:20],
            )
            both = [r for r in rows if r["index"] in ref_p and r.get("official_match") is not None]
            if both and items is not None:
                d["official_match_this"] = sum(r["official_match"] for r in both) / len(both)
                d["official_match_mode_a"] = sum(
                    official_match(ref_p[r["index"]]["prediction"], items[r["index"]].answers)
                    for r in both
                ) / len(both)
                d["n_accuracy_compared"] = len(both)
        out[cid] = d
    return out


if __name__ == "__main__":
    main()
