"""동일 추정 유효 부하 (equal effective load) live 실험 준비·검증·보고. GPU 를 쓰지 않는다.

기존 EVC live 는 crag_rule 과 keep_all 에 같은 도착 trace (ρ = λ / μ_max, μ_max 는 crag_rule 포화
처리량) 를 썼다. keep_all 의 요청당 무부하 LLM 작업량이 crag_rule 의 F 배이면 추정 유효 부하는
ρ x F 이다. 여기서는 정책별 도착률을 λ' = λ x target / (ρ x F) 로 낮춰 같은 목표 유효 부하를 만든다.

workload 변환 = 기존 workload 의 도착 시각에 s = ρ x F / target 을 곱한다 (상대 간격 비율 유지).
workload.make_workload 의 Poisson 도착은 seed 의 표준 지수분포 난수 / λ 의 누적합이므로, 같은
seed·질문 순서에서 rate = λ / s 로 새로 만든 것과 같다 (check 단계에서 다시 생성해 대조).
질문 순서·query_id·seed·요청 수는 바꾸지 않는다.

    # 1) workload 생성 (F 는 diag_evc_live analysis.json 의 work_ratio_vs_crag_rule 또는 직접 지정)
    uv run python experiments/crag_system/equal_load.py make \
        --source results/crag_system/workloads/SV1_rho0.8_s0.json \
        --source results/crag_system/workloads/SV1_rho0.8_s1.json \
        --policy keep_all --work-ratio 1.513 --target-rho-eff 0.8 --tag EQL1
    # 2) 검증 (기존 workload 와 질문·순서 동일, 도착 시각만 s 배, run 폴더 충돌 없음)
    uv run python experiments/crag_system/equal_load.py check --workload <생성 파일> ... \
        --live-root results/crag_system/pre/PRE-EQL1-LIVE --k 4 \
        [--seq results/crag_system/pre/PRE-SV1-A/eval_seq] [--evf <evf run>]
    # 3) live 는 run_equal_load.sh live (사용자가 서버에서)
    # 4) 보고
    uv run python experiments/crag_system/equal_load.py report \
        --new-root results/crag_system/pre/PRE-EQL1-LIVE \
        --ref-root results/crag_system/pre/PRE-EVC-SV1-LIVE --seq ... [--evf ...] [--evc-seq ...]

유효 부하는 무부하 작업시간 기반 추정이다 (no-load 초 단위 처리 용량이 정책 간 같다는 가정).
같은 추정 유효 부하가 같은 GPU 자원 점유 (prefill / decode 비율, KV cache, batching 효율) 를
보장하지 않는다. 결과: results/crag_system/analysis/equal_load/<name>/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_evc_live import (  # noqa: E402
    BASE,
    decompose,
    discover,
    fit_noload,
    load_evc,
    load_seq_rows,
    queueing,
)
from diag_sched import mean, pct, read_json, read_jsonl  # noqa: E402
from diag_sv_policy import load_run, md_table, write_csv  # noqa: E402
from evidence_cost import _cfg_policy  # noqa: E402
from workload import load_workload, make_workload, requests_sha, save_workload  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WL_DIR = ROOT / "results" / "crag_system" / "workloads"


# ================================================================ make
def ratio_from_diag(path: Path, policy: str, rho: str | None) -> tuple[float, dict]:
    """diag_evc_live analysis.json -> 정책의 work_ratio_vs_crag_rule (run 평균)."""
    res = read_json(path)
    rows = [q for q in res.get("queueing") or [] if q.get("policy") == policy
            and q.get("work_ratio_vs_crag_rule") is not None
            and (rho is None or str(q.get("rho")) == str(rho))]  # fmt: skip
    if not rows:
        raise SystemExit(f"{path}: {policy} 의 work_ratio_vs_crag_rule 없음")
    v = [float(q["work_ratio_vs_crag_rule"]) for q in rows]
    return float(np.mean(v)), {"diag": str(path), "runs": [f"rho{q['rho']}_s{q['seed']}"
                                                          for q in rows], "values": v}  # fmt: skip


def run_id(wl: dict, policy: str, k: int) -> str:
    """diag_evc_live.LIVE_DIR 형식. rho 는 명목 ρ (= λ' / μ_max) 3자리."""
    p = wl["params"]
    return f"rho{p['rho']:.3f}_s{p['seed']}_{policy}_K{k}"


def scaled(src: dict, src_path: Path, s: float, policy: str, f: float, f_src: dict,
           target: float, name: str) -> dict:  # fmt: skip
    p = src["params"]
    if p.get("arrival") != "poisson" or p.get("rate") is None:
        raise SystemExit(f"{src_path}: poisson workload 가 아님")
    reqs = [{**r, "arrival": r["arrival"] * s} for r in src["requests"]]
    lam = p["rate"] / s
    rho = p["rho"] / s if p.get("rho") is not None else None
    return {
        "version": src.get("version", 1),
        "name": name,
        "created_utc": datetime.now().astimezone().isoformat(),
        "params": {**p, "rate": lam, "rho": rho, "rate_basis": "rho_capacity_work_scaled"},
        "query_pool": list(src["query_pool"]),
        "requests": reqs,
        "sha256": requests_sha(reqs),
        "source": src.get("source"),
        "equal_load": {
            "source_workload": str(src_path), "source_name": src.get("name"),
            "source_sha256": src["sha256"], "source_rate": p["rate"], "source_rho": p.get("rho"),
            "policy": policy, "work_ratio": f, "work_ratio_source": f_src,
            "target_rho_eff": target, "arrival_scale": s,
            "rho_eff_est": (rho or 0) * f,
            "note": ("유효 부하 = 명목 rho x 무부하 작업량 비 (추정). "
                     "같은 GPU 자원 점유를 보장하지 않음"),
        },
    }  # fmt: skip


def cmd_make(a) -> list[Path]:
    if (a.work_ratio is None) == (a.from_diag is None):
        raise SystemExit("--work-ratio 또는 --from-diag 중 하나")
    if a.work_ratio is not None:
        f, f_src = float(a.work_ratio), {"given": float(a.work_ratio)}
    else:
        f, f_src = ratio_from_diag(Path(a.from_diag), a.policy, a.diag_rho)
    if f <= 0:
        raise SystemExit("work ratio > 0")
    out = []
    for sp in a.source:
        sp = Path(sp)
        src = load_workload(sp)
        s = src["params"]["rho"] * f / a.target_rho_eff
        name = f"{a.tag}_{a.policy}_rhoeff{a.target_rho_eff:g}_s{src['params']['seed']}"
        wl = scaled(src, sp, s, a.policy, f, f_src, a.target_rho_eff, name)
        path = Path(a.out_dir) / f"{name}.json"
        if path.exists():  # 덮어쓰지 않음. 같은 변환이면 그대로 사용
            old = load_workload(path)
            same = (old["sha256"] == wl["sha256"]
                    and old["equal_load"]["source_sha256"] == src["sha256"])  # fmt: skip
            if not same:
                raise SystemExit(f"이미 있고 내용이 다름 (TAG 를 바꿀 것): {path}")
            print(f"이미 있음 (같은 내용, 그대로 사용): {path}")
        else:
            save_workload(wl, path)
            print(f"saved -> {path} (scale {s:.4f}, rate {wl['params']['rate']:.4f}/s, "
                  f"rho {wl['params']['rho']:.4f}, "
                  f"run_id {run_id(wl, a.policy, a.k)})")  # fmt: skip
        out.append(path)
    return out


# ================================================================ check
def check_one(path: Path, a) -> dict:
    wl = load_workload(path)  # sha256 검사 포함
    eq = wl.get("equal_load") or {}
    errs, info = [], {"workload": str(path), "name": wl.get("name")}
    src_path = Path(eq.get("source_workload", ""))
    if not src_path.is_file():
        return {**info, "ok": False, "errors": [f"원본 workload 없음: {src_path}"]}
    src = load_workload(src_path)
    if src["sha256"] != eq.get("source_sha256"):
        errs.append("원본 workload 의 sha256 이 기록과 다름 (원본이 바뀜)")
    r0, r1 = src["requests"], wl["requests"]
    s = float(eq.get("arrival_scale", 0))
    same_q = [x["query_id"] for x in r0] == [x["query_id"] for x in r1]
    same_i = [x["i"] for x in r0] == [x["i"] for x in r1]
    ratio = np.array([b["arrival"] / x["arrival"] for x, b in zip(r0, r1, strict=True)
                      if x["arrival"]]) if len(r0) == len(r1) else np.array([])  # fmt: skip
    rel = float(np.max(np.abs(ratio / s - 1))) if len(ratio) and s else None
    p0, p1 = src["params"], wl["params"]
    keep = {k: (p0.get(k), p1.get(k)) for k in ("n_requests", "seed", "arrival", "sample",
                                                 "capacity_rps")}  # fmt: skip
    if not (same_q and same_i):
        errs.append("질문 순서 / 요청 번호가 원본과 다름")
    if rel is None or rel > 1e-9:
        errs.append(f"도착 시각 비율이 일정하지 않음 (max rel dev {rel})")
    errs += [f"params.{k} 다름: {v}" for k, v in keep.items() if v[0] != v[1]]
    if wl["query_pool"] != src["query_pool"]:
        errs.append("query_pool 다름")
    # 같은 seed·질문 순서에서 rate = λ' 로 새로 생성한 workload 와 같은지
    regen = make_workload(src["query_pool"], p0["n_requests"], seed=p0["seed"],
                          arrival=p0["arrival"], sample=p0["sample"], rate=p1["rate"])  # fmt: skip
    regen_q = [x["query_id"] for x in regen["requests"]] == [x["query_id"] for x in r1]
    regen_t = np.allclose([x["arrival"] for x in regen["requests"]],
                          [x["arrival"] for x in r1], rtol=1e-9, atol=1e-9)  # fmt: skip
    src_regen_t = np.allclose(
        [x["arrival"] for x in make_workload(src["query_pool"], p0["n_requests"],
                                             seed=p0["seed"], arrival=p0["arrival"],
                                             sample=p0["sample"], rate=p0["rate"])["requests"]],
        [x["arrival"] for x in r0], rtol=1e-9, atol=1e-9)  # fmt: skip
    if not regen_q:
        errs.append("같은 seed 로 다시 생성한 질문 순서가 다름")
    if src_regen_t and not regen_t:
        errs.append("rate=λ' 로 다시 생성한 도착 시각과 다름")
    arr = np.array([x["arrival"] for x in r1])
    gaps = np.diff(np.concatenate([[0.0], arr]))
    info.update(
        n_requests=len(r1), same_query_order=same_q, arrival_scale=s, max_rel_dev=rel,
        regenerated_same_order=regen_q, regenerated_same_arrivals=bool(regen_t),
        source_regenerates=bool(src_regen_t), rate_nominal=p1["rate"], rho_nominal=p1.get("rho"),
        rate_empirical=float((len(arr) - 1) / (arr[-1] - arr[0])) if len(arr) > 1 else None,
        mean_gap_s=float(gaps.mean()), arrival_span_s=float(arr[-1] - arr[0]),
        source_arrival_span_s=float(r0[-1]["arrival"] - r0[0]["arrival"]),
        work_ratio=eq.get("work_ratio"), target_rho_eff=eq.get("target_rho_eff"),
        rho_eff_est=eq.get("rho_eff_est"), policy=eq.get("policy"),
    )  # fmt: skip
    if not src_regen_t:
        info["note"] = (
            "원본 workload 가 현재 make_workload 로 재생성되지 않음 (numpy 버전 등) -> 비율 검사만"
        )
    if a.live_root:
        rid = run_id(wl, eq.get("policy"), a.k)
        d = Path(a.live_root) / rid
        info["run_id"] = rid
        info["run_dir_exists"] = d.exists()
        info["gpu_log_exists"] = Path(str(d) + ".gpu.csv").exists()
        if d.exists() and not (d / "summary.json").exists():
            errs.append(f"중단된 run 폴더가 있음 (덮어쓰지 않음): {d}")
    if a.seq:
        rows, _, _ = load_seq_rows(Path(a.seq))
        qs = [x["query_id"] for x in r1]
        corr = [q for q in qs if q in rows and rows[q].get("corrective_action") != "correct"]
        info["not_in_seq"] = sum(q not in rows for q in qs)
        info["seq_corrected_no_rewrite_record"] = sum(
            1
            for q in corr
            if not ((rows[q].get("extra") or {}).get("rewritten_query") or "").strip()
        )
    if a.evf:
        items = {it["query_id"]: it for it in read_jsonl(Path(a.evf) / "items.jsonl")}
        info["evf_excluded_in_workload"] = sum(1 for x in r1 if items.get(x["query_id"], {})
                                               .get("problems"))  # fmt: skip
    info["ok"] = not errs
    info["errors"] = errs
    return info


def cmd_check(a) -> int:
    res = [check_one(Path(p), a) for p in a.workload]
    if a.live_root:
        ids = [r.get("run_id") for r in res]
        if len(set(ids)) != len(ids):
            for r in res:
                r["errors"].append("run_id 충돌 (같은 run_id 가 여러 workload 에서 나옴)")
                r["ok"] = False
    print(json.dumps(res, ensure_ascii=False, indent=1))
    bad = [r for r in res if not r["ok"]]
    print("check:", "OK" if not bad else f"실패 {len(bad)}개")
    return 1 if bad else 0


# ================================================================ report
def boot_ci(x: np.ndarray, y: np.ndarray, f, n_boot: int, seed: int = 0) -> tuple:
    """같은 요청 쌍 (paired) bootstrap: f(y) - f(x) 의 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return None, None
    idx = rng.integers(0, n, (n_boot, n))
    d = np.array([f(y[i]) - f(x[i]) for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_summary(label: str, cond: str, key, run: dict, d: Path, dec: list, k: int) -> dict:
    q, _ = queueing(key, run, dec, k, 1, d)
    lat = [r["latency"] for r in run["rows"].values()]
    wl = run["meta"].get("workload") or {}
    p = wl.get("params") or {}
    return {
        "label": label, "condition": cond, "rho_nominal": key[0], "seed": key[1],
        "policy": key[2], "K": key[3], "n": len(lat), "n_errors": run["n_errors"],
        "acc": mean([r.get("acc") for r in run["rows"].values()]),
        "e2e_mean": mean(lat), "e2e_p50": pct(lat, 50), "e2e_p95": pct(lat, 95),
        "e2e_p99": pct(lat, 99),
        "gate_wait_mean": mean([x["gate_wait"] for x in dec]),
        "exec_mean": mean([x["exec"] for x in dec]),
        "wait_only_mean": mean([x["wait_only"] for x in dec]),
        "noload_mean": mean([x["noload"] for x in dec]),
        "noload_exact_share": mean([x["noload_exact_share"] for x in dec]),
        "lambda_nominal": p.get("rate"), "lambda_obs": q["lambda_obs"],
        "throughput": q["throughput"], "thr_over_lambda": q["thr_over_lambda"],
        "N_mean": q["N_mean"], "gate_waiting_mean": q["gate_waiting_mean"],
        "running_mean": q["running_mean"], "slots_full_share": q["slots_full_share"],
        "drain_after_last_arrival_s": q["drain_after_last_arrival_s"],
        "lat_slope_per_s": q["lat_slope_per_s"], "vllm_waiting_mean": q["vllm_waiting_mean"],
        "kv_usage_mean": q["kv_usage_mean"], "gpu_util_mean": q.get("gpu_util_mean"),
        "capacity_rps": p.get("capacity_rps"), "workload_sha256": wl.get("sha256"),
        "workload_name": wl.get("name"),
    }  # fmt: skip


def cmd_report(a) -> Path:
    seq_rows, seq_calls, _ = load_seq_rows(Path(a.seq))
    evc = load_evc(Path(a.evc_seq) if a.evc_seq else None)
    model = fit_noload(seq_calls, evc)
    runs, warns = {}, []
    for label, root in (("new", a.new_root), ("ref", a.ref_root)):
        for key, d in discover(Path(root)).items():
            if key[3] != a.k:
                continue
            if label == "ref" and key[0] != a.ref_rho:
                continue
            cp = _cfg_policy(d)
            if cp != key[2]:
                warns.append(f"{d.name}: config evidence_policy={cp!r} != 폴더 {key[2]!r}")
            run = load_run(d)
            if run["policy"] != a.sched_policy:
                warns.append(f"{d.name}: scheduler={run['policy']} (기대 {a.sched_policy})")
            runs[(label, *key)] = (run, d)
    if not any(k[0] == "new" for k in runs):
        raise SystemExit(f"새 run 없음: {a.new_root}")
    summ, decs = [], {}
    for (label, rho, seed, pol, k), (run, d) in sorted(runs.items(), key=str):
        dec = decompose(run, pol, seq_calls, seq_rows, evc, model)
        decs[(label, rho, seed, pol)] = (run, dec)
        cond = ("effective-equal (추정)" if label == "new"
                else f"nominal-equal rho{rho}")  # fmt: skip
        summ.append(run_summary(label, cond, (rho, seed, pol, k), run, d, dec, k))
    # 추정 유효 부하: λ_obs x E[noload] / (μ_max x E[noload | crag_rule, ref 같은 seed])
    base = {r["seed"]: r for r in summ if r["label"] == "ref" and r["policy"] == BASE}
    for r in summ:
        b = base.get(r["seed"])
        mu = r["capacity_rps"]
        r["rho_nominal_obs"] = r["lambda_obs"] / mu if mu and r["lambda_obs"] else None
        if b and mu and b["noload_mean"]:
            r["work_ratio_vs_ref_crag_rule"] = r["noload_mean"] / b["noload_mean"]
            r["rho_eff_obs_est"] = r["rho_nominal_obs"] * r["work_ratio_vs_ref_crag_rule"]
        else:
            r["work_ratio_vs_ref_crag_rule"] = r["rho_eff_obs_est"] = None
        r["workload_is_equal_load"] = bool(r["workload_name"] and "rhoeff" in r["workload_name"])
    # 같은 요청 쌍 비교 (같은 seed: 질문 순서가 같으므로 request_key 가 같다)
    pairs = []
    for (label, rho, seed, pol), (run, _) in decs.items():
        if label != "new":
            continue
        for ref_pol in (BASE, pol):
            ref = next((v for k, v in decs.items() if k[0] == "ref" and k[2] == seed
                        and k[3] == ref_pol), None)  # fmt: skip
            if ref is None:
                continue
            rr = ref[0]["rows"]
            ks = sorted(set(run["rows"]) & set(rr))
            x = np.array([rr[k]["latency"] for k in ks])
            y = np.array([run["rows"][k]["latency"] for k in ks])
            lo, hi = boot_ci(x, y, np.mean, a.n_boot)
            plo, phi = boot_ci(x, y, lambda v: np.percentile(v, 95), a.n_boot)
            pairs.append({"seed": seed, "new": f"{pol}@rho{rho}",
                          "ref": f"{ref_pol}@rho{a.ref_rho}",
                          "n_pairs": len(ks), "d_mean": float(y.mean() - x.mean()) if ks else None,
                          "d_mean_lo": lo, "d_mean_hi": hi,
                          "d_p95": float(np.percentile(y, 95) - np.percentile(x, 95)) if ks
                          else None, "d_p95_lo": plo, "d_p95_hi": phi,
                          "acc_new": mean([run["rows"][k].get("acc") for k in ks]),
                          "acc_ref": mean([rr[k].get("acc") for k in ks])})  # fmt: skip
    res = {"inputs": vars(a), "warnings": warns, "noload_model": model, "runs": summ,
           "pairs": pairs}  # fmt: skip
    out = new_out_dir(Path(a.out_root), a.name)
    (out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                                       encoding="utf-8")  # fmt: skip
    write_csv(out / "runs.csv", summ)
    write_csv(out / "pairs.csv", pairs)
    md = report_md(res, a.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    if not a.quiet:
        print(md)
    print(f"saved -> {out}")
    return out


def report_md(res: dict, name: str) -> str:
    L = [f"# 동일 추정 유효 부하 live 비교 ({name})", "",
         "- new: 정책별로 도착률을 낮춘 run (추정 유효 부하 = 목표). "
         "ref: 기존 EVC live (같은 명목 ρ, 같은 도착 trace)",
         "- 유효 부하는 무부하 작업시간 기반 추정 "
         "(no-load 초 단위 처리 용량이 정책 간 같다는 가정). "
         "같은 추정 유효 부하가 같은 GPU 자원 점유를 보장하지 않는다",
         f"- no-load 모델: {json.dumps(res['noload_model'])}", ""]  # fmt: skip
    if res["warnings"]:
        L += ["## 경고", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    L += ["## run 별", ""]
    L += md_table(res["runs"], ["label", "condition", "rho_nominal", "seed", "policy", "n",
                                "n_errors", "acc", "e2e_mean", "e2e_p50", "e2e_p95", "e2e_p99",
                                "gate_wait_mean", "exec_mean", "noload_mean",
                                "noload_exact_share", "lambda_nominal", "lambda_obs",
                                "throughput", "thr_over_lambda", "N_mean",
                                "drain_after_last_arrival_s", "lat_slope_per_s",
                                "rho_nominal_obs", "work_ratio_vs_ref_crag_rule",
                                "rho_eff_obs_est", "gpu_util_mean"])  # fmt: skip
    L += ["", "- rho_eff_obs_est = (λ_obs / μ_max) x "
          "(E[noload] / E[noload | ref crag_rule, 같은 seed])"
          " [추정]. noload_exact_share < 1 이면 일부 호출의 무부하 시간이 토큰 모델 [추정]",
          "", "## 같은 요청 쌍 비교 (new - ref, paired bootstrap 95% CI)", ""]  # fmt: skip
    L += md_table(res["pairs"], ["seed", "new", "ref", "n_pairs", "d_mean", "d_mean_lo",
                                 "d_mean_hi", "d_p95", "d_p95_lo", "d_p95_hi", "acc_new",
                                 "acc_ref"])  # fmt: skip
    L += ["", "- new keep_all vs ref crag_rule: 같은 추정 유효 부하에서의 정책 비용 차이 "
          "(단, 도착 간격이 달라 같은 요청의 경쟁 상대가 다르다)",
          "- new keep_all vs ref keep_all: 같은 정책에서 부하만 바꾼 효과 "
          "(과부하 가설 H2 의 직접 검증)",
          ""]  # fmt: skip
    return "\n".join(L)


def new_out_dir(root: Path, name: str) -> Path:
    base = root / "equal_load" / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--source", action="append", required=True, help="기존 workload json")
    m.add_argument("--policy", default="keep_all")
    m.add_argument("--work-ratio", type=float, default=None)
    m.add_argument("--from-diag", default=None, help="diag_evc_live analysis.json")
    m.add_argument("--diag-rho", default=None, help="--from-diag 에서 이 rho 의 run 만 평균")
    m.add_argument("--target-rho-eff", type=float, default=0.8)
    m.add_argument("--tag", default="EQL1")
    m.add_argument("--k", type=int, default=4)
    m.add_argument("--out-dir", default=str(WL_DIR))
    c = sub.add_parser("check")
    c.add_argument("--workload", action="append", required=True)
    c.add_argument("--live-root", default=None)
    c.add_argument("--k", type=int, default=4)
    c.add_argument("--seq", default=None)
    c.add_argument("--evf", default=None)
    r = sub.add_parser("run-id")
    r.add_argument("--workload", required=True)
    r.add_argument("--k", type=int, default=4)
    p = sub.add_parser("report")
    p.add_argument("--new-root", required=True)
    p.add_argument("--ref-root", required=True)
    p.add_argument("--ref-rho", default="0.8")
    p.add_argument("--seq", required=True)
    p.add_argument("--evc-seq", default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--sched-policy", default="fcfs_query")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--name", default="EQL1")
    p.add_argument("--out-root", default=str(ROOT / "results" / "crag_system" / "analysis"))
    p.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        return cmd_make(a)
    if a.cmd == "check":
        rc = cmd_check(a)
        if argv is None:
            sys.exit(rc)
        return rc
    if a.cmd == "run-id":
        wl = load_workload(a.workload)
        print(run_id(wl, wl["equal_load"]["policy"], a.k))
        return None
    return cmd_report(a)


if __name__ == "__main__":
    main()
