"""run.py 결과 폴더의 정합성 판정 (요청 3절). 하나라도 FAIL 이면 성능 실험 전에 원인부터 해결한다.

    uv run python experiments/c2_gating/smoke_check.py <run 폴더>
    (run 폴더 = results/c2_gating/<experiment_mode>/smoke/<run_id>)
-> <run>/smoke_report.md, smoke_report.json, 종료 코드 (0 = 모두 통과)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ACTIONS = ("correct", "ambiguous", "incorrect")


def check(run_dir: Path) -> dict:
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    conds = s["conditions"]
    emode = s.get("experiment_mode", "controlled_replay")
    res: list[dict] = []

    def add(name, ok, detail):
        res.append({"check": name, "ok": bool(ok), "detail": detail})

    for cid, c in conds.items():
        missing = [a for a in ACTIONS if not c["action_counts"].get(a)]
        add(
            f"[{cid}] 분기별 정상 동작 (오류 0, 세 판정 모두 실행)",
            c["n_errors"] == 0 and not missing,
            {"errors": c["errors"], "action_counts": c["action_counts"], "missing": missing},
        )
    ident = s["identity"]
    for k in ("action", "evidence_id", "evidence_sha16", "prompt_sha16"):
        add(
            f"B/C/Cbn {k} 일치",
            ident["all_identical"][k],
            {"n_mismatch": ident["n_mismatch"][k], "examples": ident["examples"].get(k)},
        )
    add(
        "(참고) prediction 일치 — evidence/prompt 같아도 동시 요청 batching 으로 다를 수 있음",
        True,
        {
            "n_mismatch": ident["n_mismatch"]["prediction"],
            "examples": ident["examples"].get("prediction"),
        },
    )
    for cid, r in (s.get("reference") or {}).items():
        if "action_mismatch" in r:
            add(f"[{cid}] MODE A score run 과 판정 일치", r["action_mismatch"] == 0, r)
        if "prediction_mismatch" in r:
            add(f"[{cid}] (참고) MODE A generate 예측과 일치 수", True, r)
        if "official_match_mode_a" in r:
            name = (
                f"[{cid}] MODE A 와 같은 문항 official_match 일치 "
                "(Controlled Replay: evidence 고정 하의 "
                "생성 재현성 확인. E2E 정확도 보존 증거 아님)"
                if emode == "controlled_replay"
                else f"[{cid}] (참고) MODE A (저자 지식) 대비 official_match "
                "— runtime evidence 는 다름"
            )
            add(
                name,
                emode != "controlled_replay"
                or r["official_match_this"] == r["official_match_mode_a"],
                {
                    k: r[k]
                    for k in (
                        "official_match_this",
                        "official_match_mode_a",
                        "n_accuracy_compared",
                        "prediction_mismatch",
                    )
                },
            )
    for cid, c in conds.items():
        mode = c["cond"]["mode"]
        ce = c["correct_external_refine"]
        if mode in ("B", "Cbn"):
            add(
                f"[{cid}] Correct 요청의 external refinement T5 호출 0",
                ce["n_jobs_submitted"] == 0 and ce["n_pairs_forwarded"] == 0,
                ce,
            )
            add(
                f"[{cid}] 판정 전 external refinement submit 0",
                c["external_submit_before_decision"] == 0,
                c["external_submit_before_decision"],
            )
        else:
            add(f"[{cid}] (참고) C 의 Correct external refinement", True, ce)
        if mode in ("C", "Cbn"):
            ov = c["overlap"]
            add(
                f"[{cid}] grading 과 correction 경로 실제 중첩",
                (ov["stage_overlap_frac"] or 0) > 0 and (ov["exec_overlap_frac"] or 0) > 0,
                ov,
            )
        gp = c.get("generation_precondition") or {}
        add(
            f"[{cid}] 필요한 correction 작업 완료 후 generation 시작 (전 요청)",
            gp.get("n_checked") == c["n_ok"] and gp.get("n_violations") == 0,
            gp,
        )
        add(f"[{cid}] cancellation 이후 task / queue 누수 없음", c["leak"]["ok"], c["leak"])
        t5 = c["t5"]
        svc = c.get("t5_service") or {}
        info = {k: t5[k] for k in ("memo_hits", "memo_mismatch", "max_memo_abs_diff")}
        if svc.get("memo_verify"):
            add(f"[{cid}] memo 결과 비트 단위 동일 (memo_verify)", t5["memo_mismatch"] == 0, info)
        elif t5["memo_hits"]:
            add(
                f"[{cid}] (참고) memo 사용, 이 run 에서는 미검증 — "
                "check_memo.py / memo_verify run 결과로 판단",
                True,
                info,
            )
    acc = {cid: c["accuracy"] for cid, c in conds.items()}
    return {
        "run": str(run_dir),
        "experiment_mode": emode,
        "claim_scope": s.get("claim_scope"),
        "all_ok": all(r["ok"] for r in res),
        "checks": res,
        "accuracy": acc,
    }


def main(argv=None) -> int:
    run_dir = Path((argv or sys.argv[1:])[0])
    rep = check(run_dir)
    lines = [
        f"# Smoke check — {run_dir.name}",
        "",
        f"**all_ok: {rep['all_ok']}**",
        "",
        f"experiment_mode: `{rep['experiment_mode']}` — {rep['claim_scope']}",
        "",
        "| 결과 | 항목 |",
        "|---|---|",
    ]
    for r in rep["checks"]:
        lines.append(f"| {'PASS' if r['ok'] else '**FAIL**'} | {r['check']} |")
    lines += [
        "",
        "## 정확도 (조건별)",
        "",
        "| 조건 | official_match | acc | n |",
        "|---|---|---|---|",
    ]
    for cid, a in rep["accuracy"].items():
        lines.append(f"| {cid} | {a['official_match']} | {a['acc']} | {a['n_scored']} |")
    lines += ["", "## 실패 항목 상세", ""]
    for r in rep["checks"]:
        if not r["ok"]:
            lines.append(f"- {r['check']}: `{json.dumps(r['detail'], ensure_ascii=False)[:1500]}`")
    (run_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "smoke_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("\n".join(lines[:200]))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
