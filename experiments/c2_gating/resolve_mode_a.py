"""MODE A 기준 결과 확인 (서버, 읽기 전용) + 재검증용 prediction 비교.

resolve        score / rag / crag run 과 MODE A 설정이 서로 맞는지, 정답 파일이 무엇이었는지 찾고
               official_match 를 다시 계산해 기대값 (rag 705, crag 769 / 1399) 과 비교한다.
               정답 파일: (1) --answers, (2) 해당 preds 를 평가한 evaluate run 의 config.yaml,
               (3) data/ 아래 popqa_longtail_w_gs.jsonl. 기대값과 맞는 파일만 채택.
               -> results/c2_gating/mode_a_ref/<run_id>/{ref.json, ref.env}
compare-preds  generate run 두 개의 문항별 prediction·official_match 비교
               (serving 설정 변경 후 재검증)

  uv run python experiments/c2_gating/resolve_mode_a.py resolve --run-id llama2_tp2_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_data import load_items, read_jsonl  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official.generation import official_match  # noqa: E402
from rag.telemetry.run_meta import create_run_dir, write_new  # noqa: E402

ROOT = HERE.parents[1]
MODE_A = ROOT / "results" / "official_crag" / "mode_a"
DEFAULTS = {
    "mode_a_config": "experiments/official_crag/configs/mode_a_llama2_7b_tp2.yaml",
    "scores": "results/official_crag/mode_a/score/20261126_072612",
    "rag_preds": "results/official_crag/mode_a/generate_rag/20261126_093945",
    "crag_preds": "results/official_crag/mode_a/generate_crag/20261127_023204",
    "prompt_version": "v3_2024_10",
    "expect_rag": 705,
    "expect_crag": 769,
    "n": 1399,
}
LLM_KEYS = ("model", "api", "temperature", "top_p", "max_tokens", "extra_body")


def _same(p: str | None, q: str | Path) -> bool:
    if not p:
        return False
    a, b = Path(p), Path(q)
    a = a if a.is_absolute() else ROOT / a
    b = b if b.is_absolute() else ROOT / b
    return a.resolve() == b.resolve()


def _yaml(p: Path) -> dict:
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def answer_candidates(explicit, preds_dirs) -> list[dict]:
    out = []
    if explicit:
        out.append({"path": explicit, "from": "--answers"})
    for cfg_path in sorted((MODE_A / "evaluate").glob("*/config.yaml")):
        cfg = _yaml(cfg_path)
        if any(_same(cfg.get("preds"), d) for d in preds_dirs) and cfg.get("answers"):
            out.append({"path": cfg["answers"], "from": str(cfg_path.parent)})
    for p in sorted((ROOT / "data").rglob("popqa_longtail_w_gs.jsonl")):
        out.append({"path": str(p), "from": "data/ glob"})
    seen, uniq = set(), []
    for c in out:
        key = str(Path(c["path"]).expanduser())
        if key not in seen:
            seen.add(key)
            uniq.append({**c, "path": key})
    return uniq


def count_match(preds_dir, items) -> tuple[int, int, dict]:
    rows = read_jsonl(Path(preds_dir) / "preds.jsonl")
    m = [official_match(r["prediction"], items[r["index"]].answers) for r in rows]
    by = Counter()
    for r, x in zip(rows, m, strict=True):
        by[(r.get("action"), x)] += 1
    return sum(m), len(rows), {f"{a}:{x}": v for (a, x), v in sorted(by.items(), key=str)}


def cmd_resolve(a) -> dict:
    mode_a_llm = load_layered_config(a.mode_a_config, [], [])["llm"]
    checks, ok = [], True

    def add(name, cond, detail=None):
        nonlocal ok
        ok &= bool(cond)
        checks.append({"check": name, "ok": bool(cond), "detail": detail})

    for name in ("scores", "rag_preds", "crag_preds"):
        d = (
            ROOT / getattr(a, name)
            if not Path(getattr(a, name)).is_absolute()
            else Path(getattr(a, name))
        )
        add(f"{name} 폴더 존재", d.is_dir(), str(d))
    gen = {}
    for name, method in (("rag_preds", "rag"), ("crag_preds", "crag")):
        cfg = _yaml(ROOT / getattr(a, name) / "config.yaml")
        gen[name] = cfg
        llm = cfg.get("llm") or {}
        diff = {
            k: (llm.get(k), mode_a_llm.get(k)) for k in LLM_KEYS if llm.get(k) != mode_a_llm.get(k)
        }
        add(f"{name}: method={method}", cfg.get("method") == method, cfg.get("method"))
        add(f"{name}: llm 설정 == {a.mode_a_config}", not diff, diff)
        add(
            f"{name}: prompt_version={a.prompt_version}",
            (cfg.get("prompt_version") or "v3_2024_10") == a.prompt_version,
            cfg.get("prompt_version"),
        )
    add(
        "crag_preds: scores_run == scores",
        _same(gen["crag_preds"].get("scores_run"), a.scores),
        gen["crag_preds"].get("scores_run"),
    )
    sc = read_jsonl(ROOT / a.scores / "scores.jsonl")
    dist = Counter(r["action"] for r in sc)
    add(f"scores 문항 수 = {DEFAULTS['n']}", len(sc) == DEFAULTS["n"], len(sc))
    crag_rows = read_jsonl(ROOT / a.crag_preds / "preds.jsonl")
    act_diff = sum(1 for r in crag_rows if r.get("action") != sc[r["index"]]["action"])
    add("crag preds 의 action == scores action", act_diff == 0, act_diff)

    chosen, tried = None, []
    for c in answer_candidates(a.answers, [a.rag_preds, a.crag_preds]):
        p = Path(c["path"])
        if not p.is_absolute():
            p = ROOT / p
        info = {**c, "exists": p.is_file()}
        if p.is_file():
            try:
                items, _ = load_items(a.repo, p)
                r = count_match(ROOT / a.rag_preds, items)
                g = count_match(ROOT / a.crag_preds, items)
                info.update(rag=r[:2], crag=g[:2], crag_by_action=g[2])
                if r[0] == a.expect_rag and g[0] == a.expect_crag:
                    chosen = {**info, "path": str(p)}
            except Exception as e:  # noqa: BLE001 - 후보 실패는 기록만
                info["error"] = f"{type(e).__name__}: {e}"[:300]
        tried.append(info)
        if chosen:
            break
    add(
        f"정답 파일로 rag {a.expect_rag} / crag {a.expect_crag} 재현",
        chosen is not None,
        {"tried": tried},
    )
    return {
        "ok": ok,
        "checks": checks,
        "answers": chosen["path"] if chosen else None,
        "answers_source": chosen["from"] if chosen else None,
        "official_match": None if not chosen else {"rag": chosen["rag"], "crag": chosen["crag"]},
        "crag_by_action": chosen.get("crag_by_action") if chosen else None,
        "action_distribution": dict(dist),
        "paths": {k: getattr(a, k) for k in ("mode_a_config", "scores", "rag_preds", "crag_preds")},
        "prompt_version": a.prompt_version,
    }


def cmd_compare_preds(a) -> dict:
    items, _ = load_items(a.repo, a.answers)
    x = {r["index"]: r for r in read_jsonl(Path(a.a) / "preds.jsonl")}
    y = {r["index"]: r for r in read_jsonl(Path(a.b) / "preds.jsonl")}
    common = sorted(set(x) & set(y))
    diff = [i for i in common if x[i]["prediction"] != y[i]["prediction"]]
    mx = sum(official_match(x[i]["prediction"], items[i].answers) for i in common)
    my = sum(official_match(y[i]["prediction"], items[i].answers) for i in common)
    return {
        "a": a.a,
        "b": a.b,
        "n_common": len(common),
        "only_a": len(set(x) - set(y)),
        "only_b": len(set(y) - set(x)),
        "prediction_identical": len(common) - len(diff),
        "prediction_diff": len(diff),
        "diff_indices": diff[:50],
        "official_match": {"a": mx, "b": my},
        "verdict": "identical" if not diff and len(set(x) ^ set(y)) == 0 else "different",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    for k in ("mode_a_config", "scores", "rag_preds", "crag_preds", "prompt_version"):
        r.add_argument("--" + k.replace("_", "-"), dest=k, default=DEFAULTS[k])
    r.add_argument("--expect-rag", type=int, default=DEFAULTS["expect_rag"])
    r.add_argument("--expect-crag", type=int, default=DEFAULTS["expect_crag"])
    r.add_argument("--answers", default=None)
    r.add_argument("--run-id", default=None)
    c = sub.add_parser("compare-preds")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--answers", required=True)
    c.add_argument("--run-id", default=None)
    for p in (r, c):
        p.add_argument("--repo", default=str(ROOT / "third_party" / "CRAG"))
    a = ap.parse_args(argv)
    res = cmd_resolve(a) if a.cmd == "resolve" else cmd_compare_preds(a)
    d = create_run_dir(
        f"c2_gating/{'mode_a_ref' if a.cmd == 'resolve' else 'reverify'}", "results", a.run_id
    )
    write_new(d / "result.json", json.dumps(res, indent=2, ensure_ascii=False))
    if a.cmd == "resolve" and res["answers"]:
        write_new(d / "ref.env", f"export ANSWERS={res['answers']}\n")
    print(json.dumps(res, indent=2, ensure_ascii=False)[:6000])
    print(f"saved -> {d}")
    ok = res["ok"] if a.cmd == "resolve" else res["verdict"] == "identical"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
