"""PRE-Q: 두 엔진 설정 (예: BF16 TP=2 vs AWQ INT4) 의 소규모 품질 비교.

같은 질문·같은 prompt·같은 검색 결과·같은 correction 문서 (web replay) 에서 엔진만 다른 두 run 을
query 단위로 짝지어 비교한다. 소규모 예비 비교이며 비열등성 (ε) 판단에 쓰지 않는다.

    uv run python experiments/crag_system/compare_quality.py <run_A> <run_B> [--out report.json]

전제 (다르면 실패): 평가 질문 순서, query 별 최초 검색 결과, web replay 원본.
비교 항목:
  accuracy / F1, 짝지은 불일치 (A 만 정답 / B 만 정답) 와 McNemar exact p (참고)
  답변 일치율 (정규화 문자열), corrective action 일치율과 교차표
  grader 점수: 같은 문서 쌍의 평균 절대 차이, 완전 일치율, 상관,
    임계값 구간 (>=upper / 중간 / <lower) 일치율
  grade 출력 형식 (run.record_texts=true 일 때): 숫자 하나로만 답한 비율, 숫자를 못 찾은 비율
  latency (E2E, stage), 호출당 토큰
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparability import compare_runs  # noqa: E402

ENGINE_CHECKS = {"llm", "engine_identity"}  # 엔진 비교에서는 달라도 되는 항목
NUM_ONLY = re.compile(r"^\s*\d*\.?\d+\s*\.?\s*$")
ANY_NUM = re.compile(r"\d*\.?\d+")


def _norm(text: str | None) -> str:
    t = (text or "").lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


def _scores(row: dict) -> list:
    g = (row.get("extra") or {}).get("grading") or {}
    return g.get("scores_full") or row.get("grader_scores") or []


def _band(s: float, upper: float, lower: float) -> str:
    return "high" if s >= upper else ("low" if s < lower else "mid")


def _grade_format(run_dir: str) -> dict | None:
    path = Path(run_dir) / "calls.jsonl"
    outs = [
        json.loads(x).get("output_text")
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and json.loads(x).get("kind") == "grade" and not json.loads(x).get("warmup")
    ]
    outs = [o for o in outs if o is not None]
    if not outs:
        return None
    return {
        "n": len(outs),
        "number_only_rate": sum(bool(NUM_ONLY.match(o)) for o in outs) / len(outs),
        "no_number_rate": sum(ANY_NUM.search(o) is None for o in outs) / len(outs),
        "examples_non_numeric": [o[:80] for o in outs if not NUM_ONLY.match(o)][:5],
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compare_quality(run_a: str, run_b: str, upper: float = 0.7, lower: float = 0.3) -> dict:
    base = compare_runs([run_a, run_b])
    blocking = [c for c in base["critical"] if c["check"] not in ENGINE_CHECKS]
    # B 가 A 의 web 기록을 replay 하는 구성 (A: 실제 Tavily 기록, B: replay) 은 허용
    srcs = [((c.get("web_search") or {}).get("sources") or []) for c in base["confounders"]]
    a_root = str(Path(run_a).resolve())
    b_replays_a = (
        not srcs[0]
        and srcs[1]
        and all(str(Path(s["path"]).resolve()).startswith(a_root) for s in srcs[1])
    )
    if b_replays_a:
        blocking = [c for c in blocking if c["check"] != "web_replay_sources"]
    replay_stats = base["confounders"][1].get("web_replay_stats")
    rows = []
    for d in (run_a, run_b):
        rows.append(
            {
                r["request_index"]: r
                for r in (
                    json.loads(x)
                    for x in (Path(d) / "queries.jsonl").read_text(encoding="utf-8").splitlines()
                    if x.strip()
                )
                if not r.get("warmup") and not r.get("error")
            }
        )
    common = sorted(set(rows[0]) & set(rows[1]))
    pa, pb = [rows[0][i] for i in common], [rows[1][i] for i in common]
    both = sum(1 for a, b in zip(pa, pb, strict=True) if a["acc"] and b["acc"])
    a_only = sum(1 for a, b in zip(pa, pb, strict=True) if a["acc"] and not b["acc"])
    b_only = sum(1 for a, b in zip(pa, pb, strict=True) if b["acc"] and not a["acc"])
    actions: dict = {}
    for a, b in zip(pa, pb, strict=True):
        key = f"{a.get('corrective_action')}->{b.get('corrective_action')}"
        actions[key] = actions.get(key, 0) + 1
    xs, ys = [], []
    for a, b in zip(pa, pb, strict=True):
        for sa, sb in zip(_scores(a), _scores(b), strict=False):
            if sa is not None and sb is not None:
                xs.append(sa)
                ys.append(sb)
    n = len(common)
    report = {
        "purpose": "pre",
        "runs": [run_a, run_b],
        "prerequisites_ok": not blocking,
        "prerequisite_failures": blocking,
        "engine_differences": [c for c in base["critical"] if c["check"] in ENGINE_CHECKS],
        "web_replay": {
            "b_replays_a": bool(b_replays_a),
            "b_stats": replay_stats,  # misses = B 의 rewrite 가 A 와 달라 실제 검색으로 대체된 수
            "note": "misses 가 있으면 그 query 의 correction 문서는 A 와 다를 수 있음",
        },
        "n_paired": n,
        "accuracy": {
            "a": both / n + a_only / n if n else None,
            "b": both / n + b_only / n if n else None,
            "both_correct": both,
            "a_only": a_only,
            "b_only": b_only,
            "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
        },
        "f1": {"a": _mean([r["f1"] for r in pa]), "b": _mean([r["f1"] for r in pb])},
        "answer_agreement": (
            sum(_norm(a["answer"]) == _norm(b["answer"]) for a, b in zip(pa, pb, strict=True)) / n
        )
        if n
        else None,
        "action_agreement": (
            sum(
                a.get("corrective_action") == b.get("corrective_action")
                for a, b in zip(pa, pb, strict=True)
            )
            / n
        )
        if n
        else None,
        "action_transitions": actions,
        "grade_scores": {
            "n_pairs": len(xs),
            "mean_abs_diff": _mean([abs(x - y) for x, y in zip(xs, ys, strict=True)]),
            "exact_equal_rate": _mean([float(x == y) for x, y in zip(xs, ys, strict=True)]),
            "pearson_r": _pearson(xs, ys),
            "band_agreement": _mean(
                [
                    float(_band(x, upper, lower) == _band(y, upper, lower))
                    for x, y in zip(xs, ys, strict=True)
                ]
            ),
        },
        "grade_output_format": {"a": _grade_format(run_a), "b": _grade_format(run_b)},
        "latency": {
            "a": _mean([r["latency"] for r in pa]),
            "b": _mean([r["latency"] for r in pb]),
            "stage_mean_a": {
                s: _mean([r["stage_latencies"].get(s) for r in pa])
                for s in ("grading", "rewrite", "generation")
            },
            "stage_mean_b": {
                s: _mean([r["stage_latencies"].get(s) for r in pb])
                for s in ("grading", "rewrite", "generation")
            },
        },
        "note": "소규모 예비 비교. 모델 정밀도 선택의 참고 자료이며 비열등성 판단에 쓰지 않음",
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    report = compare_quality(a.run_a, a.run_b)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0 if report["prerequisites_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
