"""analyze_grade_rerun: 출력 분류, 파싱 정책별 action, 재현성, 토큰 수 기반 [추정]."""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import analyze_grade_rerun as agr  # noqa: E402


@pytest.mark.parametrize(
    "text, cls",
    [
        ("0", "numeric_zero"),
        ("0.", "numeric_zero"),
        (" 0.0 ", "numeric_zero"),
        ("1", "numeric"),
        ("0.8", "numeric"),
        ("Yes.", "no_digit_yes"),
        ("yes, it is relevant", "no_digit_yes"),
        ("No.", "no_digit_no"),
        ("Relevant.", "no_digit_other"),
        ("", "empty"),
        ("The document mentions 1990.", "number_in_text"),
        ("Score: 0.9", "number_in_text"),
        (None, None),
    ],
)
def test_output_class(text, cls):
    assert agr.output_class(text) == cls


def test_policy_score():
    assert agr.policy_score("current", "Yes.", 0.0) == 0.0
    assert agr.policy_score("missing", "Yes.", 0.0) is None
    assert agr.policy_score("yes_no", "Yes.", 0.0) == 1.0
    assert agr.policy_score("yes_no", "No.", 0.0) == 0.0
    assert agr.policy_score("yes_no", "Relevant.", 0.0) is None
    assert agr.policy_score("yes_no", "0.8", 0.8) == 0.8  # 숫자 출력은 그대로
    assert agr.policy_score("yes_no", None, 0.0) == 0.0  # 원문 없음 -> 기록 점수
    assert agr.decide([None, None], 0.7, 0.3) == "incorrect_all_missing"
    assert agr.decide([None, 0.0, 1.0], 0.7, 0.3) == "correct"


# ---------------------------------------------------------------- run 폴더 fixture
def _write_run(d: Path, rows: list[dict], calls: list[dict], record_texts: bool) -> Path:
    d.mkdir(parents=True)
    cfg = {
        "pipeline": "crag_graph",
        "grading": {"strategy": "full"},
        "llm": {"model": "m"},
        "crag": {"upper_threshold": 0.7, "lower_threshold": 0.3},
        "run": {"record_texts": record_texts},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for name, data in (("queries.jsonl", rows), ("calls.jsonl", calls)):
        (d / name).write_text("".join(json.dumps(x) + "\n" for x in data), encoding="utf-8")
    return d


# 질문별: (기존 점수, 기존 출력 토큰 수, 재실행 원문, 재실행 토큰 수)
CASES = {
    "q1": [(0.0, 3, "Yes.", 3), (0.0, 2, "0", 2), (0.0, 2, "0", 2)],  # 파싱 실패로 incorrect
    "q2": [(1.0, 2, "1", 2), (0.0, 2, "0", 2), (0.0, 3, "Yes.", 3)],  # correct 유지
    "q3": [(0.0, 2, "0", 2), (0.0, 2, "No.", 2), (0.0, 2, "0", 2)],  # 실제 incorrect
}


def _rows_calls(record: bool, texts: bool):
    rows, calls = [], []
    for i, (q, docs) in enumerate(CASES.items()):
        scores = [s for s, *_ in docs]
        action = agr.decide_action(scores, 0.7, 0.3)
        rows.append(
            {
                "request_index": i,
                "query_id": q,
                "warmup": False,
                "corrective_action": action,
                "answer": f"a{q}",
                "acc": 1.0 if q == "q2" else 0.0,
                "contexts": [],
                "extra": {
                    "retrieval_doc_ids": [f"{q}d{k}" for k in range(3)],
                    "grading": {"strategy": "full", "scores_full": scores},
                },
            }
        )
        for k, (_s, ft, text, rt) in enumerate(docs):
            calls.append(
                {
                    "request_index": i,
                    "query_id": q,
                    "warmup": False,
                    "kind": "grade",
                    "idx": k,
                    "out_tokens": rt if texts else ft,
                    "finish_reason": "stop",
                    "output_text": text if texts else None,
                }
            )
    return rows, calls


@pytest.fixture
def world(tmp_path):
    full = _write_run(tmp_path / "full", *_rows_calls(False, False), record_texts=False)
    r1 = _write_run(tmp_path / "r1", *_rows_calls(True, True), record_texts=True)
    van_rows = [
        {"request_index": i, "query_id": q, "warmup": False, "acc": 1.0, "answer": "v"}
        for i, q in enumerate(CASES)
    ]
    van = _write_run(tmp_path / "van", van_rows, [], record_texts=False)
    sel = tmp_path / "sel.csv"
    with open(sel, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "stratum"])
        w.writeheader()
        w.writerows(
            [{"query_id": "q1", "stratum": "target"}, {"query_id": "q3", "stratum": "target"}]
        )
    return tmp_path, full, r1, van, sel


def _csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_end_to_end(world):
    tmp, full, r1, van, sel = world
    out = agr.main(
        [
            "--rerun",
            f"r1={r1}",
            "--rerun",
            f"r2={r1}",
            "--full",
            str(full),
            "--vanilla",
            str(van),
            "--selection",
            str(sel),
            "--out-root",
            str(tmp / "out"),
            "--name",
            "t",
        ]
    )
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["problems"] == []

    docs = _csv(out / "grade_docs.csv")
    assert len(docs) == 18  # 2 rerun x 3 질문 x 3 문서
    assert {d["stratum"] for d in docs if d["query_id"] == "q2"} == {"control"}

    ts = {
        (r["rerun"], r["full_out_tokens"], r["rerun_text"]): int(r["count"])
        for r in _csv(out / "token_strings.csv")
    }
    assert ts[("r1", "3", "Yes.")] == 2
    assert ts[("r1", "2", "0")] == 5 and ts[("r1", "2", "No.")] == 1

    ps = {(r["rerun"], r["group"]): r for r in _csv(out / "parse_summary.csv")}
    allr = ps[("r1", "all")]
    assert int(allr["n_score0"]) == 8 and int(allr["n_score0_parse_failure"]) == 3
    assert int(allr["n_no_digit_yes"]) == 2 and int(allr["n_numeric_zero"]) == 5

    rp = {r["pair"]: r for r in _csv(out / "repro.csv")}
    assert rp["r1 vs full"]["score_same"] == "9" and rp["r1 vs full"]["action_same"] == "3"
    assert rp["r1 vs r2"]["text_same"] == "9"

    aq = {(r["rerun"], r["query_id"]): r for r in _csv(out / "action_by_query.csv")}
    assert aq[("r1", "q1")]["action_current"] == "incorrect"
    assert aq[("r1", "q1")]["action_yes_no"] == "correct"
    assert aq[("r1", "q3")]["action_yes_no"] == "incorrect"
    assert aq[("r1", "q2")]["changed_by_parse_only"] == "False"

    # [추정]: 3 토큰 0점은 전부 Yes. -> 1.0 적용,
    # 2 토큰 0점은 numeric_zero 다수지만 5/6 < 0.9 -> 미적용
    band = meta["extrapolation"]["band_majority"]
    assert band["3"]["majority"] == "no_digit_yes" and band["3"]["applied"] is True
    assert band["3"]["value"] == 1.0
    assert band["2"]["applied"] is False
    ex = {
        (r["full_action"], r["estimated_action"]): int(r["n_queries"])
        for r in _csv(out / "extrapolation.csv")
    }
    assert ex[("incorrect", "correct")] == 1 and ex[("incorrect", "incorrect")] == 1


def test_problems_reported_and_no_overwrite(world, tmp_path):
    tmp, full, r1, van, sel = world
    # record_texts=false 인 run 을 재실행으로 주면 경고
    out = agr.main(
        ["--rerun", f"x={full}", "--full", str(full), "--out-root", str(tmp / "o2"), "--name", "t"]
    )
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert any("record_texts" in p for p in meta["problems"])
    with pytest.raises(FileExistsError):
        out.mkdir(parents=True, exist_ok=False)
