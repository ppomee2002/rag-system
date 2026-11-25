"""analyze_flips.py / freeze_inputs --question-ids-from-run / mode_a compare / prompt 버전 검증.

가상 데이터만 사용 (실제 실험 결과 아님). crag_system 형식 run 은 test_analyze_regression 의
synth fixture 를 그대로 쓴다.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "official_crag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_flips as af  # noqa: E402
from test_analyze_regression import (  # noqa: E402
    snapshot,
    synth,  # noqa: E402, F401
    write_jsonl,
)

from rag.crag_official import generation as offgen  # noqa: E402
from rag.crag_official import source  # noqa: E402


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run_flips(tmp, v_dir, f_dir, *extra, name="t"):
    return af.main(
        [
            "--baseline",
            str(v_dir),
            "--treatment",
            str(f_dir),
            "--name",
            name,
            "--out-root",
            str(tmp / "out"),
            *extra,
        ]
    )


def by_id(rows):
    return {r["query_id"]: r for r in rows}


def test_flip_classification_crag_system(synth):  # noqa: F811
    tmp, v_dir, f_dir = synth
    before = snapshot(v_dir, f_dir)
    out = run_flips(tmp, v_dir, f_dir)
    assert snapshot(v_dir, f_dir) == before  # 읽기 전용
    cases = by_id(read_csv(out / "flip_cases.csv"))
    # q5 는 둘 다 정답 -> flip 아님, err / warm 제외
    assert set(cases) == {"q1", "q2", "q3", "q4", "q6", "q7", "q8", "q9"}
    exp = {
        "q1": ("harm", "grader"),  # 정답 문서 0.5 < upper -> Correct 필터 제거
        "q2": ("harm", "grader"),  # Incorrect: 정답 문서 0.1 < lower -> 폐기
        "q3": ("harm", "prompt"),  # 최종 context 에 정답 있음
        "q4": ("harm", "prompt"),  # 근거 문자열 없이 정답, web 없음
        "q6": ("gain", "external"),  # web 이 정답 추가
        "q7": ("harm", "hold"),  # 원문 없음
        "q8": ("harm", "grader"),  # Incorrect, 정답 문서 0.2 < lower, web 0개
        "q9": ("harm", "prompt"),  # 정답 문서 유지, 부분 일치
    }
    for q, (d, cause) in exp.items():
        assert (cases[q]["direction"], cases[q]["primary_cause"]) == (d, cause), q
    assert "external" in cases["q2"]["contributing"]  # web 에 정답 없음
    assert "external" in cases["q8"]["contributing"]  # web 0개
    assert "evaluation" in cases["q9"]["contributing"]  # 부분 일치
    assert "lower 0.3" in cases["q2"]["cause_detail"]
    rep = (out / "report.md").read_text(encoding="utf-8")
    # Incorrect HARM 이 먼저
    assert rep.index("## 1. Incorrect 판정 HARM") < rep.index("`q2`") < rep.index("## 2.")
    assert rep.index("`q8`") < rep.index("## 2.")
    summ = read_csv(out / "cause_summary.csv")
    assert sum(int(r["n"]) for r in summ) == 8
    docs = read_csv(out / "flip_docs.csv")
    assert {d["query_id"] for d in docs} == set(cases)


def test_metric_disagreement_is_evaluation(synth):  # noqa: F811
    tmp, v_dir, f_dir = synth
    rows = [json.loads(x) for x in (v_dir / "queries.jsonl").read_text().splitlines()]
    for r in rows:
        if r["query_id"] == "q1":
            r["answer"] = "paris"  # 정규화 acc=1, 공식 match (대소문자 구분) = 0
    write_jsonl(v_dir / "queries.jsonl", rows)
    out = run_flips(tmp, v_dir, f_dir)
    c = by_id(read_csv(out / "flip_cases.csv"))["q1"]
    assert c["primary_cause"] == "evaluation"
    assert c["baseline_official_match"] == "0" and c["treatment_official_match"] == "0"


def test_flip_ids_filter_and_not_a_flip(synth):  # noqa: F811
    tmp, v_dir, f_dir = synth
    ids = tmp / "flip20.csv"
    ids.write_text("query_id,direction\nq2,harm\nq6,gain\nq5,harm\nq1,gain\nnope,harm\n")
    out = run_flips(tmp, v_dir, f_dir, "--flip-ids", str(ids))
    cases = by_id(read_csv(out / "flip_cases.csv"))
    assert set(cases) == {"q1", "q2", "q6"}
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    nf = {r["query_id"]: r["observed"] for r in meta["flip_ids"]["not_a_flip"]}
    assert nf == {"q5": "same_acc", "q1": "harm", "nope": "missing_in_run"}
    assert cases["q1"]["direction"] == "harm"  # 관측 방향 사용


def test_read_flip_ids_formats(tmp_path):
    (tmp_path / "a.txt").write_text("q1\nq2\n")
    assert af.read_flip_ids(tmp_path / "a.txt") == {"q1": None, "q2": None}
    (tmp_path / "b.json").write_text(json.dumps([{"question_id": "x", "direction": "GAIN"}]))
    assert af.read_flip_ids(tmp_path / "b.json") == {"x": "gain"}


def _mode_b_row(qid, action, kept, added, scores, pred, acc, answers, i):
    return {
        "schema": af.MODE_B_SCHEMA,
        "request_index": i,
        "question_id": qid,
        "question": f"question {qid}?",
        "warmup": False,
        "policy_id": "crag_rule@official_t5" if action else "vanilla",
        "thresholds": {"upper": 0.592, "lower": -0.995} if action else None,
        "decision": {"action": action},
        "retrieval_doc_ids": ["d1", "d2"],
        "grading_events": [{"rank": r, "score": s} for r, s in enumerate(scores)],
        "evidence": {"kept_retrieved_ids": kept, "added_ids": added},
        "correction": {"performed": bool(added), "web_doc_ids": added, "query": "rw"},
        "prediction": pred,
        "ground_truth": answers,
        "metrics": {"acc": acc},
    }


def test_mode_b_format(tmp_path):
    frozen = tmp_path / "frozen.jsonl"
    write_jsonl(
        frozen,
        [
            {
                "question_id": "m1",
                "docs": [
                    {"id": "d1", "text": "Alice Smith is a painter."},
                    {"id": "d2", "text": "Unrelated text."},
                ],
            }
        ],
    )
    runs = {}
    for name, row in (
        ("van", _mode_b_row("m1", None, ["d1", "d2"], [], [], "painter", 1.0, ["painter"], 0)),
        (
            "t5",
            _mode_b_row("m1", "incorrect", [], [], [-2.0, -1.5], "singer", 0.0, ["painter"], 0),
        ),
    ):
        d = tmp_path / "results" / name / "r"
        d.mkdir(parents=True)
        write_jsonl(d / "queries.jsonl", [row])
        (d / "calls.jsonl").write_text("")
        cfg = {"inputs": {"frozen_path": str(frozen)}, "web_search": {}}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        runs[name] = d
    out = run_flips(tmp_path, runs["van"], runs["t5"])
    c = read_csv(out / "flip_cases.csv")[0]
    assert (c["direction"], c["action"], c["primary_cause"]) == ("harm", "incorrect", "grader")
    assert "-0.995" in c["cause_detail"]
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["treatment_format"] == "mode_b" and meta["upper"] == 0.592


# ---------------------------------------------------------------- freeze ids from run
def test_ids_from_runs_excludes_warmup_and_intersects(tmp_path):
    import freeze_inputs as fi

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_jsonl(
        a / "queries.jsonl",
        [{"query_id": "w", "warmup": True}]
        + [{"query_id": f"q{i}", "warmup": False} for i in range(5)]
        + [{"query_id": "e", "warmup": False, "error": "x"}],
    )
    write_jsonl(b / "queries.jsonl", [{"query_id": f"q{i}"} for i in (4, 1, 2, 3)])
    ids, info = fi.ids_from_runs([a, b])
    assert ids == ["q1", "q2", "q3", "q4"]
    assert info["runs"][0] == {
        "run": str(a),
        "n_measured_ok": 5,
        "n_warmup_rows": 1,
        "n_error_rows": 1,
    }


def test_freeze_expect_n_mismatch(tmp_path):
    import freeze_inputs as fi

    a = tmp_path / "a"
    a.mkdir()
    write_jsonl(a / "queries.jsonl", [{"query_id": "q1"}])
    with pytest.raises(SystemExit, match="--expect-n"):
        fi.main(
            [
                "--question-ids-from-run",
                str(a),
                "--expect-n",
                "200",
                "--out",
                str(tmp_path / "o.jsonl"),
            ]
        )


# ---------------------------------------------------------------- prompt versions
def test_prompt_versions():
    v3 = offgen.format_popqa_prompt("Q {x}?", "P {y}")
    assert v3 == source.POPQA_PROMPT.replace("{paragraph}", "P {y}").replace("{question}", "Q {x}?")
    old = offgen.format_popqa_prompt("Q?", "P", "v1v2_selfrag_format")
    # 1b34082 ~ 684bd67 CRAG_Inference.py format_prompt (task=popqa)
    assert old == "### Instruction:\nQ?\n\n### Response:\n[Retrieval]<paragraph>P</paragraph>"
    with pytest.raises(ValueError):
        offgen.format_popqa_prompt("Q", "P", "nope")


# ---------------------------------------------------------------- mode_a compare
def _eval_run(root: Path, name: str, method: str, matches: list[int], actions, model="m"):
    preds = root / f"gen_{name}"
    preds.mkdir(parents=True)
    (preds / "config.yaml").write_text(
        yaml.safe_dump({"method": method, "llm": {"model": model, "api": "completions"}}),
        encoding="utf-8",
    )
    ev = root / f"eval_{name}"
    ev.mkdir()
    import mode_a

    summ = {"generate": mode_a.generate_info(preds)}
    (ev / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    write_jsonl(
        ev / "per_question.jsonl",
        [
            {"index": i, "question_id": f"id{i}", "action": a, "official_match": m, "acc": m}
            for i, (m, a) in enumerate(zip(matches, actions, strict=True))
        ],
    )
    return ev


def test_mode_a_compare(tmp_path, monkeypatch):
    import mode_a

    monkeypatch.setattr(mode_a, "_run_dir", lambda step, rid: _mk(tmp_path / "cmp"))
    a = _eval_run(tmp_path, "rag", "rag", [1, 1, 0, 0, 1], [None] * 5)
    b = _eval_run(
        tmp_path, "crag", "crag", [1, 0, 1, 1, 1], ["correct", "incorrect", "ambiguous", "c", "c"]
    )
    out = mode_a.main(["compare", "--a", str(a), "--b", str(b)]) or tmp_path / "cmp"
    s = json.loads((tmp_path / "cmp" / "summary.json").read_text(encoding="utf-8"))
    m = s["metrics"]["official_match"]
    assert (m["harm_a_only"], m["gain_b_only"], m["both"], m["n"]) == (1, 2, 2, 5)
    assert m["diff_b_minus_a"] == pytest.approx(0.2)
    assert s["by_action_official_match"]["incorrect"]["harm"] == 1
    assert out is not None
    # 생성 조건 (model) 이 다르면 거부
    c = _eval_run(tmp_path / "x", "crag", "crag", [1] * 5, ["c"] * 5, model="other")
    with pytest.raises(SystemExit, match="생성 조건"):
        mode_a.cmd_compare(
            type("A", (), {"a": str(a), "b": str(c), "allow_condition_diff": False})()
        )


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
