"""diag_crag_logs.py 검증. 가상 데이터만 사용 (실제 실험 결과 아님)."""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import diag_crag_logs as dg  # noqa: E402
from crag_fixtures import FakeEngine, Retriever, Web  # noqa: E402
from runner import Components, Runner  # noqa: E402

from rag.retrieval.web_cache import RecordingWebSearch  # noqa: E402

LONG = "The document discusses opera houses and does not clearly relate to the question asked."
R1_RW = 'Here is a concise web search query: "Magic Flute composer"'
R3_RW = (
    "Mount Everest height\nThis query will help find the height of the mountain "
    "in meters and feet today"
)
# (qid, vanilla_acc, full_acc, action, scores, grade_outputs, grade_out_tokens, rewrite, web)
CASES = [
    ("r1", 1, 0, "incorrect", [0.0, 0.0, 0.0], [LONG, "Not relevant", "0"], [40, 3, 3], R1_RW,
     [("w1", "Opera houses in Europe."), ("w2", "")]),
    ("r2", 1, 0, "correct", [0.5, 0.8, 0.2], ["0.5", "0.8", "0.2"], [3, 3, 3], None, None),
    ("r3", 1, 0, "ambiguous", [0.5, 0.1, 0.0], ["0.5", "0.1", "0"], [3, 3, 3], R3_RW,
     [("w3", "Mount Everest is 8,849 m tall.")]),
    ("r4", 1, 0, "incorrect", [0.1, 0.1, 0.1], ["0.1"] * 3, [3, 3, 3], "Hamlet author", []),
    ("c1", 0, 1, "correct", [0.9, 0.1, 0.1], ["0.9", "0.1", "0.1"], [3, 3, 3], None, None),
    ("c2", 1, 1, "incorrect", [0.1, 0.1, 0.1], ["0.1"] * 3, [3, 3, 3],
     "Earth's largest river by discharge", [("w4", "The Amazon has the largest discharge.")]),
    ("b1", 0, 0, "correct", [0.9, 0.2, 0.2], ["0.9", "0.2", "0.2"], [3, 3, 3], None, None),
]  # fmt: skip


def approx(x):
    return pytest.approx(x, abs=1e-6)  # write_csv 는 소수 6자리로 반올림


GOLD = {"r1": "Mozart", "r3": "Everest", "c2": "Amazon"}


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def synth(tmp_path):
    base = tmp_path / "results" / "crag_system" / "smoke"
    v_dir, f_dir = base / "V" / "r", base / "F" / "r"
    v_dir.mkdir(parents=True)
    f_dir.mkdir(parents=True)
    cfg = {
        "crag": {"upper_threshold": 0.7, "lower_threshold": 0.3, "web_search": {}},
        "grading": {"strategy": "full"},
        "run": {"record_texts": True},
    }
    (f_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (v_dir / "config.yaml").write_text(yaml.safe_dump({"pipeline": "vanilla"}), encoding="utf-8")
    vrows, frows, calls, cache = [], [], [], []
    for i, (qid, vacc, facc, act, sc, outs, toks, rw, web) in enumerate(CASES):
        gold = [GOLD.get(qid, "zzz")]
        ret = [f"{qid}_d{j}" for j in range(3)]
        base_row = {"request_index": i, "query_id": qid, "warmup": False, "question": f"Q {qid}?",
                    "gold_answers": gold, "retrieval_doc_ids": ret}  # fmt: skip
        vrows.append({**base_row, "acc": vacc, "contexts": ret, "answer": "v"})
        frows.append(
            {
                **base_row,
                "acc": facc,
                "answer": "f",
                "corrective_action": act,
                "contexts": [],
                "extra": {
                    "retrieval_doc_ids": ret,
                    "grading": {"strategy": "full", "scores_full": sc},
                    "rewritten_query": rw,
                    "web_doc_ids": [w for w, _ in web or []],
                },
            }
        )
        for j, (o, t) in enumerate(zip(outs, toks, strict=True)):
            calls.append({"request_index": i, "query_id": qid, "kind": "grade", "idx": j,
                          "wall": 0.1, "out_tokens": t, "prompt_tokens": 100,
                          "finish_reason": "stop", "output_text": o})  # fmt: skip
        calls.append({"request_index": i, "query_id": qid, "kind": "generate", "idx": 0,
                      "wall": 0.5, "out_tokens": 5})  # fmt: skip
        if rw is not None and web:
            cache.append({"query": rw.strip(), "top_k": 5, "latency": 0.1,
                          "docs": [{"id": w, "text": t, "metadata": {"title": f"T {w}"}}
                                   for w, t in web]})  # fmt: skip
    frows.append({"request_index": 99, "query_id": "err", "warmup": False, "error": "Boom"})
    write_jsonl(v_dir / "queries.jsonl", vrows)
    write_jsonl(f_dir / "queries.jsonl", frows)
    write_jsonl(v_dir / "calls.jsonl", [])
    write_jsonl(f_dir / "calls.jsonl", calls)
    write_jsonl(f_dir / "web_cache.jsonl", cache)
    return tmp_path, v_dir, f_dir


def snapshot(*dirs):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in dirs
        for p in sorted(Path(d).rglob("*"))
        if p.is_file()
    }


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run(tmp, v_dir, f_dir, *extra):
    return dg.main(
        ["--vanilla", str(v_dir), "--full", str(f_dir), "--name", "t",
         "--out-root", str(tmp / "analysis"), *extra]
    )  # fmt: skip


def rows_by(path, *keys):
    return {tuple(r[k] for k in keys): r for r in read_csv(path)}


def test_synthetic_end_to_end(synth):
    tmp, v_dir, f_dir = synth
    before = snapshot(v_dir, f_dir)
    out = run(tmp, v_dir, f_dir)
    assert snapshot(v_dir, f_dir) == before  # run 폴더는 읽기만
    for f in ("report.md", "score_distribution.csv", "score_values.csv", "score_by_action.csv",
              "grade_tokens.csv", "grade_tokens_by_band.csv", "long_grade_calls.csv",
              "rewrite_cases.csv", "rewrite_summary.csv", "web_docs.csv", "web_summary.csv",
              "diag_meta.json"):  # fmt: skip
        assert (out / f).exists(), f

    meta = json.loads((out / "diag_meta.json").read_text(encoding="utf-8"))
    # c2 는 Vanilla 도 정답이지만 Full 정답이므로 full_correct. 오류 행 제외
    assert meta["group_sizes"] == {"regression": 4, "full_correct": 2, "both_wrong": 1}
    assert meta["errors"]["full"] == ["err"]

    # 1. 점수 분포 (회귀 12 문서: 0.0 x4, (0,0.3) x5, [0.3,0.7) x2, >=0.7 x1)
    sd = rows_by(out / "score_distribution.csv", "group", "band")
    bands = ("0.0", "(0,0.3)", "[0.3,0.7)", ">=0.7")
    assert [int(sd[("regression", b)]["n_docs"]) for b in bands] == [4, 5, 2, 1]
    assert float(sd[("regression", "0.0")]["share_of_graded"]) == approx(4 / 12)
    assert int(sd[("full_correct", ">=0.7")]["n_docs"]) == 1
    assert int(sd[("all", "0.0")]["n_docs"]) == 4
    assert sd[("regression", "0.0")]["basis"] == "로그"

    # 2. 출력 토큰
    gt = {r["group"]: r for r in read_csv(out / "grade_tokens.csv")}
    assert float(gt["regression"]["mean"]) == approx((40 + 3 * 11) / 12)
    assert float(gt["regression"]["max"]) == 40
    assert float(gt["regression"]["share_gt_8_tokens"]) == approx(1 / 12)
    assert float(gt["regression"]["share_no_digit"]) == approx(2 / 12)
    assert float(gt["full_correct"]["share_gt_8_tokens"]) == 0
    long_calls = read_csv(out / "long_grade_calls.csv")
    assert [(c["query_id"], c["idx"], c["out_tokens"]) for c in long_calls] == [("r1", "0", "40")]
    tb = rows_by(out / "grade_tokens_by_band.csv", "group", "band")
    assert float(tb[("regression", "0.0")]["max"]) == 40 and tb[("regression", "0.0")]["n"] == "4"

    # 3. action 별
    sa = rows_by(out / "score_by_action.csv", "group", "action")
    assert sa[("regression", "incorrect")]["n_queries"] == "2"
    assert sa[("regression", "incorrect")]["queries_all_zero"] == "1"  # r1
    assert float(sa[("regression", "correct")]["mean_max_score"]) == approx(0.8)
    assert float(sa[("full_correct", "incorrect")]["share_0.0"]) == 0

    # 4. rewrite
    rc = {r["query_id"]: r for r in read_csv(out / "rewrite_cases.csv")}
    assert set(rc) == {"r1", "r3", "r4", "c2"}  # correction 한 query 만
    assert set(rc["r1"]["flags"].split(";")) == {"preface", "quotes"}
    assert {"multiline", "explanation", "long_words"} <= set(rc["r3"]["flags"].split(";"))
    assert rc["r3"]["lines"] == "2"
    assert rc["c2"]["flags"] == ""  # 단어 안 apostrophe 는 따옴표 아님
    assert rc["r1"]["found_in_web_cache"] == "True" and rc["r4"]["found_in_web_cache"] == "False"
    rs = {r["group"]: r for r in read_csv(out / "rewrite_summary.csv")}
    assert float(rs["regression"]["share_any_flag"]) == approx(2 / 3)
    assert float(rs["full_correct"]["share_any_flag"]) == 0

    # 5. web
    ws = rows_by(out / "web_summary.csv", "group", "action")
    reg = ws[("regression", "all")]
    assert reg["n_queries"] == "3" and reg["queries_no_web_docs"] == "1"
    assert reg["n_web_docs"] == "3" and float(reg["share_empty_text"]) == approx(1 / 3)
    assert float(reg["est_share_queries_any_gold"]) == approx(1 / 2)  # r3 만 (r4 는 문서 없음)
    assert float(ws[("full_correct", "all")]["est_share_queries_any_gold"]) == 1.0
    wd = {r["doc_id"]: r for r in read_csv(out / "web_docs.csv") if r["doc_id"]}
    assert wd["w1"]["chars"] == str(len("Opera houses in Europe.")) and wd["w1"]["title"] == "T w1"
    assert wd["w2"]["empty_text"] == "True" and wd["w3"]["gold_string"] == "True"

    # 6. 비교 요약
    comp = {r["metric"]: r for r in meta["comparison"]}
    assert comp["점수 0.0 비율 (문서)"]["regression"] == approx(round(4 / 12, 4))
    assert comp["점수 0.0 비율 (문서)"]["full_correct"] == 0
    assert comp["rewrite 이상 징후 비율"]["basis"] == "추정"

    md = (out / "report.md").read_text(encoding="utf-8")
    for s in ("[로그]", "[추정]", "## 1.", "## 6.", "서버 실행 후 확인할 항목"):
        assert s in md, s
    assert "Here is a concise" in md  # 징후 예시


def test_without_record_texts(synth):
    """출력 원문이 없어도 out_tokens 기반 지표는 계산, 원문 기반 지표는 비워 둔다."""
    tmp, v_dir, f_dir = synth
    rows = [json.loads(x) for x in (f_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    write_jsonl(f_dir / "calls.jsonl", [{**r, "output_text": None} for r in rows])
    out = run(tmp, v_dir, f_dir)
    gt = {r["group"]: r for r in read_csv(out / "grade_tokens.csv")}
    assert gt["regression"]["share_no_digit"] == "" and float(gt["regression"]["max"]) == 40
    meta = json.loads((out / "diag_meta.json").read_text(encoding="utf-8"))
    assert meta["availability"]["n_grade_output_text"] == 0


def test_parallel_grading_disables_doc_mapping(synth):
    tmp, v_dir, f_dir = synth
    cfg = yaml.safe_load((f_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["grading"]["parallel"] = True
    (f_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = run(tmp, v_dir, f_dir)
    tb = rows_by(out / "grade_tokens_by_band.csv", "group", "band")
    assert tb[("regression", "0.0")]["n"] == "0"
    meta = json.loads((out / "diag_meta.json").read_text(encoding="utf-8"))
    assert meta["availability"]["mapping"].startswith("대응 불가")


def test_missing_web_text_is_not_counted_as_no_gold(synth):
    tmp, v_dir, f_dir = synth
    (f_dir / "web_cache.jsonl").unlink()
    out = run(tmp, v_dir, f_dir)
    reg = rows_by(out / "web_summary.csv", "group", "action")[("regression", "all")]
    assert reg["n_text_found"] == "0" and reg["est_share_queries_any_gold"] == ""


def test_refuses_output_inside_run_dir(synth):
    tmp, v_dir, f_dir = synth
    with pytest.raises(ValueError, match="run 폴더 안"):
        dg.main(["--vanilla", str(v_dir), "--full", str(f_dir), "--out-root", str(f_dir / "x")])


def test_helpers():
    assert dg.band(0.0, 0.7, 0.3) == "0.0" and dg.band(0.05, 0.7, 0.3) == "(0,0.3)"
    assert dg.band(0.3, 0.7, 0.3) == "[0.3,0.7)" and dg.band(0.7, 0.7, 0.3) == ">=0.7"
    assert dg.band(None, 0.7, 0.3) == "not_graded"

    def f(s, q="q"):
        return dg.rewrite_flags(s, q, 100, 15)

    assert f("Sure! The query is: Mozart opera") == ["preface"]
    assert f("Query: Mozart") == ["preface"]
    assert f("`Mozart`") == ["quotes"] and f("'Mozart'") == ["quotes"]
    assert f("O'Brien Earth's") == []
    assert f("- Mozart opera") == ["markdown"]
    assert f("") == ["empty"] and f(None) == ["empty"]
    assert f("Who composed it?", "Who composed it?") == ["same_as_question"]
    assert "long_chars" in f("x" * 101)
    s = dg.stats([1, 2, 3, None])
    assert s["n"] == 3 and s["p50"] == 2 and dg.stats([])["mean"] is None


class WebHurtsEngine(FakeEngine):
    def complete(self, request):
        r = super().complete(request)
        if request.kind == "generate" and "[1] web 0" in request.prompt:
            r.text = "unknown"
        return r


def test_reads_runner_output(setup, tmp_path):
    """Runner 가 실제로 쓰는 queries / calls / web_cache 형식으로 끝까지 동작하는지."""
    tmp, make = setup
    cfg_v, _ = make(["exp_id=SMOKE-V", "pipeline=vanilla"])
    v_dir = Runner(cfg_v, Components(Retriever(), Web(), FakeEngine()), run_id="r").run()
    rec = tmp_path / "web_rec.jsonl"
    cfg_f, _ = make(["exp_id=SMOKE-F", "grading.strategy=full", "run.record_texts=true"])
    f_dir = Runner(
        cfg_f, Components(Retriever(), RecordingWebSearch(Web(), rec), WebHurtsEngine()), run_id="r"
    ).run()
    shutil.copy(rec, f_dir / "web_cache.jsonl")
    before = snapshot(v_dir, f_dir)
    out = run(tmp_path, v_dir, f_dir)
    assert snapshot(v_dir, f_dir) == before
    meta = json.loads((out / "diag_meta.json").read_text(encoding="utf-8"))
    sizes = meta["group_sizes"]
    assert sizes["regression"] > 0 and sizes["full_correct"] > 0
    sd = rows_by(out / "score_distribution.csv", "group", "band")
    # 가짜 grader: 회귀 (홀수 query) 는 모두 0.1, Full 정답 (짝수) 은 모두 0.9
    assert float(sd[("regression", "(0,0.3)")]["share_of_graded"]) == 1.0
    assert float(sd[("full_correct", ">=0.7")]["share_of_graded"]) == 1.0
    tb = rows_by(out / "grade_tokens_by_band.csv", "group", "band")
    assert int(tb[("regression", "(0,0.3)")]["n"]) == 3 * sizes["regression"]  # top_k=3 대응
    rc = read_csv(out / "rewrite_cases.csv")
    assert rc and all(r["found_in_web_cache"] == "True" for r in rc)
    ws = rows_by(out / "web_summary.csv", "group", "action")
    assert ws[("regression", "incorrect")]["n_web_docs"] == str(2 * sizes["regression"])
