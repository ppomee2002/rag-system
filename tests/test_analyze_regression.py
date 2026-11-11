"""analyze_regression.py 검증. 가상 데이터만 사용 (실제 실험 결과 아님).

1) 손으로 만든 run 폴더: 사례마다 의도한 신호 / 잠정 분류가 나오는지
2) Runner + 가짜 엔진으로 만든 run 폴더: 실제 결과 파일 형식을 그대로 읽는지
3) 읽기 전용: 분석 전후 run 폴더 파일이 바뀌지 않는지
"""

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import analyze_regression as ar  # noqa: E402
from crag_fixtures import FakeEngine, Retriever, Web  # noqa: E402
from runner import Components, Runner  # noqa: E402

from rag.retrieval.web_cache import RecordingWebSearch  # noqa: E402

CHUNKS = {
    "a1": "Paris is the capital and largest city of France.",
    "a2": "France is a country in Western Europe.",
    "a3": "The Seine river flows through northern France.",
    "b1": "The Magic Flute is an opera by Wolfgang Amadeus Mozart.",
    "b2": "An opera is a form of theatre.",
    "b3": "Vienna was a center of classical music.",
    "c1": "Mount Everest is Earth's highest mountain above sea level.",
    "c2": "The Himalayas span several countries.",
    "c3": "Climbing requires oxygen at altitude.",
    "d1": "The theory of relativity changed physics.",
    "d2": "Physics studies matter and energy.",
    "d3": "A Nobel Prize is awarded annually.",
    "e1": "Tokyo is the capital of Japan.",
    "e2": "Japan is an island country.",
    "e3": "Kyoto was the former capital.",
    "f1": "Rivers of South America.",
    "f2": "Rainforest biodiversity.",
    "f3": "Tropical climate.",
    "h1": "Shakespeare wrote Hamlet around 1600.",
    "h2": "Hamlet is a tragedy.",
    "h3": "Elizabethan theatre.",
    "i1": "Barack Obama was the 44th president of the United States.",
    "i2": "The White House is in Washington.",
    "i3": "Presidents serve four-year terms.",
}
WEB = {
    "rewritten mozart": [("w2a", "Opera houses in Europe."), ("w2b", "Famous singers.")],
    "rewritten everest": [
        ("w3a", "A long article about mountains in Asia. " * 20),
        ("w3b", "K2 is the second highest mountain. " * 20),
    ],
    "rewritten amazon": [("w6a", "The Amazon is the largest river by discharge.")],
}


def q(qid, i, question, gold, ret, scores, action, ctx, answer, acc, *, web=None, rw=None):
    return {
        "request_index": i,
        "query_id": qid,
        "warmup": False,
        "question": question,
        "gold_answers": gold,
        "answer": answer,
        "acc": acc,
        "em": 0.0,
        "f1": 0.0,
        "latency": 1.0,
        "contexts": ctx,
        "corrective_action": action,
        "retrieval_doc_ids": ret,
        "grader_scores": [s for s in scores if s is not None],
        "extra": {
            "retrieval_doc_ids": ret,
            "retrieval_scores": [0.9, 0.8, 0.7][: len(ret)],
            "grading": {"strategy": "full", "scores_full": scores},
            "rewritten_query": rw,
            "web_doc_ids": web or [],
        },
    }


# (qid, question, gold, ret, scores, action, full_ctx, full_answer, full_acc, vanilla_answer,
#  vanilla_acc, web_ids, rewritten, gen_finish, grade_outputs)
CASES = [
    # 1 Correct 필터가 정답 문서 제거 (정답 문서 0.5, 무관 문서 0.9)
    (
        "q1",
        "What is the capital of France?",
        ["Paris"],
        ["a1", "a2", "a3"],
        [0.5, 0.9, 0.2],
        "correct",
        ["a2"],
        "Lyon",
        0.0,
        "Paris",
        1.0,
        None,
        None,
        "stop",
        ["0.5", "0.9", "0.2"],
    ),
    # 2 Incorrect: 정답 문서 0.1 -> 폐기, web 에 정답 없음, 모름 답변, grade 출력 하나 파싱 불가
    (
        "q2",
        "Who composed The Magic Flute?",
        ["Mozart", "Wolfgang Amadeus Mozart"],
        ["b1", "b2", "b3"],
        [0.1, 0.1, 0.0],
        "incorrect",
        ["w2a", "w2b"],
        "I don't know.",
        0.0,
        "Mozart",
        1.0,
        ["w2a", "w2b"],
        "rewritten mozart",
        "stop",
        ["0.1", "0.1", "Not relevant."],
    ),
    # 3 Ambiguous: 정답 문서 유지 + 긴 web 문서, 그래도 오답, 생성 잘림
    (
        "q3",
        "What is the highest mountain?",
        ["Everest", "Mount Everest"],
        ["c1", "c2", "c3"],
        [0.5, 0.1, 0.1],
        "ambiguous",
        ["c1", "w3a", "w3b"],
        "K2",
        0.0,
        "Everest",
        1.0,
        ["w3a", "w3b"],
        "rewritten everest",
        "length",
        ["0.5", "0.1", "0.1"],
    ),
    # 4 어느 context 에도 정답 문자열 없음 (Vanilla 는 모델 지식으로 정답)
    (
        "q4",
        "Who developed the theory of relativity?",
        ["Einstein", "Albert Einstein"],
        ["d1", "d2", "d3"],
        [0.9, 0.2, 0.1],
        "correct",
        ["d1"],
        "Not mentioned in the context.",
        0.0,
        "Einstein",
        1.0,
        None,
        None,
        "stop",
        ["0.9", "0.2", "0.1"],
    ),
    # 5 둘 다 정답 (회귀 아님)
    (
        "q5",
        "What is the capital of Japan?",
        ["Tokyo"],
        ["e1", "e2", "e3"],
        [0.9, 0.2, 0.5],
        "correct",
        ["e1"],
        "Tokyo",
        1.0,
        "Tokyo",
        1.0,
        None,
        None,
        "stop",
        ["0.9", "0.2", "0.5"],
    ),
    # 6 개선: Vanilla 오답 -> Full 이 web 에서 정답
    (
        "q6",
        "What is the largest river by discharge?",
        ["Amazon"],
        ["f1", "f2", "f3"],
        [0.1, 0.1, 0.1],
        "incorrect",
        ["w6a"],
        "The Amazon",
        1.0,
        "Nile",
        0.0,
        ["w6a"],
        "rewritten amazon",
        "stop",
        ["0.1", "0.1", "0.1"],
    ),
    # 7 문서 원문 없음 -> 판정 보류
    (
        "q7",
        "Unknown doc question?",
        ["Zeta"],
        ["zz1", "zz2", "zz3"],
        [0.1, 0.9, 0.1],
        "correct",
        ["zz2"],
        "Alpha",
        0.0,
        "Zeta",
        1.0,
        None,
        None,
        "stop",
        ["0.1", "0.9", "0.1"],
    ),
    # 8 Incorrect 인데 web 결과 0개
    (
        "q8",
        "Who wrote Hamlet?",
        ["Shakespeare"],
        ["h1", "h2", "h3"],
        [0.2, 0.1, 0.1],
        "incorrect",
        [],
        "Marlowe",
        0.0,
        "Shakespeare",
        1.0,
        [],
        "zzz qqq",
        "stop",
        ["0.2", "0.1", "0.1"],
    ),
    # 9 정답 문서 유지, 답변이 부분 일치 (F1 > 0, accuracy 0)
    (
        "q9",
        "Who was the 44th US president?",
        ["Barack Obama"],
        ["i1", "i2", "i3"],
        [0.9, 0.1, 0.1],
        "correct",
        ["i1"],
        "Obama",
        0.0,
        "Barack Obama",
        1.0,
        None,
        None,
        "stop",
        ["0.9", "0.1", "0.1"],
    ),
]


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def synth(tmp_path):
    base = tmp_path / "results" / "crag_system" / "smoke"
    v_dir, f_dir = base / "V" / "r", base / "F" / "r"
    v_dir.mkdir(parents=True)
    f_dir.mkdir(parents=True)
    chunks = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, [{"id": k, "text": t} for k, t in CHUNKS.items()])
    cfg = {
        "pipeline": "crag_graph",
        "retriever": {"docs_path": str(chunks)},
        "crag": {"upper_threshold": 0.7, "lower_threshold": 0.3, "web_search": {}},
        "grading": {"strategy": "full"},
        "run": {"record_texts": True},
    }
    (f_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    vcfg = {**cfg, "pipeline": "vanilla", "run": {"record_texts": False}}
    (v_dir / "config.yaml").write_text(yaml.safe_dump(vcfg), encoding="utf-8")
    vrows, frows, fcalls, vcalls = [], [], [], []
    for i, c in enumerate(CASES):
        qid, quest, gold, ret, sc, act, fctx, fans, facc, vans, vacc, web, rw, fin, gout = c
        frows.append(q(qid, i, quest, gold, ret, sc, act, fctx, fans, facc, web=web, rw=rw))
        v = q(qid, i, quest, gold, ret, [], None, ret, vans, vacc)
        v["extra"] = {"retrieval_scores": [0.9, 0.8, 0.7]}
        vrows.append(v)
        for j, o in enumerate(gout):
            fcalls.append(
                {
                    "request_index": i,
                    "query_id": qid,
                    "kind": "grade",
                    "idx": j,
                    "wall": 0.1,
                    "output_text": o,
                }
            )
        fcalls.append(
            {
                "request_index": i,
                "query_id": qid,
                "kind": "generate",
                "idx": 0,
                "wall": 0.5,
                "prompt_tokens": 300,
                "out_tokens": 5,
                "finish_reason": fin,
            }
        )
        vcalls.append(
            {
                "request_index": i,
                "query_id": qid,
                "kind": "generate",
                "idx": 0,
                "wall": 0.5,
                "prompt_tokens": 200,
                "out_tokens": 5,
                "finish_reason": "stop",
            }
        )
    # warmup / 오류 행은 제외되어야 함
    vrows.append({**vrows[0], "request_index": -1, "warmup": True, "query_id": "warm"})
    frows.append({"request_index": 99, "query_id": "err", "warmup": False, "error": "Boom"})
    write_jsonl(v_dir / "queries.jsonl", vrows)
    write_jsonl(f_dir / "queries.jsonl", frows)
    write_jsonl(v_dir / "calls.jsonl", vcalls)
    write_jsonl(f_dir / "calls.jsonl", fcalls)
    write_jsonl(
        f_dir / "web_cache.jsonl",
        [
            {
                "query": k,
                "top_k": 2,
                "latency": 0.1,
                "docs": [{"id": i, "text": t, "metadata": {}, "score": None} for i, t in v],
            }
            for k, v in WEB.items()
        ],
    )
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


def run_analysis(tmp, v_dir, f_dir, name="t"):
    return ar.main(
        [
            "--vanilla",
            str(v_dir),
            "--full",
            str(f_dir),
            "--name",
            name,
            "--out-root",
            str(tmp / "analysis"),
        ]
    )


def test_synthetic_signals_and_buckets(synth):
    tmp, v_dir, f_dir = synth
    before = snapshot(v_dir, f_dir)
    out = run_analysis(tmp, v_dir, f_dir)
    assert snapshot(v_dir, f_dir) == before  # run 폴더는 읽기만

    for f in (
        "report.md",
        "regression_cases.csv",
        "regression_docs.csv",
        "signal_summary.csv",
        "bucket_summary.csv",
        "action_summary.csv",
        "gold_doc_grading.csv",
        "analysis_meta.json",
    ):
        assert (out / f).exists(), f
    cases = {r["query_id"]: r for r in read_csv(out / "regression_cases.csv")}
    # 회귀 = Vanilla 정답 & Full 오답: q5 (둘 다 정답), q6 (개선) 제외, 오류/warmup 행 제외
    assert set(cases) == {"q1", "q2", "q3", "q4", "q7", "q8", "q9"}

    def sig(qid):
        return set(cases[qid]["signals"].split(";"))

    assert {
        "G1_gold_doc_below_upper",
        "G3_nongold_kept_gold_dropped",
        "E1_gold_ctx_removed",
    } <= sig("q1")
    assert "A1_gold_in_ctx_answer_wrong" not in sig("q1")
    assert cases["q1"]["provisional_bucket"].startswith("근거 제거 의심: Correct")

    assert {
        "G2_gold_doc_below_lower",
        "G4_incorrect_action",
        "G5_grade_output_unparsable",
        "E1_gold_ctx_removed",
        "E2_vanilla_ctx_all_dropped",
        "C2_web_no_gold",
        "A2_refusal_like",
    } <= sig("q2")
    assert cases["q2"]["provisional_bucket"].startswith("근거 제거 의심: Incorrect")
    assert cases["q2"]["rewritten_query"] == "rewritten mozart"

    assert {
        "A1_gold_in_ctx_answer_wrong",
        "C2_web_no_gold",
        "C6_context_grew",
        "A4_gen_truncated",
    } <= sig("q3")
    assert cases["q3"]["provisional_bucket"].startswith("생성 문제 의심")
    assert cases["q3"]["str_gold_source_full"] == "faiss"

    assert {"E3_no_gold_string_anywhere", "A2_refusal_like"} <= sig("q4")
    assert cases["q4"]["provisional_bucket"].startswith("문자열 근거 없음")

    assert "D2_text_missing" in sig("q7")
    assert cases["q7"]["provisional_bucket"] == "보류: 문서 원문 없음"

    assert {"C1_web_empty", "E1_gold_ctx_removed"} <= sig("q8")
    assert "C2_web_no_gold" not in sig("q8")

    assert {"A1_gold_in_ctx_answer_wrong", "A3_partial_overlap"} <= sig("q9")

    # 수동 확인 열은 비어 있음 (사람이 채움)
    assert all(r["manual_label"] == "" for r in cases.values())

    # 문서 단위 표: q1 의 정답 문서 a1 은 0.5 로 제거됨
    docs = read_csv(out / "regression_docs.csv")
    a1 = next(d for d in docs if d["query_id"] == "q1" and d["doc_id"] == "a1")
    assert a1["gold_string"] == "True" and a1["kept_in_full"] == "False"
    assert a1["grade_band"] == "0.3-0.7" and a1["grade_output"] == "0.5"
    w = [d for d in docs if d["query_id"] == "q2" and d["source"] == "web"]
    assert {d["doc_id"] for d in w} == {"w2a", "w2b"} and all(
        d["gold_string"] == "False" for d in w
    )

    # 신호 대조: 개선 사례 q6 의 C3 은 Full 정답 쪽에서 세짐
    ss = {r["signal"]: r for r in read_csv(out / "signal_summary.csv")}
    assert ss["C3_web_has_gold"]["n_full_correct"] == "1"
    assert ss["C3_web_has_gold"]["n_regression"] == "0"
    assert ss["E1_gold_ctx_removed"]["n_regression"] == "3"  # q1, q2, q8
    assert ss["E1_gold_ctx_removed"]["basis"] == "문자열"

    acts = {r["action"]: r for r in read_csv(out / "action_summary.csv")}
    assert acts["incorrect"]["regressions"] == "2" and acts["incorrect"]["improvements"] == "1"

    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["n_common"] == 9 and meta["n_regressions"] == 7 and meta["n_improvements"] == 1
    assert meta["missing"]["errors_full"] == ["err"]
    assert meta["availability"]["chunk_ids_found"] == 24  # zz* 3개는 없음
    assert meta["grade_format"]["no_digit_rate"] == pytest.approx(1 / 27)
    assert meta["gen_prompt_tokens"] == {"vanilla": 200.0, "full": 300.0}

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "잠정 분류" in md and "참고 지표" in md and "서버 실행 후 확인할 항목" in md
    assert "78.5%" in md  # 사용자 보고 수치와 대조하라는 확인 항목
    # 여러 정답 (" | " 로 연결) 이 Markdown 표 칸을 깨지 않음
    q2_line = next(x for x in md.splitlines() if x.startswith("| q2 |"))
    assert "Mozart \\| Wolfgang Amadeus Mozart" in q2_line
    assert len(re.findall(r"(?<!\\)\|", q2_line)) == 7  # 6열 표의 칸 구분자


def test_output_never_overwrites(synth, monkeypatch):
    tmp, v_dir, f_dir = synth
    out = run_analysis(tmp, v_dir, f_dir, name="same")

    class FixedNow:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime.strptime(out.name, "%Y%m%d_%H%M%S")

    monkeypatch.setattr(ar, "datetime", FixedNow)
    with pytest.raises(FileExistsError):
        run_analysis(tmp, v_dir, f_dir, name="same")


def test_refuses_output_inside_run_dir(synth):
    tmp, v_dir, f_dir = synth
    with pytest.raises(ValueError, match="run 폴더 안"):
        ar.main(["--vanilla", str(v_dir), "--full", str(f_dir), "--out-root", str(f_dir / "x")])
    assert not (f_dir / "x").exists()


def test_missing_text_without_web_cache_is_not_guessed(synth):
    """web 원문이 없으면 'web 에 정답 없음' 이라고 단정하지 않고 C5 로 남긴다."""
    tmp, v_dir, f_dir = synth
    (f_dir / "web_cache.jsonl").unlink()
    out = run_analysis(tmp, v_dir, f_dir)
    cases = {r["query_id"]: r for r in read_csv(out / "regression_cases.csv")}
    s2 = set(cases["q2"]["signals"].split(";"))
    assert "C5_web_text_missing" in s2 and "C2_web_no_gold" not in s2
    assert "D2_text_missing" in s2 and cases["q2"]["provisional_bucket"].startswith("보류")


def test_no_record_texts_reports_unavailable(synth):
    tmp, v_dir, f_dir = synth
    rows = ar.read_jsonl(f_dir / "calls.jsonl")
    write_jsonl(f_dir / "calls.jsonl", [{**r, "output_text": None} for r in rows])
    out = run_analysis(tmp, v_dir, f_dir)
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["grade_format"] is None
    assert meta["availability"]["grade_mappable_reason"] == "출력 원문 미기록"
    assert "판정 불가" in (out / "report.md").read_text(encoding="utf-8")


def test_string_indicator_helpers():
    g = ar.gold_norms(["1", "The Beatles"])
    assert ar.has_gold("Founded in 2010.", ["1"]) is False  # 단어 경계 (부분 문자열 아님)
    assert ar.has_gold("Rank 1 overall", ["1"]) is True
    assert ar.has_gold("the beatles were a band", g) is True
    assert ar.has_gold(None, g) is None
    assert ar.any_gold([None, "x"], g) is None and ar.any_gold([None, "beatles"], g) is True
    assert ar.weak_gold(ar.gold_norms(["1"])) and not ar.weak_gold(g)
    assert ar.rewrite_recall("Who composed The Magic Flute?", "magic flute composer") == 2 / 3
    assert ar.score_band(0.7, 0.7, 0.3) == ">=0.7" and ar.score_band(None, 0.7, 0.3) == "not_graded"


def test_crag_pipeline_rows_without_graph_extra():
    """기존 crag pipeline 행 (extra.grading / web_doc_ids 없음) 도 같은 형태로 읽는다."""
    row = {
        "contexts": ["a", "w1"],
        "retrieval_doc_ids": ["a", "b"],
        "grader_scores": [0.5, 0.1],
        "extra": {"retrieval_scores": [0.9, 0.8]},
    }
    v = ar.full_view(row)
    assert v["scores"] == [0.5, 0.1] and v["kept_faiss"] == ["a"] and v["web_ids"] == ["w1"]


# ---------------------------------------------------------------- runner 결과 형식
class WebHurtsEngine(FakeEngine):
    def complete(self, request):
        r = super().complete(request)
        if request.kind == "generate" and "[1] web 0" in request.prompt:
            r.text = "unknown"
        return r


def test_reads_runner_output(setup, tmp_path):
    """Runner 가 실제로 쓰는 queries / calls / web_cache 형식으로 끝까지 동작하는지."""
    tmp, make = setup
    docs = tmp_path / "chunks.jsonl"
    write_jsonl(
        docs,
        [{"id": f"{t}{i}", "text": f"{t} doc {i} ans"} for t in ("even", "odd") for i in range(5)],
    )
    common = ["run.record_texts=true", f"retriever.docs_path={docs.as_posix()}"]
    cfg_v, _ = make(common + ["exp_id=SMOKE-V", "pipeline=vanilla"])
    v_dir = Runner(cfg_v, Components(Retriever(), Web(), FakeEngine()), run_id="r").run()
    rec = tmp_path / "web_rec.jsonl"
    cfg_f, _ = make(common + ["exp_id=SMOKE-F", "grading.strategy=full"])
    f_dir = Runner(
        cfg_f, Components(Retriever(), RecordingWebSearch(Web(), rec), WebHurtsEngine()), run_id="r"
    ).run()
    shutil.copy(rec, f_dir / "web_cache.jsonl")
    before = snapshot(v_dir, f_dir)
    out = run_analysis(tmp_path, v_dir, f_dir)
    assert snapshot(v_dir, f_dir) == before

    cases = read_csv(out / "regression_cases.csv")
    # 홀수 query: grader 0.1 -> incorrect -> web 문서만으로 'unknown'
    assert cases and all(c["action"] == "incorrect" for c in cases)
    for c in cases:
        s = set(c["signals"].split(";"))
        assert {
            "G2_gold_doc_below_lower",
            "E1_gold_ctx_removed",
            "C2_web_no_gold",
            "A2_refusal_like",
        } <= s
        assert c["provisional_bucket"].startswith("근거 제거 의심: Incorrect")
        assert c["retrieval_same"] == "True"
    meta = json.loads((out / "analysis_meta.json").read_text(encoding="utf-8"))
    assert meta["acc_vanilla"] == 1.0 and meta["n_regressions"] == len(cases)
    assert meta["availability"]["grade_mappable_reason"].startswith("호출 순서")
