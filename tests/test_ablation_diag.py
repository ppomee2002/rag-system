"""진단 실험 (대표 질문 선정 -> grading 원문 재실행, C0 / Ablation A / B -> 비교) 검증.

가짜 엔진·검색기·web 으로 만든 시나리오이며 실제 실험 결과가 아니다.
질문 i 의 유형 (CAT[i]) 에 따라 grading 출력과 문서 내용이 정해진다.

  I   : FAISS 3개 모두 정답 포함, grade 전부 "0"  -> Incorrect, web 에 정답 없음 -> 회귀
  I2  : 위와 같지만 grade "0.1" (0 이 아닌 점수)  -> Incorrect 회귀 (선정에서 제외돼야 함)
  C   : d0 정답 없음 0.9 / d1, d2 정답 0.5      -> Correct, d0 만 남아 회귀
  A   : d0 정답 없음 0.5 / d1, d2 정답 0         -> Ambiguous, d0 + web (정답 없음) -> 회귀
  K   : d0 정답 0.9 / d1, d2 0.2                 -> Correct, 정답 (대조군)
  W   : FAISS 정답 없음, grade 0, web 에 정답      -> Incorrect, Full 정답 / Vanilla 오답
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "crag_system"))
import compare_ablation as ca  # noqa: E402
import diag_grade_texts as dgt  # noqa: E402
import select_diag_queries as sel  # noqa: E402
from ablation import AblationCRAGGraph, parse_ablation  # noqa: E402
from crag_fixtures import FakeEngine, pool_ids  # noqa: E402
from run_ablation import AblationRunner  # noqa: E402
from runner import Components, Runner, build_web_search, load_layered_config  # noqa: E402
from workload import load_workload  # noqa: E402

from rag.graph import CRAGGraph  # noqa: E402
from rag.retrieval.web_cache import RecordingWebSearch  # noqa: E402
from rag.types import Document, QAExample  # noqa: E402

CAT = ["I"] * 9 + ["I2"] * 2 + ["C"] * 5 + ["A"] * 4 + ["K"] * 6 + ["W"] * 4
DIAG = ROOT / "experiments" / "crag_system" / "configs" / "diag"
Q_RE = re.compile(r"Question (\d+)\?")


def doc_text(i: int, j: int) -> str:
    c = CAT[i]
    has = c in ("I", "I2", "K") or (c in ("C", "A") and j > 0)
    return f"doc {i}-{j} {'ans' if has else 'none'}"


class ScenarioRetriever:
    def retrieve(self, query, top_k):
        i = int(Q_RE.search(query).group(1))
        return [Document(f"d{i}-{j}", doc_text(i, j), score=0.9 - 0.1 * j) for j in range(top_k)]


class ScenarioWeb:
    def search(self, query, top_k):
        i = int(query.split()[-1])
        tag = "ans" if CAT[i] == "W" else "none"
        return [Document(f"w{i}-{j}", f"web {i} {tag}") for j in range(top_k)]


def grade_text(i: int, j: int) -> str:
    return {
        "I": "0",
        "I2": "0.1",
        "C": "0.9" if j == 0 else "0.5",
        "A": "0.5" if j == 0 else "0",
        "K": "0.9" if j == 0 else "0.2",
        "W": "0",
    }[CAT[i]]


class ScenarioEngine(FakeEngine):
    def __init__(self, grade=grade_text):
        super().__init__(delay=0.0005)
        self.grade = grade

    def complete(self, request):
        r = super().complete(request)
        p = request.prompt
        i = int(Q_RE.search(p).group(1))
        if p.startswith("Is the following document relevant"):
            r.text = self.grade(i, int(re.search(r"doc \d+-(\d+)", p).group(1)))
        elif p.startswith("Rewrite the question"):
            r.text = f"search {i}"
        else:
            ctx = p.split("Context:")[1].split("\nQuestion:")[0]
            r.text = "ans" if re.search(r"\bans\b", ctx) else "unknown"
        return r


def rerun_grade(i, j):
    """재실행: K 의 d1 점수가 달라지고, I 의 d2 는 숫자 없는 출력 (점수는 같은 0.0)."""
    if CAT[i] == "K" and j == 1:
        return "0.3"
    if CAT[i] == "I" and j == 2:
        return "Not relevant"
    return grade_text(i, j)


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


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture
def world(setup, tmp_path):
    """Vanilla / 기존 Full (실제 검색 기록) run 과 진단용 config 생성 함수."""
    tmp, make = setup
    qa = tmp_path / "qa30.jsonl"
    qa.write_text(
        "".join(
            json.dumps({"id": f"q{i}", "question": f"Question {i}?", "answers": ["ans"]}) + "\n"
            for i in range(len(CAT))
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "".join(
            json.dumps({"id": f"d{i}-{j}", "text": doc_text(i, j)}) + "\n"
            for i in range(len(CAT))
            for j in range(3)
        ),
        encoding="utf-8",
    )
    common = [f"dataset.path={qa.as_posix()}", f"retriever.docs_path={chunks.as_posix()}"]
    cfg_v, _ = make(common + ["exp_id=SMOKE-V", "pipeline=vanilla"])
    v_dir = Runner(
        cfg_v, Components(ScenarioRetriever(), ScenarioWeb(), ScenarioEngine()), run_id="r"
    ).run()
    rec = tmp_path / "rec.jsonl"
    cfg_f, _ = make(common + ["exp_id=SMOKE-F", "grading.strategy=full"])
    f_dir = Runner(
        cfg_f,
        Components(ScenarioRetriever(), RecordingWebSearch(ScenarioWeb(), rec), ScenarioEngine()),
        run_id="r",
    ).run()
    shutil.copy(rec, f_dir / "web_cache.jsonl")

    def diag_cfg(exp_id, overlays=(), sets=()):
        """서버 명령과 같은 구성: --config <Full>/config.yaml + replay overlay (+ ablation)."""
        cfg = load_layered_config(
            f_dir / "config.yaml",
            [DIAG / "replay_full_retry.yaml", *[DIAG / o for o in overlays]],
            [f"crag.web_search.replay_from_runs=['{f_dir.as_posix()}']", *sets],
        )
        cfg["exp_id"] = exp_id
        return cfg

    def comps(cfg, engine=None):
        return Components(
            ScenarioRetriever(), build_web_search(cfg, tmp_path), engine or ScenarioEngine()
        )

    return {
        "tmp": tmp_path,
        "v": v_dir,
        "f": f_dir,
        "cfg_f": cfg_f,
        "diag_cfg": diag_cfg,
        "comps": comps,
    }


def test_full_diagnostic_flow(world):
    tmp, v_dir, f_dir = world["tmp"], world["v"], world["f"]
    diag_cfg, comps = world["diag_cfg"], world["comps"]
    before = snapshot(v_dir, f_dir)
    pool = pool_ids(world["cfg_f"])[1]
    cat = {q: CAT[int(q[1:])] for q in pool}

    # ---------------------------------------------------------------- 1. 대표 질문 선정
    wl_path = sel.main(
        [
            "--vanilla",
            str(v_dir),
            "--full",
            str(f_dir),
            "--name",
            "wl",
            "--out-dir",
            str(tmp / "wl"),
        ]
    )
    rows = read_csv(tmp / "wl" / "wl_selection.csv")
    strata = {}
    for r in rows:
        strata.setdefault(r["stratum"], []).append(r["query_id"])
    assert {k: len(v) for k, v in strata.items()} == {
        "inc_gold_removed": 2, "inc_zero": 4, "cor": 3, "amb": 2,
        "ctrl_correct": 2, "ctrl_incorrect": 2,
    }  # fmt: skip
    ids = [r["query_id"] for r in rows]
    assert len(ids) == 15 == len(set(ids))
    for k in ("inc_gold_removed", "inc_zero"):
        assert all(cat[q] == "I" for q in strata[k])  # I2 (0 아닌 점수) 는 선택 안 됨
    assert all(r["all_scores_zero"] == "True" for r in rows if r["stratum"].startswith("inc"))
    assert all(cat[q] == "C" for q in strata["cor"]) and all(cat[q] == "A" for q in strata["amb"])
    assert all(cat[q] == "K" for q in strata["ctrl_correct"])
    assert all(cat[q] == "W" for q in strata["ctrl_incorrect"])
    smeta = json.loads((tmp / "wl" / "wl_selection.json").read_text(encoding="utf-8"))
    assert smeta["n_incorrect_regression_nonzero_excluded"] == sum(c == "I2" for c in cat.values())
    wl = load_workload(wl_path)
    assert sorted(r["query_id"] for r in wl["requests"]) == sorted(ids)
    with pytest.raises(FileExistsError):  # 같은 이름 덮어쓰기 금지
        sel.main(["--vanilla", str(v_dir), "--full", str(f_dir), "--name", "wl",
                  "--out-dir", str(tmp / "wl")])  # fmt: skip

    # ---------------------------------------------------------------- 2. grading 원문 재실행
    cfg_d = diag_cfg("PRE-DIAG-GRADETEXT", sets=["run.record_texts=true"])
    d_dir = Runner(cfg_d, comps(cfg_d, ScenarioEngine(rerun_grade)), workload=wl, run_id="r").run()
    assert "/pre/PRE-DIAG-GRADETEXT/" in d_dir.as_posix()
    out = dgt.main(["--rerun", str(d_dir), "--full", str(f_dir), "--name", "g",
                    "--selection", str(tmp / "wl" / "wl_selection.csv"),
                    "--out-root", str(tmp / "an")])  # fmt: skip
    gm = json.loads((out / "diag_meta.json").read_text(encoding="utf-8"))
    assert gm["n_queries"] == 15 and gm["n_docs"] == 45 and gm["n_docs_with_text"] == 45
    assert gm["issue_counts"]["no_digit"] == 6  # 선택된 I 6개의 d2
    assert gm["score_equal"] == 45 - 2  # ctrl_correct (K) 2개의 d1 만 점수 다름
    assert gm["action_same"] == 15 and gm["reparse_mismatch"] == 0
    docs = read_csv(out / "grade_docs.csv")
    nd = [d for d in docs if "no_digit" in d["issues"]]
    assert all(d["score_equal"] == "True" and d["rerun_score"] == "0.0" for d in nd)
    assert {d["stratum"] for d in nd} <= {"inc_gold_removed", "inc_zero"}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "단정하지 않는다" in md and "재실행 출력에 대한 사실" in md

    # ---------------------------------------------------------------- 3. C0 / A / B
    cfg_c0 = diag_cfg("PRE-ABL-C0-FULL")
    c0 = Runner(cfg_c0, comps(cfg_c0), run_id="r").run()
    cfg_a = diag_cfg("PRE-ABL-A-INC-KEEP", ["abl_incorrect_keep_faiss.yaml"])
    a_dir = AblationRunner(cfg_a, comps(cfg_a), run_id="r").run()
    # B: 대조군 K 하나를 grading 0 으로 바꿔 Full 기록에 없는 correction (replay miss) 을 만든다
    miss_q = next(q for q in pool if cat[q] == "K")
    mi = int(miss_q[1:])
    cfg_b = diag_cfg("PRE-ABL-B-COR-KEEP", ["abl_correct_keep_topk.yaml"])
    miss_engine = ScenarioEngine(lambda i, j: "0" if i == mi else grade_text(i, j))
    b_dir = AblationRunner(cfg_b, comps(cfg_b, miss_engine), run_id="r").run()

    f_rows = {r["query_id"]: r for r in read_jsonl(f_dir / "queries.jsonl") if not r["warmup"]}
    c_rows = {r["query_id"]: r for r in read_jsonl(c0 / "queries.jsonl") if not r["warmup"]}
    a_rows = {r["query_id"]: r for r in read_jsonl(a_dir / "queries.jsonl") if not r["warmup"]}
    b_rows = {r["query_id"]: r for r in read_jsonl(b_dir / "queries.jsonl") if not r["warmup"]}
    # C0 는 기존 Full 을 재현
    assert {q: r["acc"] for q, r in c_rows.items()} == {q: r["acc"] for q, r in f_rows.items()}
    assert all(c_rows[q]["contexts"] == f_rows[q]["contexts"] for q in pool)
    # A: Incorrect 에 FAISS + web, 그 외는 C0 와 같은 context. 호출 수·action 동일
    for q in pool:
        a, c = a_rows[q], c_rows[q]
        assert a["corrective_action"] == c["corrective_action"]
        assert a["calls_summary"]["n_calls"] == c["calls_summary"]["n_calls"]
        assert a["pipeline"].endswith("+abl_inc_keep_faiss")
        assert a["extra"]["ablation"] == {"incorrect_keep_faiss": True, "correct_keep_topk": False}
        if c["corrective_action"] == "incorrect":
            assert a["contexts"] == a["retrieval_doc_ids"] + a["extra"]["web_doc_ids"]
        else:
            assert a["contexts"] == c["contexts"]
    # B: Correct 에 top-k 전부 (= Vanilla context)
    v_rows = {r["query_id"]: r for r in read_jsonl(v_dir / "queries.jsonl") if not r["warmup"]}
    for q in pool:
        if q == miss_q:
            assert "web cache miss" in b_rows[q]["error"]
            continue
        if c_rows[q]["corrective_action"] == "correct":
            assert b_rows[q]["contexts"] == v_rows[q]["contexts"]
            assert b_rows[q]["answer"] == v_rows[q]["answer"]
    end_b = json.loads((b_dir / "run_end.json").read_text(encoding="utf-8"))
    assert end_b["web_search"]["misses"] == 1

    # ---------------------------------------------------------------- 4. 비교
    out = ca.main(
        ["--reference", "c0", "--run", f"c0={c0}", "--run", f"abl_a={a_dir}",
         "--run", f"abl_b={b_dir}", "--run", f"full_retry={f_dir}", "--run", f"vanilla={v_dir}",
         "--bootstrap", "300", "--name", "t", "--out-root", str(tmp / "an")]
    )  # fmt: skip
    meta = json.loads((out / "compare_meta.json").read_text(encoding="utf-8"))
    common = [q for q in pool if q != miss_q]
    assert meta["n_common"] == len(common)
    assert meta["n_replay_problems"] == 1
    excl = read_csv(out / "excluded.csv")
    assert {(e["run"], e["query_id"], e["reason"]) for e in excl} >= {
        ("abl_b", miss_q, "replay miss (오류 행)"),
        ("c0", miss_q, "다른 run 에 정상 결과가 없어 공통 질문에서 제외"),
    }
    rep = {r["run"]: r for r in read_csv(out / "replay.csv")}
    assert rep["abl_b"]["misses"] == "1" and rep["abl_b"]["n_error_replay_miss"] == "1"
    assert rep["c0"]["misses"] == "0" and rep["full_retry"]["provider"] == "tavily"

    summ = {r["run"]: r for r in read_csv(out / "summary.csv")}
    n = len(common)

    def acc(cats):
        return sum(cat[q] in cats for q in common) / n

    assert float(summ["c0"]["accuracy"]) == pytest.approx(acc({"K", "W"}), abs=1e-6)
    assert float(summ["abl_a"]["accuracy"]) == pytest.approx(acc({"K", "W", "I", "I2"}), abs=1e-6)
    assert float(summ["abl_b"]["accuracy"]) == pytest.approx(acc({"K", "W", "C"}), abs=1e-6)
    assert float(summ["vanilla"]["accuracy"]) == pytest.approx(1 - acc({"W"}), abs=1e-6)
    assert summ["abl_a"]["llm_calls_mean"] == summ["c0"]["llm_calls_mean"]
    paired = {r["pair"]: r for r in read_csv(out / "paired.csv")}
    pa = paired["abl_a - c0"]
    d = acc({"I", "I2"})
    assert float(pa["acc_diff"]) == pytest.approx(d, abs=1e-6)
    assert float(pa["acc_ci95_low"]) <= d <= float(pa["acc_ci95_high"])
    assert int(pa["ref_wrong_run_right"]) == round(d * n) and pa["ref_right_run_wrong"] == "0"
    assert float(pa["gen_tokens_mean_diff"]) > 0  # Incorrect 에 FAISS 문서가 추가됨
    assert float(pa["llm_calls_mean_diff"]) == 0
    assert float(paired["full_retry - c0"]["acc_diff"]) == 0
    bap = {(r["run"], r["ref_action"]): r for r in read_csv(out / "by_action_paired.csv")}
    inc = bap[("abl_a", "incorrect")]
    assert int(inc["n"]) == sum(cat[q] in ("I", "I2", "W") for q in common)
    assert float(inc["acc_run"]) == 1.0 and float(bap[("abl_a", "correct")]["acc_diff"]) == 0
    assert float(bap[("abl_b", "correct")]["acc_run"]) == 1.0
    own = {(r["run"], r["own_action"]) for r in read_csv(out / "by_action_own.csv")}
    assert ("abl_b", "correct") in own and not any(r == "vanilla" for r, _ in own)
    chs = {r["run"]: r for r in meta["action_change_summary"]}
    assert chs["abl_a"]["n_action_changed"] == 0 and chs["abl_a"]["n_grade_scores_differ"] == 0
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "replay miss / 오류가 1건" in md and miss_q in md

    assert snapshot(v_dir, f_dir) == before  # 기존 run 폴더는 읽기만


def test_action_changes_listed():
    ref = {"q1": {"corrective_action": "correct", "acc": 1, "grader_scores": [0.9]},
           "q2": {"corrective_action": "incorrect", "acc": 0, "grader_scores": [0.0]}}  # fmt: skip
    run = {"q1": {"corrective_action": "ambiguous", "acc": 0, "grader_scores": [0.5]},
           "q2": {"corrective_action": "incorrect", "acc": 0, "grader_scores": [0.0]}}  # fmt: skip
    ch = ca.action_changes(ref, run, ["q1", "q2"])
    got = [(c["query_id"], c["ref_action"], c["run_action"], c["grade_scores_equal"]) for c in ch]
    assert got == [("q1", "correct", "ambiguous", False)]


def test_parse_ablation_rejects_bad_config():
    for cfg in ({}, {"ablation": {}}, {"ablation": {"incorrect_keep_faiss": False}},
                {"ablation": {"keep_all": True}}, {"ablation": {"correct_keep_topk": "yes"}},
                {"pipeline": "crag", "ablation": {"correct_keep_topk": True}}):  # fmt: skip
        with pytest.raises(ValueError):
            parse_ablation(cfg)
    assert parse_ablation({"ablation": {"correct_keep_topk": True}}) == {
        "incorrect_keep_faiss": False,
        "correct_keep_topk": True,
    }


def test_ablation_runner_validates_before_creating_run_dir(world):
    cfg = world["diag_cfg"]("PRE-ABL-BAD")
    with pytest.raises(ValueError, match="ablation"):
        AblationRunner(cfg, world["comps"](cfg), run_id="r")
    assert not (world["tmp"] / "results" / "crag_system" / "pre" / "PRE-ABL-BAD").exists()


def test_ablation_graph_with_flags_off_matches_crag_graph():
    """ablation 을 모두 끈 그래프는 기존 CRAGGraph 와 같은 결과.

    하위 클래스가 다른 동작을 바꾸지 않는지 확인한다.
    """
    args = (ScenarioRetriever(), None, ScenarioWeb())
    kw = {"top_k": 3, "web_top_k": 2}
    base = CRAGGraph(*args, **kw)
    off = AblationCRAGGraph.from_graph(
        base, {"incorrect_keep_faiss": False, "correct_keep_topk": False}
    )
    from rag.llm.engine_llm import EngineLLM

    llm = EngineLLM(ScenarioEngine())
    for i, c in enumerate(CAT):
        ex = QAExample(f"q{i}", f"Question {i}?", ["ans"])
        r0, r1 = base.run(ex, llm=llm), off.run(ex, llm=llm)
        assert [d.id for d in r0.contexts] == [d.id for d in r1.contexts], c
        assert (r0.answer, r0.corrective_action, r0.grader_scores) == (
            r1.answer,
            r1.corrective_action,
            r1.grader_scores,
        )
        assert set(r0.stage_latencies) == set(r1.stage_latencies)


def test_diag_overlays():
    rp = yaml.safe_load((DIAG / "replay_full_retry.yaml").read_text(encoding="utf-8"))
    ws = rp["crag"]["web_search"]
    assert ws["provider"] == "replay" and ws["replay_key"] == "query_id"
    assert ws["replay_on_miss"] == "error" and ws["replay_fallback"] is False
    assert ws["replay_from_runs"] == [
        "results/crag_system/smoke/SMOKE-A1-FULL-RETRY/20261110_111726"
    ]
    assert rp["run"]["purpose"] == "pre"
    for f, key in (("abl_incorrect_keep_faiss.yaml", "incorrect_keep_faiss"),
                   ("abl_correct_keep_topk.yaml", "correct_keep_topk")):  # fmt: skip
        assert parse_ablation(yaml.safe_load((DIAG / f).read_text(encoding="utf-8")))[key]


def test_selection_shortage_fails(world, tmp_path):
    """후보가 부족하면 다른 층으로 채우지 않고 실패한다."""
    van = sel.load_run("vanilla", world["v"])
    full = sel.load_run("full", world["f"])
    cand, _ = sel.candidates(van, full, {})  # 원문 없음 -> inc_gold_removed 후보 0
    assert cand["inc_gold_removed"] == [] and cand["inc_zero"]
    _, shortage = sel.select(cand, 0)
    assert shortage == {"inc_gold_removed": {"need": 2, "got": 0}}
    with pytest.raises(SystemExit, match="후보 부족"):
        sel.main(["--vanilla", str(world["v"]), "--full", str(world["f"]), "--name", "x",
                  "--docs-path", str(tmp_path / "none.jsonl"),
                  "--out-dir", str(tmp_path / "w")])  # fmt: skip
    assert not (tmp_path / "w" / "x.json").exists()
