"""analyze_runs.py 기능 검증 (가짜 엔진으로 만든 run 폴더, 실제 결과 아님)."""

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

from rag.retrieval.web_cache import RecordingWebSearch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import analyze_runs  # noqa: E402
from crag_fixtures import FakeEngine, Retriever, Web  # noqa: E402
from runner import Components, Runner  # noqa: E402

from rag.types import Document  # noqa: E402

# str.splitlines() 는 나누지만 JSON Lines 구분자는 아닌 문자 (실제 Tavily 본문에 나올 수 있음)
SEPARATORS = "\u2028\u2029\x85\x1c\x1d\x1e\x0b\x0c"


class SeparatorWeb(Web):
    """본문에 줄 구분 유니코드 문자가 들어 있는 web 검색 결과."""

    def search(self, query, top_k):
        self.n += 1
        return [Document(id=f"w{i}", text=f"web {i}{SEPARATORS}second part") for i in range(top_k)]


class WebHurtsEngine(FakeEngine):
    """web 문서만으로 생성하면 오답 (Incorrect 경로 회귀를 흉내)."""

    def complete(self, request):
        r = super().complete(request)
        if request.kind == "generate" and "[1] web 0" in request.prompt:
            r.text = "unknown"
        return r


@pytest.fixture
def runs(setup, tmp_path):
    tmp, make = setup
    docs = tmp_path / "chunks.jsonl"
    docs.write_text(
        "".join(
            json.dumps({"id": f"{t}{i}", "text": f"{t} doc {i} ans"}) + "\n"
            for t in ("even", "odd")
            for i in range(5)
        ),
        encoding="utf-8",
    )
    common = ["run.record_texts=true", f"retriever.docs_path={docs.as_posix()}"]
    rec = tmp_path / "web_cache.jsonl"
    specs = {
        "vanilla": (["exp_id=SMOKE-V", "pipeline=vanilla"], Web(), FakeEngine()),
        "full": (
            ["exp_id=SMOKE-F", "grading.strategy=full"],
            RecordingWebSearch(SeparatorWeb(), rec),
            WebHurtsEngine(),
        ),
        "lazy_hit": (["exp_id=SMOKE-L", "grading.strategy=lazy_hit"], Web(), WebHurtsEngine()),
        "batch": (["exp_id=SMOKE-B", "grading.strategy=batch"], Web(), WebHurtsEngine(delay=0.004)),
    }
    out = {}
    for name, (sets, web, eng) in specs.items():
        cfg, _ = make(common + sets)
        out[name] = Runner(cfg, Components(Retriever(), web, eng), run_id="r").run()
    shutil.copy(rec, out["full"] / "web_cache.jsonl")
    return tmp, out


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_analyze_runs_end_to_end(runs, tmp_path):
    tmp, dirs = runs
    base = tmp / "results" / "crag_system" / "smoke"
    args = [
        "--base",
        str(base),
        "--baseline",
        "vanilla",
        "--reference",
        "full",
        "--bootstrap",
        "500",
        "--out-root",
        str(tmp_path / "analysis"),
        "--name",
        "t",
    ]
    for k, d in dirs.items():
        args += ["--run", f"{k}={d.relative_to(base).as_posix()}"]
    out = analyze_runs.main(args)
    for f in (
        "accuracy.csv",
        "paired_accuracy.csv",
        "latency.csv",
        "paired_latency.csv",
        "stages.csv",
        "calls_by_kind.csv",
        "transitions.csv",
        "regression_cases.csv",
        "per_query.csv",
        "runs.json",
        "regression.json",
        "batch_diagnosis.json",
        "report.md",
    ):
        assert (out / f).exists(), f
    accs = {r["run"]: r for r in read_csv(out / "accuracy.csv")}
    n_inc = int(accs["full"]["incorrect"])  # web 만으로 생성한 query 는 오답
    assert n_inc > 0 and float(accs["vanilla"]["accuracy"]) == 1.0
    assert float(accs["full"]["accuracy"]) == pytest.approx(1 - n_inc / 10)
    pa = {r["pair"]: r for r in read_csv(out / "paired_accuracy.csv")}
    full_v = pa["full - vanilla"]
    d = -n_inc / 10
    assert float(full_v["diff"]) == pytest.approx(d)
    assert float(full_v["ci95_low"]) <= d <= float(full_v["ci95_high"])
    assert int(full_v["a_right_b_wrong"]) == n_inc and full_v["a"] == "vanilla"
    assert "lazy_hit - full" in pa and "batch - full" in pa and "batch - lazy_hit" in pa
    lat = {r["pair"]: r for r in read_csv(out / "paired_latency.csv")}
    assert float(lat["full - vanilla"]["mean_diff"]) > 0
    reg = json.loads((out / "regression.json").read_text(encoding="utf-8"))
    assert reg["n_regressions"] == n_inc and reg["gold_check_available"]
    assert reg["classes"] == {"Incorrect: FAISS 문서 폐기 + web 근거에 정답 없음": n_inc}
    assert reg["grade_output_format"]["number_only_rate"] == 1.0
    kinds = {(r["run"], r["kind"]) for r in read_csv(out / "calls_by_kind.csv")}
    assert ("batch", "grade_batch") in kinds and ("full", "grade") in kinds
    assert ("vanilla", "grade") not in kinds
    bd = json.loads((out / "batch_diagnosis.json").read_text(encoding="utf-8"))
    assert bd["per_query_mean"]["n_calls"] == {"full": 3.0, "batch": 1.0}  # fixture top_k=3
    assert bd["findings"]
    meta = json.loads((out / "runs.json").read_text(encoding="utf-8"))
    assert meta["n_common_queries"] == 10 and meta["runs"]["batch"]["grading"] == "batch"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "paired bootstrap" in md and "Batch grading" in md


def test_preset_paths_and_missing_run(tmp_path):
    p = analyze_runs.PRESETS["a-smoke"]
    assert p["runs"]["full"] == "SMOKE-A1-FULL-RETRY/20261110_111726"
    assert "SMOKE-A1-FULL" in p["excluded"] and "SMOKE-A1-CHECK" in p["excluded"]
    with pytest.raises(FileNotFoundError):
        analyze_runs.main(
            ["--preset", "a-smoke", "--base", str(tmp_path), "--out-root", str(tmp_path / "o")]
        )


def test_bootstrap_ci_helpers():
    assert analyze_runs.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    idx = analyze_runs.boot_idx(4, 10, 0)
    assert idx.shape == (10, 4) and idx.max() < 4


def test_read_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"text": f"a{SEPARATORS}b"}, {"text": "plain"}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert len(p.read_text(encoding="utf-8").splitlines()) > 2  # 이전 구현이 깨지던 조건
    assert analyze_runs.read_jsonl(p) == rows


def test_read_jsonl_reports_path_and_line_on_real_corruption(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\n{"broken": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        analyze_runs.read_jsonl(p)
