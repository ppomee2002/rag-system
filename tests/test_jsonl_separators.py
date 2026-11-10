"""comparability / compare_quality / calibrate_k 의 JSONL 읽기 회귀 테스트.

LLM 출력·검색 본문에 U+2028 등 (str.splitlines 가 줄로 오인하는 문자) 이 있어도 결과 파일을
읽을 수 있어야 하고, 실제로 손상된 줄은 무시하지 않고 경로와 줄 번호로 보고해야 한다.
"""

import json
import shutil

import calibrate_k
import pytest
from comparability import compare_runs
from compare_quality import _grade_format, compare_quality
from crag_fixtures import FakeEngine, Retriever, Web
from runner import Components, Runner

SEPARATORS = "  \x85\x1c\x1d\x1e\x0b\x0c"


class SeparatorEngine(FakeEngine):
    """모든 출력 끝에 줄 구분 유니코드 문자를 붙인다 (파싱·정답 판정에는 영향 없음)."""

    def complete(self, request):
        r = super().complete(request)
        r.text = r.text + SEPARATORS
        return r


@pytest.fixture
def two_runs(setup):
    _, make = setup
    out = []
    for rid in ("a", "b"):
        cfg, _ = make(["exp_id=PRE-Q", "run.purpose=pre", "run.record_texts=true"])
        out.append(Runner(cfg, Components(Retriever(), Web(), SeparatorEngine()), run_id=rid).run())
    raw = (out[0] / "queries.jsonl").read_text(encoding="utf-8")
    assert " " in raw and len(raw.splitlines()) > raw.count("\n")  # 이전 구현이 깨지던 조건
    return out


def corrupt_copy(src, dst, name):
    shutil.copytree(src, dst)
    n = sum(1 for _ in open(dst / name, encoding="utf-8"))
    with open(dst / name, "a", encoding="utf-8") as f:
        f.write('{"broken": \n')
    return n + 1


def test_comparability_reads_separator_text_and_reports_corruption(two_runs, tmp_path):
    rep = compare_runs(two_runs)
    assert rep["ok"] and rep["n_common_requests"] == 10
    bad = tmp_path / "bad"
    line = corrupt_copy(two_runs[1], bad, "queries.jsonl")
    with pytest.raises(ValueError, match=rf"queries\.jsonl:{line}: JSON"):
        compare_runs([two_runs[0], bad])


def test_compare_quality_reads_separator_text_and_reports_corruption(two_runs, tmp_path):
    rep = compare_quality(str(two_runs[0]), str(two_runs[1]))
    assert rep["prerequisites_ok"] and rep["n_paired"] == 10
    assert rep["answer_agreement"] == 1.0
    fmt = _grade_format(str(two_runs[0]))
    assert fmt["n"] > 0 and fmt["no_number_rate"] == 0.0
    bad = tmp_path / "bad"
    line = corrupt_copy(two_runs[0], bad, "calls.jsonl")
    with pytest.raises(ValueError, match=rf"calls\.jsonl:{line}: JSON"):
        _grade_format(str(bad))


def test_calibrate_row_for_reads_separator_text_and_reports_corruption(two_runs, tmp_path):
    row = calibrate_k.row_for(2, two_runs[0])
    assert row["K"] == 2 and row["n_errors"] == 0 and row["llm_calls_per_s"] > 0
    bad = tmp_path / "bad"
    line = corrupt_copy(two_runs[0], bad, "calls.jsonl")
    with pytest.raises(ValueError, match=rf"calls\.jsonl:{line}: JSON"):
        calibrate_k.row_for(2, bad)


def test_separator_rows_round_trip_through_all_readers(two_runs):
    """세 도구가 같은 공용 reader 를 써서 같은 행 수를 읽는다."""
    from rag.retrieval.web_cache import read_jsonl

    n = len(read_jsonl(two_runs[0] / "queries.jsonl"))
    assert n == 12  # warmup 2 + 측정 10
    calls = [json.loads(x) for x in open(two_runs[0] / "calls.jsonl", encoding="utf-8")]
    assert len(read_jsonl(two_runs[0] / "calls.jsonl")) == len(calls)
