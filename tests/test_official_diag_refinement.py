"""experiments/official_crag/diag_refinement.py.

순수 함수 + (공식 저장소가 있으면) 실제 데이터 검사.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "official_crag"))
import diag_refinement as dr  # noqa: E402

REPO = Path(os.environ.get("CRAG_OFFICIAL_REPO", ROOT / "third_party" / "CRAG"))
has_repo = pytest.mark.skipif(
    not (REPO / "data" / "popqa" / "ref" / "correct").exists(),
    reason="공식 저장소 없음 (fetch_official.py repo)",
)

PS = [" Alpha one two three.", "", "Beta four five six ", " Gamma seven eight nine.", ""]


def test_ordered_subset_space_join():
    assert dr.ordered_subset(PS, "Alpha one two three.  Gamma seven eight nine.") == [0, 3]
    assert dr.ordered_subset(PS, "Beta four five six") == [2]
    # 순서가 다르면 검색 순서 부분집합이 아니다
    assert dr.ordered_subset(PS, "Gamma seven eight nine. Alpha one two three.") is None
    # '; ' 결합은 공백 결합이 아니다
    assert dr.ordered_subset(PS, "Alpha one two three.; Beta four five six") is None
    assert dr.ordered_subset(PS, "") is None


def test_semicolon_parts():
    assert dr.semicolon_parts(PS, "Gamma seven eight nine.; Alpha one two three.") == [3, 0]
    assert dr.semicolon_parts(PS, "Alpha one two three.  Gamma seven eight nine.") is None


def test_classify():
    base = dict(
        ours_text="A; B; C",
        official_text="A; B; C",
        ours_top=[1, 2, 3],
        official_top=[1, 2, 3],
        score_max_abs_diff=0.0,
    )
    c = dr.classify(**base, ref_subset=[1, 2, 3], ref_text="A; B; C")
    assert c == {"impl": [], "ref": [], "impl_exact": True, "ref_exact": True}
    # 같은 문서·같은 순서, 결합자만 다름
    assert dr.classify(**base, ref_subset=[1, 2, 3], ref_text="A B C")["ref"] == ["format"]
    # 같은 집합, 검색 순서
    assert dr.classify(**base, ref_subset=[3, 1, 2], ref_text="C A B")["ref"] == ["ordering"]
    # 다른 문서 (개수 다름)
    assert dr.classify(**base, ref_subset=[1], ref_text="A")["ref"] == ["selection"]
    assert dr.classify(**base, ref_subset=None, ref_text="Z")["ref"] == ["source"]
    # 재구현 쪽 점수 차이 + 선택 차이
    c = dr.classify(
        **{**base, "ours_top": [1, 2, 4], "ours_text": "A; B; D", "score_max_abs_diff": 0.1},
        ref_subset=[1, 2, 3],
        ref_text="A; B; C",
    )
    assert c["impl"] == ["score", "selection"] and not c["impl_exact"]


def test_rule_check():
    rows = [
        {"scores": [0.9, -2.0, 0.1, -0.5], "kept": [0, 2], "nonempty": [0, 1, 2, 3]},
        {"scores": [0.3, 0.8, -0.2, -3.0], "kept": [0, 1], "nonempty": [0, 1, 2, 3]},
    ]
    r = dr.rule_check(rows)
    assert r["match_counts"]["topk_set"] == 2
    assert r["match_counts"]["top3_set"] == 0
    assert r["match_counts"]["thr_zero"] == 2
    g = r["global_threshold"]
    assert g["exists"] and g["interval_open_closed"] == [-0.2, 0.1]
    rows.append({"scores": [0.0, 1.0], "kept": [0], "nonempty": [0, 1]})  # 분리 불가
    assert not dr.rule_check(rows)["global_threshold"]["exists"]


@has_repo
def test_provenance_on_official_data(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_run_dir", lambda run_id: tmp_path / "run")
    (tmp_path / "run").mkdir()
    run = dr.main(["provenance", "--repo", str(REPO)])
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["n"] == 1399
    # 저자 ref/correct: 모든 줄이 검색 순서 부분집합의 공백 결합 (공개 코드의 top-3 '; ' 형식 아님)
    assert s["counts"]["ordered_subset_space_join"] == 1399
    assert s["counts"]["hash_prefix"] == 1399
    assert s["ambiguous_knowledge1_eq_correct"] == 1399
    assert s["popqa_retrieved_psgs_published"] is False
    assert sum(s["n_kept_distribution"].values()) == 1399


@has_repo
def test_recording_model_with_official_select_relevants():
    torch = pytest.importorskip("torch")
    utils = dr.load_official_utils(REPO)

    class Tok:
        def __init__(self):
            self.texts = []

        def __call__(self, text, **kw):
            self.texts.append(text)
            return {
                "input_ids": torch.tensor([[len(self.texts) - 1]]),
                "attention_mask": torch.ones(1, 1),
            }

    tok = Tok()
    vals = {"q [SEP] one two three four": 0.5, "q [SEP] five six seven eight": 1.5}

    class M:
        def to(self, d):
            return self

        def __call__(self, ids, attention_mask=None):
            return {"logits": torch.tensor([[vals[tok.texts[int(ids[0, 0])]]]])}

    rec = dr.RecordingModel(M())
    text, idx = utils.select_relevants(
        ["one two three four", "x y", "five six seven eight"], "q", tok, rec, "cpu", 3
    )
    assert rec.logits == [0.5, 1.5]  # 짧은 strip 은 모델을 부르지 않는다
    assert idx == "2, 0, 1" and text == "five six seven eight; one two three four; x y"
