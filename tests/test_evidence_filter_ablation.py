"""evidence_filter_ablation.py 기능 검증 (가짜 엔진, 실제 성능 결과 아님).

가짜 엔진 규칙: grade 는 "even doc 0" 만 0.9 (나머지 0.1) -> even 질문 correct (doc0 만 유지),
odd 질문 incorrect (FAISS 전부 폐기). generate 는 context 에 "doc 2" 가 있을 때만 정답 ("ans").
따라서 crag_rule 은 모두 오답, 필터를 끈 조건은 해당 action 질문만 정답이 된다.
"""

import json
from dataclasses import replace

import evidence_filter_ablation as efa
import numpy as np
import pytest
from crag_fixtures import FakeEngine, Retriever, Web
from runner import Components, Runner

from rag.retrieval.web_cache import RecordingWebSearch, read_jsonl
from rag.telemetry.calls import infer_call_kind
from rag.types import Document


class DocEngine(FakeEngine):
    def __init__(self):
        super().__init__(delay=0.0)
        self.n_generate = 0

    def complete(self, request):
        r = super().complete(request)
        kind = request.kind if request.kind != "other" else infer_call_kind(request.prompt)
        if kind == "grade":
            return replace(r, text="0.9" if "even doc 0" in request.prompt else "0.1")
        if kind == "generate":
            self.n_generate += 1
            return replace(r, text="ans" if "doc 2" in request.prompt else "nope")
        return r


@pytest.fixture
def ref(setup):
    tmp, make = setup
    web_log = tmp / "web_cache_rec.jsonl"
    cfg, _ = make()
    run = Runner(
        cfg, Components(Retriever(), RecordingWebSearch(Web(), web_log), DocEngine()), run_id="ref"
    ).run()
    docs = tmp / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as f:
        for tag in ("even", "odd"):
            for i in range(3):
                f.write(json.dumps({"id": f"{tag}{i}", "text": f"{tag} doc {i}"}) + "\n")
    return run, docs, web_log


def test_build_context_matches_rules():
    docs = [Document(id=f"d{i}", text="") for i in range(3)]
    web = [Document(id="w0", text="")]
    sc = [0.9, 0.5, 0.1]
    ids = lambda c, a: [d.id for d in efa.build_context(c, a, docs, sc, web, 0.7, 0.3)]  # noqa: E731
    assert ids("crag_rule", "correct") == ["d0"]
    assert ids("crag_rule", "ambiguous") == ["d0", "d1", "w0"]
    assert ids("crag_rule", "incorrect") == ["w0"]
    assert ids("no_correct_filter", "correct") == ["d0", "d1", "d2"]
    assert ids("no_correct_filter", "incorrect") == ["w0"]
    assert ids("no_incorrect_discard", "incorrect") == ["d0", "d1", "d2", "w0"]
    assert ids("no_ambiguous_filter", "ambiguous") == ["d0", "d1", "d2", "w0"]
    assert ids("keep_all", "correct") == ["d0", "d1", "d2"]
    assert ids("vanilla_ctx", "incorrect") == ["d0", "d1", "d2"]
    assert efa.mcnemar_exact(0, 0) is None
    assert efa.mcnemar_exact(0, 5) == pytest.approx(2 / 32)


def test_generate_and_analyze(ref, tmp_path):
    run, docs, web_log = ref
    eng = DocEngine()
    out = efa.main(
        ["generate", "--ref", str(run), "--split", "all", "--exp-id", "PRE-EVF-T",
         "--docs-path", str(docs), "--web-cache", str(web_log), "--output-dir", str(tmp_path),
         "--out-root", str(tmp_path / "ana"), "--n-boot", "200", "--quiet"],
        engine=eng,
    )  # fmt: skip
    items = read_jsonl(out / "items.jsonl")
    assert items and all(not it["problems"] for it in items)  # 재구성 = 기준 run
    gens = read_jsonl(out / "generations.jsonl")
    n_q = len(items)
    assert len(gens) == n_q * len(efa.CONDITIONS)
    # 같은 context 는 generation 공유. 고유 context: correct 2개 (doc0 / top-k),
    # incorrect 3개 (web / top-k + web / top-k)
    n_inc = sum(it["action"] == "incorrect" for it in items)
    assert 0 < n_inc < n_q
    assert eng.n_generate == 2 * (n_q - n_inc) + 3 * n_inc
    ana = sorted((tmp_path / "ana" / "evidence_filter").glob("*/*/analysis.json"))[-1]
    res = json.loads(ana.read_text(encoding="utf-8"))
    s = {r["condition"]: r for r in res["summary"]}
    assert s["crag_rule"]["acc"] == 0.0 and s["crag_rule"]["ref_answer_same"] == 1.0
    assert s["vanilla_ctx"]["acc"] == 1.0 and s["keep_all"]["acc"] == 1.0
    assert s["vanilla_ctx"]["n_web"] == 0.0
    assert len({r["correction_rate"] for r in res["summary"]}) == 1  # 조건 간 동일
    share_inc = s["crag_rule"]["correction_rate"]
    assert s["no_incorrect_discard"]["acc"] == pytest.approx(share_inc)
    assert s["no_correct_filter"]["acc"] == pytest.approx(1 - share_inc)
    # 단일 규칙 효과의 합 = keep_all 효과 (action 별로 분리)
    p = {(r["target"], r["subset"]): r for r in res["paired"] if r["base"] == "crag_rule"}
    total = sum(p[(c, "all")]["diff"] for c in
                ("no_correct_filter", "no_ambiguous_filter", "no_incorrect_discard"))  # fmt: skip
    assert total == pytest.approx(p[("keep_all", "all")]["diff"])
    assert p[("no_correct_filter", "incorrect")]["diff"] == 0.0
    md = sorted((tmp_path / "ana" / "evidence_filter").glob("*/*/report.md"))[-1]
    assert "paired" in md.read_text(encoding="utf-8")


def test_resume_and_guards(ref, tmp_path):
    run, docs, web_log = ref
    base = ["--ref", str(run), "--docs-path", str(docs), "--web-cache", str(web_log),
            "--output-dir", str(tmp_path), "--out-root", str(tmp_path / "ana"), "--quiet",
            "--n-boot", "50"]  # fmt: skip
    with pytest.raises(SystemExit):  # split 불일치 (누수 방지)
        efa.main(["generate", *base, "--split", "eval", "--exp-id", "PRE-X"], engine=DocEngine())
    with pytest.raises(SystemExit):  # PRE-* 만
        efa.main(["generate", *base, "--split", "all", "--exp-id", "SYS-1"], engine=DocEngine())
    out = efa.main(["generate", *base, "--split", "all", "--exp-id", "PRE-R", "--limit", "2"],
                   engine=DocEngine())  # fmt: skip
    n1 = len(read_jsonl(out / "generations.jsonl"))
    eng = DocEngine()
    efa.main(["generate", *base, "--split", "all", "--resume", str(out)], engine=eng)
    gens = read_jsonl(out / "generations.jsonl")
    keys = [(g["query_id"], g["condition"]) for g in gens]
    assert len(keys) == len(set(keys)) and len(gens) > n1  # 이어서, 중복 없음
    # 점수를 바꾼 기준 run 은 재구성 검증에서 제외
    rows = read_jsonl(run / "queries.jsonl")
    for r in rows:
        if not r.get("warmup"):
            r["extra"]["grading"]["scores_full"] = [0.5] * len(r["extra"]["grading"]["scores_full"])
            break
    bad = tmp_path / "badref"
    bad.mkdir()
    (bad / "config.yaml").write_bytes((run / "config.yaml").read_bytes())
    (bad / "queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    items, _ = efa.load_items(bad, "all", str(docs), (str(web_log),))
    assert sum(1 for it in items if it["problems"]) == 1
    assert np.isclose(efa.paired_acc(np.array([1, 0, 1.0]), np.array([1, 1, 1.0]), 100)["diff"],
                      1 / 3)  # fmt: skip
