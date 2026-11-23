"""experiments/official_crag 스크립트 테스트 (가짜 데이터·mock evaluator·가짜 엔진).

정확도·성능 결과가 아니라 단계 간 연결 (파일 형식, key 일치, 검사 실패 처리) 만 확인한다.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

from rag.crag_official.evaluator import MockScoreBackend, OfficialRetrievalEvaluator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "official_crag"))
sys.path.insert(0, str(ROOT / "tests"))
import analyze  # noqa: E402
import freeze_inputs  # noqa: E402
import mode_a  # noqa: E402
import precompute_scores  # noqa: E402
from mode_b import Components, ModeBRunner  # noqa: E402
from test_evidence_pipeline import FakeEngine, _cfg, level  # noqa: E402


def mock_fn(q, p):
    return {"GOOD": 1.0, "MID": 0.0, "BAD": -2.0}[level(p)]


class NamedMock(MockScoreBackend):
    model_path = "mock-path"


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


# i % 3: 0 -> incorrect (전부 BAD), 1 -> correct (GOOD 포함), 2 -> ambiguous (MID 만)
DOCS_BY_KIND = {
    0: ["c2", "c3", "c5", "c6", "c7"],
    1: ["c1", "c0", "c2", "c3", "c4"],
    2: ["c2", "c1", "c3", "c4", "c5"],
}


@pytest.fixture
def ref_run(tmp_path):
    """crag_system 형식 기준 run + dataset + chunk 파일."""
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(
        data / "qa.jsonl",
        [
            {"id": str(i), "question": f"Q{i}? Paris", "answers": ["Paris"], "s_pop": 5}
            for i in range(6)
        ],
    )
    texts = [
        "GOOD chunk Paris\nline",
        "MID chunk text here",
        "BAD",
        "BAD two",
        "MID other text again",
        "BAD three",
        "BAD four",
        "BAD five",
    ]
    chunks = [{"id": f"c{j}", "text": t} for j, t in enumerate(texts)]
    write_jsonl(data / "chunks.jsonl", chunks)
    run = tmp_path / "ref_run"
    run.mkdir()
    rows = [
        {
            "query_id": str(i),
            "question": f"Q{i}? Paris",
            "gold_answers": ["Paris"],
            "warmup": False,
            "retrieval_doc_ids": DOCS_BY_KIND[i % 3],
            "extra": {"retrieval_scores": [0.9, 0.8, 0.7, 0.6, 0.5], "rewritten_query": f"rw {i}"},
        }
        for i in range(6)
    ]
    write_jsonl(run / "queries.jsonl", rows)
    write_jsonl(
        run / "web_cache.jsonl",
        [
            {
                "query": f"rw {i}",
                "top_k": 5,
                "latency": 0.0,
                "docs": [
                    {
                        "id": f"u{i}",
                        "text": "web GOOD paragraph Paris text\nanother MID paragraph text",
                        "metadata": {"title": "W"},
                    }
                ],
            }
            for i in range(6)
        ],
    )
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(data / "qa.jsonl")},
                "retriever": {"docs_path": str(data / "chunks.jsonl"), "top_k": 5},
                "embedding": {"model": "x"},
            }
        )
    )
    return run


def make_frozen(tmp_path, ref_run, ids=("1", "2", "3", "4", "5")):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids))
    out = tmp_path / "inputs" / "frozen.jsonl"
    freeze_inputs.main(
        ["--question-ids", str(ids_file), "--reference-run", str(ref_run), "--out", str(out)]
    )
    return out


def test_freeze_inputs(tmp_path, ref_run):
    out = make_frozen(tmp_path, ref_run)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["question_id"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert [d["id"] for d in rows[0]["docs"]] == DOCS_BY_KIND[1]
    man = json.loads(out.with_name("frozen.manifest.json").read_text(encoding="utf-8"))
    assert man["n"] == 5 and man["top_k"] == 5 and man["question_ids_sha256"]
    with pytest.raises(SystemExit, match="이미 있음"):
        freeze_inputs.main(
            [
                "--question-ids",
                str(tmp_path / "ids.txt"),
                "--reference-run",
                str(ref_run),
                "--out",
                str(out),
            ]
        )
    (tmp_path / "bad.txt").write_text("1\n99")
    with pytest.raises(ValueError, match="99"):
        freeze_inputs.main(
            [
                "--question-ids",
                str(tmp_path / "bad.txt"),
                "--reference-run",
                str(ref_run),
                "--out",
                str(tmp_path / "x.jsonl"),
            ]
        )
    assert freeze_inputs.read_ids(tmp_path / "ids.txt")[:2] == ["1", "2"]
    (tmp_path / "ids.csv").write_text("query_id,acc\n3,1\n4,0")
    assert freeze_inputs.read_ids(tmp_path / "ids.csv") == ["3", "4"]


def test_precompute_cache_then_mode_b_has_no_misses(tmp_path, ref_run, monkeypatch):
    frozen = make_frozen(tmp_path, ref_run)
    cfg_path = tmp_path / "cfg.yaml"
    base = _cfg(tmp_path, frozen)
    base["web_search"] = {
        "provider": "replay",
        "replay_from_runs": [str(ref_run)],
        "replay_key": "query_id",
        "top_k": 5,
    }
    cfg_path.write_text(yaml.safe_dump(base))
    monkeypatch.setattr(precompute_scores, "build_backend", lambda c: NamedMock(mock_fn))
    cache = tmp_path / "cache" / "scores.jsonl"
    precompute_scores.main(["--config", str(cfg_path), "--out", str(cache)])
    lines = cache.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "meta" and len(lines) > 10

    from mode_b import build_web_search

    from rag.crag_official.evaluator import CachedScoreBackend

    ev = OfficialRetrievalEvaluator(CachedScoreBackend([cache]))
    run_dir_base = tmp_path / "wsdir"
    run_dir_base.mkdir()
    comp = Components(
        engine=FakeEngine(),
        web_search=build_web_search(
            {**base, "web_search": {**base["web_search"], "latency_mode": "zero"}}, run_dir_base
        ),
        evaluator=ev,
    )
    out = ModeBRunner(base, components=comp, run_id="cached").run()
    rows = [json.loads(x) for x in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("error" not in r for r in rows), [r.get("error") for r in rows]
    assert ev.backend.misses == 0 and ev.backend.hits > 0
    # in-process mock 과 같은 판정
    comp2 = Components(
        engine=FakeEngine(),
        web_search=build_web_search(
            {**base, "web_search": {**base["web_search"], "latency_mode": "zero"}}, run_dir_base
        ),
        evaluator=OfficialRetrievalEvaluator(MockScoreBackend(mock_fn)),
    )
    out2 = ModeBRunner(base, components=comp2, run_id="inproc").run()
    rows2 = [
        json.loads(x) for x in (out2 / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    def key(rs):
        return [
            (r["question_id"], r["correction_path"], r["evidence"]["selected_evidence_ids"])
            for r in rs
        ]

    assert key(rows) == key(rows2)
    assert {r["correction_path"] for r in rows} == {"correct", "ambiguous", "incorrect"}
    assert any(r["correction"]["performed"] and r["correction"]["valid"] for r in rows)

    # analyze: 두 run 비교
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    analyze.main([str(out), str(out2), "--name", "cmp"])
    res = tmp_path / "results" / "official_crag" / "analysis" / "cmp"
    paired = list(__import__("csv").DictReader(open(res / "paired.csv", encoding="utf-8")))
    assert paired[0]["both_correct"] == "5" and paired[0]["mcnemar_p"] == ""
    agree = list(__import__("csv").DictReader(open(res / "agreement.csv", encoding="utf-8")))
    assert float(agree[0]["agreement"]) == 1.0
    assert "C1 비교" in (res / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        analyze.main([str(out), str(out2), "--name", "cmp"])


def test_mcnemar():
    assert analyze.mcnemar_p(0, 0) is None
    assert analyze.mcnemar_p(5, 5) == 1.0
    assert analyze.mcnemar_p(0, 6) == pytest.approx(2 / 64)


# ---------------------------------------------------------------- MODE A
@pytest.fixture
def official_repo(tmp_path):
    repo = tmp_path / "CRAG"
    d = repo / "data" / "popqa"
    (d / "ref").mkdir(parents=True)
    qs = ["Q0?", "Q1?", "Q2?"]
    lines = []
    docs = {
        "Q0?": ["GOOD answer Paris text"] + ["BAD"] * 9,
        "Q1?": ["MID text about it here"] + [""] * 9,
        "Q2?": [""] * 10,
    }
    for q in qs:
        lines += [f"{q} [SEP] {p}" if p else f"{q} [SEP]" for p in docs[q]]
    (d / "test_popqa.txt").write_text("\n".join(lines), encoding="utf-8")
    (d / "sources").write_text("\n".join(qs), encoding="utf-8")
    (d / "ref" / "correct").write_text("#Paris internal\n#int1\n#int2", encoding="utf-8")
    (d / "ref" / "incorrect").write_text("#ext0\n#ext1\n#web says Paris", encoding="utf-8")
    (d / "ref" / "ambiguous").write_text(
        "Knowledge1: a [sep] Knowledge2: b\nKnowledge1: Paris [sep] Knowledge2: x\nK",
        encoding="utf-8",
    )
    ans = tmp_path / "answers.jsonl"
    write_jsonl(
        ans, [{"question": q, "answers": ["Paris"], "id": str(10 + i)} for i, q in enumerate(qs)]
    )
    return repo, ans


def test_mode_a_pipeline(tmp_path, official_repo, monkeypatch):
    repo, ans = official_repo
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rag.telemetry.run_meta.ROOT", tmp_path)
    monkeypatch.setattr(mode_a, "create_run_dir", lambda exp, out, rid: _mk(tmp_path, exp, rid))
    monkeypatch.setattr(
        mode_a, "build_evaluator", lambda cfg: OfficialRetrievalEvaluator(NamedMock(mock_fn))
    )
    score_dir = _call(mode_a, ["--repo", str(repo), "--run-id", "s", "score"])
    rows = [
        json.loads(x) for x in (score_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["action"] for r in rows] == ["correct", "ambiguous", "incorrect"]
    assert rows[2]["scores"] == [-1.0] * 10  # 빈 문서

    ver = _call(mode_a, ["--repo", str(repo), "--run-id", "v", "verify-refinement"])
    s = json.loads((ver / "summary.json").read_text(encoding="utf-8"))
    assert s["n"] == 3 and s["exact_match"] == 0  # 가짜 ref 와는 당연히 다름 (형식만 확인)

    eng = FakeEngine()
    monkeypatch.setattr("rag.crag_official.completion_engine.build_generator_engine", lambda c: eng)
    cfgp = tmp_path / "a.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {"llm": {"provider": "vllm", "api": "completions", "model": "m", "max_tokens": 100}}
        )
    )
    gen = _call(
        mode_a,
        [
            "--repo",
            str(repo),
            "--run-id",
            "g",
            "generate",
            "--method",
            "crag",
            "--scores",
            str(score_dir),
            "--config",
            str(cfgp),
        ],
    )
    preds = [json.loads(x) for x in (gen / "preds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["action"] for p in preds] == ["correct", "ambiguous", "incorrect"]
    # 판정에 맞는 공식 지식 줄이 공식 prompt 로 들어감 ('#' 접두 유지)
    sent = [p for k, p in eng.prompts if k == "generate"]
    assert (
        "Documents: #Paris internal\n\n" in sent[0]
        and "Knowledge1: Paris" in sent[1]
        and "#web says Paris" in sent[2]
    )
    assert (gen / "preds.txt").read_text(encoding="utf-8").split("\n") == [
        "Paris",
        "Paris",
        "Paris",
    ]

    ids = tmp_path / "subset.txt"
    ids.write_text("10\n12")
    ev = _call(
        mode_a,
        [
            "--repo",
            str(repo),
            "--run-id",
            "e",
            "evaluate",
            "--preds",
            str(gen),
            "--answers",
            str(ans),
            "--subset-ids",
            str(ids),
        ],
    )
    s = json.loads((ev / "summary.json").read_text(encoding="utf-8"))
    assert s["official_match"] == 1.0 and s["subset"]["n"] == 2


def _mk(root, exp, rid):
    p = root / "results" / exp / rid
    p.mkdir(parents=True)
    return p


def _call(mod, argv):
    captured = {}
    orig = {
        k: getattr(mod, k)
        for k in ("cmd_score", "cmd_verify_refinement", "cmd_generate", "cmd_evaluate")
    }

    def wrap(fn):
        def inner(a):
            captured["dir"] = fn(a)
            return captured["dir"]

        return inner

    for k, f in orig.items():
        setattr(mod, k, wrap(f))
    try:
        mod.main(argv)
    finally:
        for k, f in orig.items():
            setattr(mod, k, f)
    return captured["dir"]
