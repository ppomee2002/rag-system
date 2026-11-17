"""evidence_cost.py / run_evidence_live.py 기능 검증 (가짜 엔진, 실제 성능 결과 아님).

DocEngine 규칙은 test_evidence_filter_ablation 과 같다: grade 는 "even doc 0" 만 0.9 -> even 질문
correct (doc0 만 유지), odd 질문 incorrect. TimedEngine 은 wall = 1 ms x prompt 토큰으로 응답해
context 가 길수록 generation latency 가 커진다.
"""

import json
from dataclasses import replace
from pathlib import Path

import evidence_cost as ec
import evidence_filter_ablation as efa
import pytest
import run_evidence_live as rel
from crag_fixtures import FakeEngine, Retriever, Web
from runner import Components, Runner

from rag.llm.engine import LLMResponse
from rag.retrieval.web_cache import RecordingWebSearch, read_jsonl
from rag.telemetry.calls import infer_call_kind


class DocEngine(FakeEngine):
    def __init__(self):
        super().__init__(delay=0.0)

    def complete(self, request):
        r = super().complete(request)
        kind = request.kind if request.kind != "other" else infer_call_kind(request.prompt)
        if kind == "grade":
            return replace(r, text="0.9" if "even doc 0" in request.prompt else "0.1")
        if kind == "generate":
            return replace(r, text="ans" if "doc 2" in request.prompt else "nope")
        return r


class TimedEngine:
    """prompt 길이에 비례하는 가짜 wall / TTFT. 호출 순서와 cache 초기화를 기록."""

    name, model = "fake", "fake-model"

    def __init__(self, reset_ok=True, params=None):
        self.prompts, self.resets, self.reset_ok, self.params = [], 0, reset_ok, params

    def complete(self, request):
        self.prompts.append(request.prompt)
        pt = len(request.prompt) // 4
        text = "ans" if "doc 2" in request.prompt else "nope"
        return LLMResponse(text, "fake", t_start=0.0, t_first_token=0.0002 * pt,
                           t_end=0.001 * pt, prompt_tokens=pt, completion_tokens=2)  # fmt: skip

    def reset_prefix_cache(self):
        self.resets += 1
        return {"ok": self.reset_ok}

    def describe(self):
        return {"engine": "fake", **({"params": self.params, "model": "m"} if self.params else {})}


@pytest.fixture
def evf(setup, tmp_path):
    tmp, make = setup
    web_log = tmp / "web_cache_rec.jsonl"
    cfg, _ = make()
    ref = Runner(
        cfg, Components(Retriever(), RecordingWebSearch(Web(), web_log), DocEngine()), run_id="ref"
    ).run()
    docs = tmp / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as f:
        for tag in ("even", "odd"):
            for i in range(3):
                f.write(json.dumps({"id": f"{tag}{i}", "text": f"{tag} doc {i}"}) + "\n")
    out = efa.main(
        ["generate", "--ref", str(ref), "--split", "all", "--exp-id", "PRE-EVF-T",
         "--docs-path", str(docs), "--web-cache", str(web_log), "--output-dir", str(tmp_path),
         "--out-root", str(tmp_path / "ana"), "--n-boot", "50", "--quiet"],
        engine=DocEngine(),
    )  # fmt: skip
    return out, ref


def measure(evf_dir, tmp_path, *extra, engine=None):
    return ec.main(["measure", "--evf", str(evf_dir), "--output-dir", str(tmp_path / "m"),
                    "--warmup", "1", *extra], engine=engine or TimedEngine())  # fmt: skip


def test_measure_dedup_order_and_report(evf, tmp_path):
    evf_dir, ref = evf
    eng = TimedEngine()
    run = measure(evf_dir, tmp_path, "--exp-id", "PRE-EVC-T", "--repeats", "2", engine=eng)
    rows = read_jsonl(run / "measurements.jsonl")
    meas = [r for r in rows if not r["warmup"]]
    units = read_jsonl(run / "units.jsonl")
    items = [it for it in read_jsonl(evf_dir / "items.jsonl") if not it["problems"]]
    n_inc = sum(it["action"] == "incorrect" for it in items)
    # 같은 context 는 한 번만: correct 2 개 / incorrect 3 개 고유 context (evf 의 공유와 같음)
    assert len(units) == 2 * (len(items) - n_inc) + 3 * n_inc
    assert len(meas) == 2 * len(units) and len(eng.prompts) == 2 * len(units) + 1  # + warmup
    assert eng.resets == len(meas)  # 매 호출 전 초기화
    assert all(r["cache_reset_ok"] and r["answer_same_evf"] for r in meas)
    order = [[r["unit_id"] for r in meas if r["repeat"] == k] for k in (0, 1)]
    assert sorted(order[0]) == sorted(order[1]) and order[0] != order[1]  # repeat 마다 다시 섞음
    assert {c for u in units for c in u["conditions"]} == set(efa.CONDITIONS)

    out = ec.main(["report", "--evf", str(evf_dir), "--measured", str(run), "--name", "t",
                   "--out-root", str(tmp_path / "ana"), "--n-boot", "100", "--quiet"])  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    s = {r["condition"]: r for r in res["summary"]}
    assert res["source"] == "측정" and res["n"] == len(items)
    assert s["crag_rule"]["acc"] == 0.0 and s["keep_all"]["acc"] == 1.0  # evf 정확도 그대로
    assert s["keep_all"]["gen_mean"] > s["crag_rule"]["gen_mean"]  # 긴 context -> 긴 latency
    assert s["keep_all"]["unique_gen_measured"] == len(items)
    pq = read_jsonl_csv(out / "per_query.csv")
    seq = ec.load_seq(ref)[0]
    for r in pq:
        q = seq[r["query_id"]]
        assert float(r["e2e_recon"]) == pytest.approx(q["non_gen_time"] + float(r["gen"]))
    p = {(r["target"], r["subset"]): r for r in res["paired"] if r["base"] == "crag_rule"}
    assert p[("no_correct_filter", "incorrect")]["gen_diff"] == 0.0  # 해당 action 외에는 영향 없음
    assert p[("keep_all", "all")]["e2e_diff"] > 0 and p[("keep_all", "all")]["acc_diff"] == 1.0
    assert "keep_all" in res["live_candidates"]["acc_significant"]
    assert "est_vs_meas" in res["validation"]
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "[측정]" in md and "[재구성]" in md

    # 측정 없이 [추정]
    out2 = ec.main(["report", "--evf", str(evf_dir), "--name", "t", "--out-root",
                    str(tmp_path / "ana"), "--n-boot", "50", "--quiet"])  # fmt: skip
    assert out2 != out  # 덮어쓰지 않음
    res2 = json.loads((out2 / "analysis.json").read_text(encoding="utf-8"))
    assert res2["source"] == "추정" and "wall" in res2["validation"]["model"]


def read_jsonl_csv(p: Path) -> list[dict]:
    import csv

    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_measure_guards_and_resume(evf, tmp_path):
    evf_dir, _ = evf
    with pytest.raises(SystemExit):  # PRE-* 만
        measure(evf_dir, tmp_path, "--exp-id", "SYS-1")
    with pytest.raises(SystemExit):  # 초기화 실패 -> 중단 (cache 조건이 섞이지 않게)
        measure(evf_dir, tmp_path, "--exp-id", "PRE-A", engine=TimedEngine(reset_ok=False))
    meta = json.loads((evf_dir / "run_meta.json").read_text(encoding="utf-8"))
    meta["engine"] = {"model": "m", "params": {"temperature": 0.0, "max_tokens": 512, "seed": 0,
                                               "stream": True}}  # fmt: skip
    (evf_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    bad = TimedEngine(params={"temperature": 0.7, "max_tokens": 512, "seed": 0, "stream": True})
    with pytest.raises(SystemExit):  # 생성 설정 불일치
        measure(evf_dir, tmp_path, "--exp-id", "PRE-B", engine=bad)
    good = TimedEngine(params={"temperature": 0.0, "max_tokens": 512, "seed": 0, "stream": True})
    run = measure(evf_dir, tmp_path, "--exp-id", "PRE-C", "--prefix-cache", "none",
                  "--conditions", "crag_rule", "keep_all", engine=good)  # fmt: skip
    n1 = len(read_jsonl(run / "measurements.jsonl"))
    eng = TimedEngine(params=good.params)
    ec.main(["measure", "--evf", str(evf_dir), "--resume", str(run)], engine=eng)
    assert len(read_jsonl(run / "measurements.jsonl")) == n1 and eng.prompts == []  # 이미 완료
    assert good.resets == 0


def test_live_graph_contexts_and_live_report(setup, tmp_path):
    _, make = setup
    with pytest.raises(ValueError):
        rel.parse_policy({"evidence_policy": "vanilla_ctx"})
    with pytest.raises(ValueError):
        rel.parse_policy({})
    runs = {}
    for pol in ("crag_rule", "keep_all", "no_incorrect_discard"):
        cfg, comp = make([f"evidence_policy={pol}"], engine=DocEngine())
        cfg["exp_id"] = "SMOKE-EVC-LIVE-T"
        runs[pol] = rel.EvidenceRunner(cfg, comp, run_id=f"rho0.8_s0_{pol}_K4").run()
    cfg, comp = make(engine=DocEngine())
    plain = Runner(cfg, comp, run_id="plain").run()
    rows = {p: {r["query_id"]: r for r in read_jsonl(d / "queries.jsonl") if not r["warmup"]}
            for p, d in runs.items()}  # fmt: skip
    base = {r["query_id"]: r for r in read_jsonl(plain / "queries.jsonl") if not r["warmup"]}
    for q, r in base.items():
        assert rows["crag_rule"][q]["contexts"] == r["contexts"]  # crag_rule = 기존 그대로
        docs = r["retrieval_doc_ids"]
        web = r["extra"]["web_doc_ids"]
        for p in rows:
            assert rows[p][q]["corrective_action"] == r["corrective_action"]  # 판정 불변
            assert rows[p][q]["calls_summary"]["n_calls"] == r["calls_summary"]["n_calls"]
        if r["corrective_action"] == "correct":
            assert rows["keep_all"][q]["contexts"] == docs
            assert rows["no_incorrect_discard"][q]["contexts"] == r["contexts"]
        else:
            assert rows["keep_all"][q]["contexts"] == docs + web
            assert rows["no_incorrect_discard"][q]["contexts"] == docs + web
    assert rows["keep_all"][q]["pipeline"].endswith("+evp_keep_all")
    assert rows["keep_all"][q]["extra"]["evidence_policy"] == "keep_all"

    root = runs["crag_rule"].parent
    out = ec.main(["live-report", "--live-root", str(root), "--seq", str(plain), "--name", "lt",
                   "--out-root", str(tmp_path / "ana"), "--quiet"])  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    pr = {r["policy"]: r for r in res["per_run"]}
    assert set(pr) == set(runs) and all(r["action_agree_seq"] == 1.0 for r in pr.values())
    assert all(r["policy_cfg"] == p for p, r in pr.items())
    assert pr["keep_all"]["acc"] == 1.0 and pr["crag_rule"]["acc"] == 0.0
    agg = {r["policy"]: r for r in res["agg"]}
    assert agg["keep_all"]["d_acc_vs_ref"] == 1.0
