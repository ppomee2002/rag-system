"""diag_evc_live.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님).

seq (DocEngine sequential) -> evf (evidence_filter_ablation) -> evc (evidence_cost measure)
-> live (run_evidence_live, fcfs_query K=2, burst) 를 만들고 진단 스크립트를 돌린다.
"""

import csv
import json
from dataclasses import replace

import diag_evc_live as dev
import evidence_cost as ec
import evidence_filter_ablation as efa
import pytest
import run_evidence_live as rel
from crag_fixtures import FakeEngine, Retriever, Web, pool_ids
from runner import Components, Runner
from workload import make_workload

from rag.retrieval.web_cache import RecordingWebSearch
from rag.telemetry.calls import infer_call_kind

POLS = ("crag_rule", "keep_all")


class DocEngine(FakeEngine):
    """grade: 'even doc 0' 만 0.9. generate: context 에 'doc 2' 가 있으면 정답."""

    def __init__(self, delay=0.003):
        super().__init__(delay=delay)

    def complete(self, request):
        r = super().complete(request)
        kind = request.kind if request.kind != "other" else infer_call_kind(request.prompt)
        if kind == "grade":
            return replace(r, text="0.9" if "even doc 0" in request.prompt else "0.1")
        if kind == "generate":
            return replace(r, text="ans" if "doc 2" in request.prompt else "nope")
        return r


class TimedEngine:
    name, model = "fake", "fake-model"

    def complete(self, request):
        from rag.llm.engine import LLMResponse

        pt = len(request.prompt) // 4
        return LLMResponse("ans" if "doc 2" in request.prompt else "nope", "fake", t_start=0.0,
                           t_end=0.001 * pt, prompt_tokens=pt, completion_tokens=2)  # fmt: skip

    def reset_prefix_cache(self):
        return {"ok": True}

    def describe(self):
        return {"engine": "fake"}


@pytest.fixture
def chain(setup, tmp_path):
    tmp, make = setup
    web_log = tmp / "web_cache_rec.jsonl"
    cfg, _ = make()
    seq = Runner(cfg, Components(Retriever(), RecordingWebSearch(Web(), web_log), DocEngine()),
                 run_id="seq").run()  # fmt: skip
    docs = tmp / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as f:
        for tag in ("even", "odd"):
            for i in range(3):
                f.write(json.dumps({"id": f"{tag}{i}", "text": f"{tag} doc {i}"}) + "\n")
    evf = efa.main(
        ["generate", "--ref", str(seq), "--split", "all", "--exp-id", "PRE-EVF-T",
         "--docs-path", str(docs), "--web-cache", str(web_log), "--output-dir", str(tmp_path),
         "--out-root", str(tmp_path / "ana"), "--n-boot", "50", "--quiet"],
        engine=DocEngine(delay=0.0),
    )  # fmt: skip
    evc = ec.main(["measure", "--evf", str(evf), "--exp-id", "PRE-EVC-T", "--output-dir",
                   str(tmp_path / "m"), "--warmup", "0"], engine=TimedEngine())  # fmt: skip
    cm = "{trace: '" + seq.as_posix() + "', service_key: wall, oof_folds: 3}"
    for s in (0, 1):
        wl = None
        for pol in POLS:
            cfg, _ = make(
                ["run.mode=live", "exp_id=PRE-EVC-LIVE", "run.purpose=pre",
                 f"evidence_policy={pol}", "scheduler.policy=fcfs_query", "scheduler.slots=2",
                 f"scheduler.cost_model={cm}"]
            )  # fmt: skip
            wl = wl or make_workload(pool_ids(cfg)[1], 10, seed=s, rate=2000.0, name=f"b{s}")
            rel.EvidenceRunner(
                cfg, Components(Retriever(), Web(), DocEngine()), workload=wl,
                run_id=f"rho0.8_s{s}_{pol}_K2",
            ).run()  # fmt: skip
    return tmp / "results" / "crag_system" / "pre" / "PRE-EVC-LIVE", seq, evf, evc


def test_helpers():
    assert dev.union_len([(0, 2), (1, 3), (5, 6)]) == pytest.approx(4.0)
    assert dev.union_len([]) == 0.0
    segs = dev.step_segments({"a": [(0, 2), (1, 3)], "b": [(2.5, 4)]}, 0, 4)
    assert dev.tw_mean(segs, lambda c: c["a"]) == pytest.approx(4 / 4)
    assert dev.tw_mean(segs, lambda c: float(c["a"] >= 2)) == pytest.approx(1 / 4)
    assert dev.tw_mean(segs, lambda c: c["b"]) == pytest.approx(1.5 / 4)


def test_end_to_end(chain, tmp_path):
    root, seq, evf, evc = chain
    out = dev.main(["--live-root", str(root), "--seq", str(seq), "--evf", str(evf), "--evc-seq",
                    str(evc), "--k", "2", "--bins", "4", "--name", "t", "--out-root",
                    str(tmp_path / "ana"), "--quiet"])  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert res["warnings"] == []
    A = res["accuracy"]
    assert A["seed_check"]["same_qset_all_runs"]
    pr = {(r["seed"], r["policy"]): r for r in A["per_run"]}
    for s in (0, 1):
        c, k = pr[(s, "crag_rule")], pr[(s, "keep_all")]
        assert c["n_ok"] == 10 and c["dup_qids"] == 0 and c["n_in_seq"] == 10
        assert c["action_agree_seq"] == 1.0 and c["action_agree_denominator"] == 10
        # 결정적 가짜 엔진: live 정확도 = 같은 질문의 evf 정확도
        assert c["acc_live_evf_common"] == c["acc_evf_same_policy_common"] == 0.0
        assert k["acc_live_evf_common"] == k["acc_evf_same_policy_common"] == 1.0
        assert c["answer_same_seq_if_same_action"] == 1.0
    for r in res["decomp_summary"]:
        assert abs(r["sum_check"]) < 1e-6 and r["wait_only"] >= -1e-9
        assert r["noload_exact_share"] > 0
    assert {(p["seed"], p["policy"]) for p in res["decomp_paired"]} == {
        (0, "keep_all"),
        (1, "keep_all"),
    }
    for q in res["queueing"]:
        assert q["running_mean"] <= 2 + 1e-9 and q["lambda_obs"] > 0 and q["throughput"] > 0
        assert 0 <= q["slots_full_share"] <= 1
        # 시간 평균 N x 관측 구간 = 요청 체류시간 합 (sweep 계산 검증)
        lat = next(r for r in res["decomp_summary"] if r["subset"] == "all"
                   and (r["seed"], r["policy"]) == (q["seed"], q["policy"]))  # fmt: skip
        assert q["N_mean"] * q["run_span_s"] == pytest.approx(lat["latency"] * lat["n"], rel=1e-6)
    ka = [q for q in res["queueing"] if q["policy"] == "keep_all"]
    assert all(q["work_ratio_vs_crag_rule"] > 1 for q in ka)  # 긴 context -> no-load 작업 증가
    sc = res["scheduling"]
    assert all(r["srpt_est_first_spearman"] is not None or r["noload_cv"] == 0 for r in sc)
    with open(out / "request_decomp.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 40
    assert "## C. Queueing" in (out / "report.md").read_text(encoding="utf-8")
    out2 = dev.main(["--live-root", str(root), "--seq", str(seq), "--name", "t", "--out-root",
                     str(tmp_path / "ana"), "--quiet", "--k", "2"])  # fmt: skip
    assert out2 != out  # 덮어쓰지 않음, evf / evc 없이도 동작
