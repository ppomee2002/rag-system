"""cost_state_ablation.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import csv
import json

import cost_state_ablation as csa
import evidence_cost as ec
import evidence_filter_ablation as efa
import fit_cost_model
import numpy as np
import pytest
import run_evidence_live as rel
from crag_fixtures import Retriever, Web
from runner import Components, Runner, apply_split, load_examples, split_warmup
from test_diag_evc_live import DocEngine, TimedEngine
from workload import make_workload

from rag.pipelines import prompts
from rag.retrieval.web_cache import RecordingWebSearch
from rag.scheduling.cost_model import CostModel
from rag.types import Document


def test_prompt_char_arithmetic():
    for docs in ([], ["a" * 7], ["x" * 3, "yy", "z" * 11]):
        ctx = [Document(id=str(i), text=t) for i, t in enumerate(docs)]
        p = prompts.GENERATE.format(context=prompts.format_context(ctx), question="Why?")
        assert len(p) == csa.gen_chars(4, [len(t) for t in docs])
    # FAISS 2개 + web 3개 (합 15)
    ctx = [Document(id=str(i), text=t) for i, t in enumerate(["aa", "bbb", "c" * 5, "d" * 4,
                                                               "e" * 6])]  # fmt: skip
    p = prompts.GENERATE.format(context=prompts.format_context(ctx), question="Q")
    assert len(p) == csa.gen_chars(1, [2, 3], 3, 15.0)
    assert csa.faiss_kept("crag_rule", "correct", [0.9, 0.2, 0.75], 0.7, 0.3) == [0, 2]
    assert csa.faiss_kept("keep_all", "incorrect", [0.1, 0.1], 0.7, 0.3) == [0, 1]
    assert csa.faiss_kept("crag_rule", "incorrect", [0.1, 0.1], 0.7, 0.3) == []


def _traj(qid, scores, action, calls):
    return csa.Traj(key=f"0:{qid}", qid=qid, policy="crag_rule", action=action, scores=scores,
                    feats=np.zeros(6), n_docs=len(scores), rchars=[100] * len(scores),
                    rscores=[0.5] * len(scores), q_len=10, s_pop=None,
                    calls=[{"kind": k, "idx": i, "chars": 50, "wall": w, "t_enq": t,
                            "t_end": t + 0.5, "src": "관측"} for k, i, w, t in calls],
                    web_n=2, web_total=80.0)  # fmt: skip


def test_obs_has_no_future_information():
    g = [("grade", 0, 1.0, 0.0), ("grade", 1, 1.0, 1.0), ("grade", 2, 1.0, 2.0)]
    t = _traj("a", [0.1, 0.2, 0.9], "correct", [*g, ("generate", 0, 2.0, 3.0)])
    o1 = csa.obs_at(t, 1)
    assert o1.action is None and o1.seen == ((0, 0.1),)
    with pytest.raises(ValueError):  # grade 단계에서 action 을 넣으면 거부
        csa.Obs(**{**o1.__dict__, "action": "correct"})
    with pytest.raises(ValueError):  # 현재 grade 점수도 관측 불가
        csa.Obs(**{**o1.__dict__, "seen": ((0, 0.1), (1, 0.2))})
    train = [_traj(f"t{i}", s, a, [*g, *([("rewrite", 0, 0.5, 3.0)] if a != "correct" else []),
                                    ("generate", 0, w, 4.0)])
             for i, (s, a, w) in enumerate([([0.9, 0.1, 0.1], "correct", 2.0),
                                            ([0.1, 0.1, 0.1], "incorrect", 3.0),
                                            ([0.5, 0.1, 0.1], "ambiguous", 4.0)])]  # fmt: skip
    st = csa.StateTables(train, 0.7, 0.3, min_cell=1)
    m = CostModel(grade_coef=np.array([1.0, 0.0]), gen_coef=np.array([0.0, 0.01]),
                  rewrite_mean=0.5, gen_corrected_mean=3.0, total_beta=np.zeros(6),
                  post_beta=np.zeros(6))  # fmt: skip
    b = csa.Bundle(m, csa.Tables(csa._as_tq(train), 0.7, 0.3, min_cell=1), st)
    # 최종 action / 이후 점수를 바꿔도 grade 시점 추정은 같다
    t2 = _traj("a", [0.1, 0.2, 0.1], "incorrect", [*g, ("rewrite", 0, 0.5, 3.0),
                                                   ("generate", 0, 9.0, 4.0)])  # fmt: skip
    for j in range(3):
        assert csa.estimate(b, csa.obs_at(t, j)) == csa.estimate(b, csa.obs_at(t2, j))
    # 관측 점수가 upper 이상이면 correct 확정 -> rewrite 항 없음
    t3 = _traj("c", [0.9, 0.1, 0.1], "correct", [*g, ("generate", 0, 2.0, 3.0)])
    p, src = st.p_action(csa.obs_at(t3, 1))
    assert src == "hit" and p["correct"] == 1.0
    # generate 시점은 M0 = M2s = M2 (실제 prompt 길이)
    e = csa.estimate(b, csa.obs_at(t, 3))
    assert e["M0"] == e["M2s"] == e["M2"] == pytest.approx(0.5)
    # evidence 정책: keep_all 의 기대 prompt 가 crag_rule 보다 길다 (correct 에서 FAISS 전부)
    o = csa.obs_at(t3, 1)
    ok = csa.Obs(**{**o.__dict__, "policy": "keep_all"})
    assert st.exp_gen_chars(ok, "correct") > st.exp_gen_chars(o, "correct")
    # 실제 정책 클래스와 같은 값
    assert csa.m0_policy_class(m, o) == pytest.approx(csa.estimate(b, o)["M0"])


@pytest.fixture
def chain(setup, tmp_path):
    tmp, make = setup
    web_log = tmp / "web_cache_rec.jsonl"
    cfg, _ = make(["dataset.split.part=train"])
    train = Runner(cfg, Components(Retriever(), Web(), DocEngine()), run_id="train").run()
    cm = tmp / "cm.json"
    fit_cost_model.main([str(train), "--out", str(cm)])
    cfg, _ = make(["dataset.split.part=eval"])
    seq = Runner(cfg, Components(Retriever(), RecordingWebSearch(Web(), web_log), DocEngine()),
                 run_id="seq").run()  # fmt: skip
    docs = tmp / "docs.jsonl"
    with open(docs, "w", encoding="utf-8") as f:
        for tag in ("even", "odd"):
            for i in range(3):
                f.write(json.dumps({"id": f"{tag}{i}", "text": f"{tag} doc {i}"}) + "\n")
    evf = efa.main(
        ["generate", "--ref", str(seq), "--split", "eval", "--exp-id", "PRE-EVF-T",
         "--docs-path", str(docs), "--web-cache", str(web_log), "--output-dir", str(tmp_path),
         "--out-root", str(tmp_path / "ana"), "--n-boot", "50", "--quiet"],
        engine=DocEngine(delay=0.0),
    )  # fmt: skip
    evc = ec.main(["measure", "--evf", str(evf), "--exp-id", "PRE-EVC-T", "--output-dir",
                   str(tmp_path / "m"), "--warmup", "0"], engine=TimedEngine())  # fmt: skip
    for pol in ("crag_rule", "keep_all"):
        cfg, _ = make(["run.mode=live", "exp_id=PRE-EVC-LIVE", "run.purpose=pre",
                       "dataset.split.part=eval", f"evidence_policy={pol}",
                       "scheduler.policy=fcfs_query", "scheduler.slots=2",
                       f"scheduler.cost_model={{path: '{cm.as_posix()}'}}"])  # fmt: skip
        _, pool = split_warmup(cfg, load_examples(cfg))
        wl = make_workload(
            [ex.id for ex in apply_split(cfg, pool)], None, seed=0, rate=2000.0, name="b0"
        )
        rel.EvidenceRunner(cfg, Components(Retriever(), Web(), DocEngine()), workload=wl,
                           run_id=f"rho0.8_s0_{pol}_K2").run()  # fmt: skip
    live = tmp / "results" / "crag_system" / "pre" / "PRE-EVC-LIVE"
    return train, seq, cm, evc, live


def test_end_to_end(chain, tmp_path):
    train, seq, cm, evc, live = chain
    args = ["--train", str(train), "--seq", str(seq), "--cost-model", str(cm), "--evc-seq",
            str(evc), "--live-root", str(live), "--policies", "crag_rule", "keep_all",
            "--min-cell", "1", "--n-boot", "50", "--name", "t", "--out-root",
            str(tmp_path / "ana"), "--quiet"]  # fmt: skip
    out = csa.main(args)
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert res["leakage"]["train_eval_overlap"] == 0
    assert res["leakage"]["cost_model_eval_overlap"] == 0
    assert res["m0_vs_policy_class"]["checked"] > 0
    assert res["m0_vs_policy_class"]["max_abs_diff"] < 1e-9
    assert res["chars_check_seq"]["exact_share"] == 1.0
    assert res["evc"]["questions_with_all_policies"] > 0
    # live 의 generate prompt 길이 = 산술 재구성 (같은 질문·action)
    assert all(r["exact_share"] == 1.0 for r in res["chars_check_live"])
    mt = res["metrics"]
    gen = [r for r in mt if r["stage"] == "generate" and r["action"] == "all"]
    for r in gen:  # generate 시점은 M0 = M2s = M2
        same = [x for x in gen if (x["set"], x["policy"]) == (r["set"], r["policy"])]
        assert len({round(x["est_mean"], 12) for x in same if x["model"] != "M1"}) == 1
    pols = {r["policy"] for r in mt if r["set"] == "evc"}
    assert pols == {"crag_rule", "keep_all"}
    assert {r["rule"] for r in res["live"]} == {*csa.MODELS, "fcfs_query"}
    m0 = [r for r in res["live"] if r["rule"] == "M0"]
    assert all(r["top_differs_vs_M0"] in (0.0, None) for r in m0)
    with open(out / "points.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) > 0
    assert "## 데이터 분리" in (out / "report.md").read_text(encoding="utf-8")
    # eval 질문으로 학습한 cost model 은 거부 (누수)
    cm_bad = tmp_path / "bad.json"
    fit_cost_model.main([str(seq), "--out", str(cm_bad)])
    with pytest.raises(SystemExit):
        csa.main([*args[:4], "--cost-model", str(cm_bad), "--name", "t2", "--out-root",
                  str(tmp_path / "ana"), "--quiet"])  # fmt: skip
