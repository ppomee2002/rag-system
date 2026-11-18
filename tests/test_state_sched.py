"""state_sched.py / run_state_sched.py 기능 검증 (가짜 엔진 run, 실제 성능 결과 아님)."""

import json

import cost_state_ablation as csa
import fit_cost_model
import numpy as np
import pytest
import run_evidence_live as rel
import state_sched as ss
from crag_fixtures import Retriever, Web
from diag_sv_policy import load_run
from run_state_sched import StateSchedRunner
from runner import Components, Runner, apply_split, load_examples, split_warmup
from test_diag_evc_live import DocEngine
from workload import make_workload

from rag.scheduling.cost_model import cost_model_from_dict
from rag.scheduling.policies import CallTicket, SRPTEst
from rag.scheduling.state import QueryState


@pytest.fixture
def trained(setup, tmp_path):
    tmp, make = setup
    cfg, _ = make(["dataset.split.part=train"])
    train = Runner(cfg, Components(Retriever(), Web(), DocEngine()), run_id="train").run()
    cm = tmp / "cm.json"
    fit_cost_model.main([str(train), "--out", str(cm)])
    sm = tmp / "state_m2.json"
    ss.main(["fit", "--train", str(train), "--cost-model", str(cm), "--out", str(sm),
             "--min-cell", "1"])  # fmt: skip
    return train, cm, sm


def test_fit_roundtrip_and_no_overwrite(trained):
    train, cm, sm = trained
    run = load_run(train)
    up, lo = csa.thresholds_of(run)
    trajs = csa.trajectories(run, up, lo)
    fresh = csa.StateTables(trajs, up, lo, min_cell=1)
    st, meta = ss.load_state_model(sm)
    assert meta["cost_model_sha256"] == ss.file_sha(cm)
    assert meta["n_train_queries"] == len({t.qid for t in trajs})
    model = cost_model_from_dict(json.loads(cm.read_text(encoding="utf-8")))
    for t in trajs:
        for j in range(len(t.calls)):
            o = csa.obs_at(t, j)
            assert st.p_action(o) == fresh.p_action(o)
            for a in csa.ACTIONS:
                assert st.exp_gen_chars(o, a) == pytest.approx(fresh.exp_gen_chars(o, a))
            # live 식 = 오프라인 ablation 의 M2
            b = csa.Bundle(model, csa.Tables(csa._as_tq(trajs), up, lo, min_cell=1), fresh)
            v, _ = ss.m2_estimate(model, st, o)
            assert v == pytest.approx(csa.estimate(b, o)["M2"])
    with pytest.raises(SystemExit):  # 덮어쓰지 않음
        ss.main(["fit", "--train", str(train), "--cost-model", str(cm), "--out", str(sm)])


def _ticket(board, kind, idx, chars=100, n_docs=3):
    q = QueryState("qA", s_pop=5.0, request_key="0:qA")
    q.retrieval_scores, q.retrieval_chars, q.n_docs = [0.5, 0.4, 0.3], [50, 60, 70], n_docs
    return CallTicket(q, kind, idx, chars, 0.0, 0)


def test_observe_uses_only_completed_scores(trained):
    _, cm, sm = trained
    st, _ = ss.load_state_model(sm)
    board = ss.ScoreBoard()
    board.start("0:qA", 12)
    board.add("0:qA", 0, 0.1)
    o = ss.observe(_ticket(board, "grade", 1), board, "keep_all", 0.7, 0.3)
    assert o.seen == ((0, 0.1),) and o.action is None and o.q_len == 12
    # 다음 grade 가 끝나기 전에는 그 점수가 보이지 않는다 / 현재 호출 점수는 Obs 가 거부
    o2 = ss.observe(_ticket(board, "grade", 1), board, "keep_all", 0.7, 0.3)
    assert o2.seen == ((0, 0.1),)
    with pytest.raises(ValueError):
        csa.Obs(**{**o.__dict__, "seen": ((0, 0.1), (1, 0.2))})
    board.add("0:qA", 1, 0.9)
    board.add("0:qA", 2, 0.2)
    with pytest.raises(ValueError):  # 중복 기록 거부
        board.add("0:qA", 2, 0.2)
    o3 = ss.observe(_ticket(board, "rewrite", 0), board, "keep_all", 0.7, 0.3)
    assert o3.action == "correct" and len(o3.seen) == 3
    # srpt_est 모드: 사용값 = 기존 SRPTEst.base
    model = cost_model_from_dict(json.loads(cm.read_text(encoding="utf-8")))
    p = ss.StateSRPT(model, st, board, "keep_all", use="M0")
    t = _ticket(board, "grade", 1)
    assert p.base(t) == SRPTEst(model).base(t)
    assert p.log[-1]["m2"] is not None and p.name == "srpt_est"
    # 미등록 요청 (tap 미연결) 은 조용히 넘어가지 않음
    with pytest.raises(RuntimeError):
        ss.observe(_ticket(ss.ScoreBoard(), "grade", 0), ss.ScoreBoard(), "keep_all", 0.7, 0.3)


def _live(make, cm, sm, sched, runner_cls, run_id, extra=()):
    sets = ["run.mode=live", "exp_id=PRE-SST-T", "run.purpose=pre", "dataset.split.part=eval",
            "evidence_policy=keep_all", f"scheduler.policy={sched}", "scheduler.slots=2",
            f"scheduler.cost_model={{path: '{cm.as_posix()}'}}",
            f"scheduler.state_model={{path: '{sm.as_posix()}'}}", *extra]  # fmt: skip
    cfg, _ = make(sets)
    _, pool = split_warmup(cfg, load_examples(cfg))
    wl = make_workload([ex.id for ex in apply_split(cfg, pool)], None, seed=0, rate=500.0,
                       name="t")  # fmt: skip
    return runner_cls(cfg, Components(Retriever(), Web(), DocEngine()), workload=wl,
                      run_id=run_id).run()  # fmt: skip


def _sig(run_dir):
    run = load_run(run_dir)
    out = {}
    for k, r in run["rows"].items():
        kinds = sorted((c["kind"], c["idx"]) for c in run["calls"][k])
        out[k] = (kinds, r["contexts"], r["answer"], r["corrective_action"],
                  r["extra"]["grading"]["scores_full"])  # fmt: skip
    return out


def test_live_runs_preserve_pipeline(setup, trained):
    _, make = setup
    _, cm, sm = trained
    base = _live(make, cm, sm, "fcfs_query", rel.EvidenceRunner, "fq")
    est = _live(make, cm, sm, "srpt_est", StateSchedRunner, "m0")
    st = _live(make, cm, sm, "srpt_state", StateSchedRunner, "m2")
    sb, se, s2 = _sig(base), _sig(est), _sig(st)
    assert sb.keys() == se.keys() == s2.keys() and len(sb) > 0
    assert sb == se == s2  # 같은 요청: 같은 호출 수·종류, 같은 evidence, 같은 답
    for d, name, used in ((est, "srpt_est", "M0"), (st, "srpt_state", "M2")):
        chk = json.loads((d / "state_check.json").read_text(encoding="utf-8"))
        assert chk["score_mismatch_rows"] == 0 and chk["grade_idx_mismatch_rows"] == 0
        assert chk["estimates_logged"] == chk["gate_dispatches"] > 0
        assert chk["m0_eq_m2_share_generate"] == 1.0
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        pol = meta["scheduler"]["policy"]
        assert pol["name"] == name and pol["estimator"] == used
        assert meta["preflight"]["state_model_leakage"]["train_eval_overlap"] == 0
        est_log = {e["seq"]: e for e in map(json.loads, (d / "state_estimates.jsonl").read_text(
            encoding="utf-8").splitlines())}  # fmt: skip
        gate = [json.loads(x) for x in (d / "scheduler_log.jsonl").read_text(
            encoding="utf-8").splitlines()]  # fmt: skip
        for g in gate:  # dispatch key 의 primary = 그 정책이 쓴 추정치
            e = est_log[g["seq"]]
            assert g["key"][1] == pytest.approx(e["m2"] if used == "M2" else e["m0"])
        ov = json.loads((d / "scheduler_overhead.json").read_text(encoding="utf-8"))
        assert ov["used"]["n"] == len(est_log) and ov["used"]["mean_us"] >= 0
    # M0 run 의 사용값은 기존 SRPTEst 와 같다 (그림자 M2 는 기록만)
    run = load_run(est)
    model = cost_model_from_dict(json.loads(cm.read_text(encoding="utf-8")))
    for e in map(json.loads, (est / "state_estimates.jsonl").read_text(
            encoding="utf-8").splitlines()):  # fmt: skip
        r = run["rows"].get(e["request_key"])
        if r is None or e["kind"] != "grade":
            continue
        q = QueryState(r["query_id"], s_pop=r.get("s_pop"))
        q.retrieval_scores = [s or 0.0 for s in r["extra"]["retrieval_scores"]]
        q.retrieval_chars, q.n_docs = r["retrieval_chars"], len(q.retrieval_scores)
        c = next(c for c in run["calls"][e["request_key"]]
                 if c["kind"] == "grade" and c["idx"] == e["idx"])  # fmt: skip
        t = CallTicket(q, "grade", e["idx"], c["prompt_chars"], 0.0, 0)
        assert e["m0"] == pytest.approx(SRPTEst(model).base(t))


def test_generation_chars_from_question_length(setup, trained):
    """keep_all correct: 모든 점수 관측 후 기대 generation prompt 길이 = 실제 길이."""
    _, make = setup
    _, cm, sm = trained
    d = _live(make, cm, sm, "srpt_state", StateSchedRunner, "m2c")
    run = load_run(d)
    st, _ = ss.load_state_model(sm)
    n = 0
    for k, r in run["rows"].items():
        if r["corrective_action"] != "correct":
            continue
        gen = next(c for c in run["calls"][k] if c["kind"] == "generate")
        scores = r["extra"]["grading"]["scores_full"]
        o = csa.Obs(qid=r["query_id"], policy="keep_all", kind="rewrite", idx=0, chars=0,
                    n_docs=len(scores), feats=tuple(np.zeros(6)),
                    rchars=tuple(r["retrieval_chars"]), rscores=tuple(scores),
                    q_len=len(r["question"]), s_pop=None, seen=tuple(enumerate(scores)),
                    action="correct")  # fmt: skip
        assert st.exp_gen_chars(o, "correct") == gen["prompt_chars"]
        n += 1
    assert n > 0


def test_refuses_bad_configs(setup, trained, tmp_path):
    _, make = setup
    train, cm, sm = trained
    with pytest.raises(ValueError, match="parallel=false"):  # parallel grading
        _live(make, cm, sm, "srpt_state", StateSchedRunner, "x1", ["grading.parallel=true"])
    with pytest.raises(ValueError, match="aging"):  # aging 은 기존 srpt_est 구조와 다름
        _live(make, cm, sm, "srpt_state", StateSchedRunner, "x2", ["scheduler.aging_alpha=0.5"])
    cm2 = tmp_path / "cm2.json"
    cm2.write_text(cm.read_text(encoding="utf-8").replace("{", "{ ", 1), encoding="utf-8")
    with pytest.raises(
        ValueError, match="cost model 과 다름"
    ):  # state model 을 만든 cost model 과 다른 파일
        _live(make, cm2, sm, "srpt_state", StateSchedRunner, "x3")
    with pytest.raises(ValueError, match="겹침"):  # 평가 질문으로 학습한 state model (누수)
        cfg, _ = make(["dataset.split.part=eval"])
        seq = Runner(cfg, Components(Retriever(), Web(), DocEngine()), run_id="seq").run()
        sm_bad = tmp_path / "bad_state.json"
        ss.main(["fit", "--train", str(seq), "--cost-model", str(cm), "--out", str(sm_bad)])
        _live(make, cm, sm_bad, "srpt_state", StateSchedRunner, "x4")


def test_expect_and_report_end_to_end(setup, trained, tmp_path):
    """가짜 run 으로 expect / report 경로만 검증 (수치는 의미 없음)."""
    import state_sched_analysis as ssa

    _, make = setup
    _, cm, sm = trained
    cfg, _ = make(["dataset.split.part=eval"])
    seq = Runner(cfg, Components(Retriever(), Web(), DocEngine()), run_id="seq").run()
    rid = "rho0.529_s0_keep_all_K2"
    sets = ["run.mode=live", "run.purpose=pre", "dataset.split.part=eval",
            "evidence_policy=keep_all", "scheduler.slots=2",
            f"scheduler.cost_model={{path: '{cm.as_posix()}'}}",
            f"scheduler.state_model={{path: '{sm.as_posix()}'}}"]  # fmt: skip
    roots = {}
    for sched, cls in (("fcfs_query", rel.EvidenceRunner), ("srpt_est", StateSchedRunner),
                       ("srpt_state", StateSchedRunner)):  # fmt: skip
        c, _ = make([*sets, f"exp_id=PRE-SSTT-{sched}", f"scheduler.policy={sched}"])
        _, pool = split_warmup(c, load_examples(c))
        wl = make_workload([ex.id for ex in apply_split(c, pool)], None, seed=0, rate=500.0,
                           name="t")  # fmt: skip
        d = cls(c, Components(Retriever(), Web(), DocEngine()), workload=wl, run_id=rid).run()
        roots[sched] = d.parent
    common = ["--seq", str(seq), "--k", "2", "--n-boot", "50", "--out-root",
              str(tmp_path / "ana"), "--quiet"]  # fmt: skip
    out = ssa.main(["expect", "--base-root", str(roots["fcfs_query"]), "--cost-model", str(cm),
                    "--state-model", str(sm), "--speedup", "1:1,2:1.6", *common])  # fmt: skip
    res = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert {r["rule"] for r in res["decisions"]} == {"fcfs_query", "M0", "M2"}
    assert {c["scenario"] for c in res["calibration"]} == {"empirical", "speedup"}
    assert {r["policy"] for r in res["sim"]} == set(ssa.SIM_POLICIES)
    assert res["jobs"]["s0"]["n_jobs"] > 0
    assert "rules" in res["targets"]
    assert "GPU 개선율이 아니다" in (out / "report.md").read_text(encoding="utf-8")
    rep = ssa.main(["report", "--base-root", str(roots["fcfs_query"]), "--m0-root",
                    str(roots["srpt_est"]), "--m2-root", str(roots["srpt_state"]),
                    "--expect", str(out / "analysis.json"), "--block", "3", *common])  # fmt: skip
    r = json.loads((rep / "analysis.json").read_text(encoding="utf-8"))
    assert {x["scheduler"] for x in r["runs"]} == {"fcfs_query", "srpt_est", "srpt_state"}
    pairs = {(p["base"], p["new"], p["method"]) for p in r["pairs"]}
    assert ("fcfs_query", "srpt_state", "block") in pairs
    assert ("srpt_est", "srpt_state", "accuracy") in pairs
    assert len(r["overhead"]) == 2 and all(c["score_mismatch_rows"] == 0
                                           for c in r["state_check"])  # fmt: skip
    assert r["env"] and r["expect_vs_actual"] is not None
    # 같은 코드·설정 (scheduler 제외) 이면 치명적 환경 차이 없음
    assert all(not e["critical"] for e in r["env"]), r["env"]


def test_simulator_basic_properties():
    """K=1, 감속 없음: fcfs_query 는 도착 순서대로, oracle 은 짧은 작업 우선."""
    import state_sched_analysis as ssa

    def job(key, arr, svc):
        return {"key": key, "arrival": arr, "action": "correct", "kinds": ["generate"],
                "svc": [svc], "phases": [("delay", 0.0), ("call", 0)],
                "keys": [{"fcfs_query": arr, "srpt_est": svc, "srpt_state": svc,
                          "oracle": svc}]}  # fmt: skip

    jobs = [job("a", 0.0, 5.0), job("b", 0.1, 3.0), job("c", 0.2, 1.0)]
    one = lambda kind, n: 1.0  # noqa: E731
    fq = ssa.simulate(jobs, "fcfs_query", 1, one)
    assert fq["a"]["latency"] == pytest.approx(5.0)
    assert fq["b"]["latency"] == pytest.approx(8.0 - 0.1)
    assert fq["c"]["latency"] == pytest.approx(9.0 - 0.2)
    orc = ssa.simulate(jobs, "oracle", 1, one)
    assert orc["c"]["latency"] == pytest.approx(6.0 - 0.2)  # a 다음 c (짧음) 먼저
    two = ssa.simulate(jobs, "fcfs_query", 1, lambda kind, n: 2.0)
    assert two["a"]["latency"] == pytest.approx(10.0)


def _dry_live_args(script: str, env: dict) -> list[list[str]]:
    import os
    import shlex
    import subprocess

    from crag_fixtures import ROOT

    out = subprocess.run(
        ["bash", script, "live"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1", **env},
        check=True,
    ).stdout
    runs = []
    for line in out.splitlines():
        if line.startswith("+ uv run python experiments/crag_system/run_"):
            toks = shlex.split(line[2:].replace("<nominal>", "X"))
            sets = [toks[i + 1] for i, t in enumerate(toks) if t == "--set"]
            runs.append(sorted(s for s in sets if not s.startswith(
                ("scheduler.policy=", "scheduler.state_model=", "run.session="))))  # fmt: skip
    return runs


def test_live_args_match_eql1_baseline():
    """신규 live 인자 = 기존 EQL1 fcfs_query 인자 (policy / state_model / session 만 다름)."""
    import shutil

    if shutil.which("bash") is None:
        pytest.skip("bash 없음")
    new = _dry_live_args("experiments/crag_system/run_state_sched.sh", {})
    old = _dry_live_args("experiments/crag_system/run_equal_load.sh", {"WORK_RATIO": "1.513"})
    assert len(new) == 4 and len(old) == 2
    assert all(n == old[0] for n in new), (new[0], old[0])


def test_check_command(setup, trained, tmp_path):
    from workload import save_workload

    _, make = setup
    _, cm, sm = trained
    cfg, _ = make(["dataset.split.part=eval"])
    _, pool = split_warmup(cfg, load_examples(cfg))
    wl = make_workload([ex.id for ex in apply_split(cfg, pool)], None, seed=0, rate=500.0,
                       name="EQLT_keep_all_rhoeff0.8_s0")  # fmt: skip
    wl["params"]["rho"] = 0.529
    wl["equal_load"] = {"policy": "keep_all"}
    wp = save_workload(wl, tmp_path / "wl.json")
    rid = "rho0.529_s0_keep_all_K2"
    c, _ = make(["run.mode=live", "run.purpose=pre", "dataset.split.part=eval",
                 "evidence_policy=keep_all", "scheduler.slots=2", "exp_id=PRE-EQLT-LIVE",
                 "scheduler.policy=fcfs_query",
                 f"scheduler.cost_model={{path: '{cm.as_posix()}'}}"])  # fmt: skip
    base = rel.EvidenceRunner(c, Components(Retriever(), Web(), DocEngine()),
                              workload=wl, run_id=rid).run()  # fmt: skip
    args = ["check", "--state-model", str(sm), "--cost-model", str(cm), "--base-root",
            str(base.parent), "--workload", str(wp), "--k", "2",
            "--new-root", str(tmp_path / "new")]  # fmt: skip
    with pytest.raises(SystemExit) as e:
        ss.main(args)
    assert e.value.code == 0
    (tmp_path / "new" / rid).mkdir(parents=True)  # 중단된 폴더 -> 실패
    with pytest.raises(SystemExit) as e:
        ss.main(args)
    assert e.value.code == 1
    with pytest.raises(SystemExit) as e:  # K 불일치
        ss.main([*args[:-2], "--k", "4"])
    assert e.value.code == 1
