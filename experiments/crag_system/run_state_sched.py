"""상태 기반 SRPT-est (M2) live runner. 인자는 run_evidence_live.py 와 같다.

EvidenceRunner (evidence_policy, 기존 CRAGGraph 하위 클래스) 를 그대로 쓰고 다음만 더한다.
  - scheduler.policy = srpt_est | srpt_state 일 때 gate 정책을 state_sched.StateSRPT 로 만든다
    (srpt_est 는 값이 기존 SRPTEst 와 같음. 다른 정책 이름은 기존 build_policy 그대로)
  - GradeTap: grader LLM 응답에서 grade 점수만 ScoreBoard 에 기록 (파이프라인 동작 불변)
  - 결과 폴더에 state_estimates.jsonl (호출별 M0 / M2 / 사용값), scheduler_overhead.json,
    state_check.json (tap 점수 = 결과 grader 점수, idx 정합성) 를 새로 쓴다

설정
  scheduler.state_model: {path: <state_sched.py fit 결과>}   (srpt_est / srpt_state 에 필수)
  scheduler.state_shadow: true    srpt_est 에서 M2 그림자 계산 (기록만). false 면 tap 만 연결
  grading: full, parallel=false 만 허용 (호출 순서 = 문서 순서)

    uv run python experiments/crag_system/run_state_sched.py \
        --config experiments/crag_system/config.yaml --overlay <engine profile> \
        --set evidence_policy=keep_all --set scheduler.policy=srpt_state \
        --set "scheduler.state_model={path: <state m2 json>}" \
        --exp-id PRE-SST1-LIVE-srpt_state --run-id rho0.529_s0_keep_all_K4 --workload <EQL1 wl> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evidence_live import EvidenceRunner  # noqa: E402
from runner import load_layered_config  # noqa: E402
from state_sched import (  # noqa: E402
    STATE_POLICIES,
    GradeTap,
    ScoreBoard,
    StateSRPT,
    file_sha,
    load_state_model,
    overhead_summary,
)
from workload import load_workload  # noqa: E402

from rag.scheduling import SchedulingGate  # noqa: E402


def check_grading(cfg: dict) -> None:
    g = cfg.get("grading") or {}
    if g.get("strategy", "full") != "full" or g.get("parallel"):
        raise ValueError("srpt_est / srpt_state (state runner) 는 grading.strategy=full, "
                         "parallel=false 에서만 (grade 호출 순서 = 문서 순서)")  # fmt: skip


class StateSchedRunner(EvidenceRunner):
    def __init__(self, cfg: dict, *args, **kwargs):
        s = cfg.get("scheduler") or {}
        self.sched_name = s.get("policy")
        self.board = ScoreBoard()
        self.state_policy: StateSRPT | None = None
        self.tables = self.state_meta = None
        if self.sched_name in STATE_POLICIES:
            check_grading(cfg)
            spec = s.get("state_model") or {}
            if not spec.get("path"):
                raise ValueError(f"{self.sched_name}: scheduler.state_model.path 가 필요")
            self.tables, self.state_meta = load_state_model(spec["path"])
            c = cfg.get("crag") or {}
            th = (c.get("upper_threshold", 0.7), c.get("lower_threshold", 0.3))
            if (self.tables.upper, self.tables.lower) != tuple(float(x) for x in th):
                raise ValueError(f"state model threshold {(self.tables.upper, self.tables.lower)}"
                                 f" != config {th}")  # fmt: skip
            cm = (s.get("cost_model") or {}).get("path")
            if not cm:
                raise ValueError("state runner 는 scheduler.cost_model.path (고정 파일) 가 필요")
            if file_sha(cm) != self.state_meta["cost_model_sha256"]:
                raise ValueError(f"state model 을 만든 cost model 과 다름: {cm}")
        super().__init__(cfg, *args, **kwargs)

    def _build_gate(self, cost_model):
        if self.sched_name not in STATE_POLICIES:
            return super()._build_gate(cost_model)
        s = self.cfg.get("scheduler") or {}
        if s.get("aging_alpha") or s.get("wait_threshold") is not None:
            raise ValueError(
                "state runner 는 aging / wait_threshold 없이 (기존 srpt_est 와 같은 구조)"
            )
        ev = {r["query_id"] for r in self.requests}
        overlap = sorted(set(self.state_meta.get("train_query_ids") or []) & ev)
        self.preflight["state_model_leakage"] = {"train_eval_overlap": len(overlap),
                                                 "examples": overlap[:5]}  # fmt: skip
        if overlap and not s.get("allow_cost_model_overlap"):
            raise ValueError(f"state model 학습 질문이 평가 workload 와 {len(overlap)}개 겹침")
        use = "M2" if self.sched_name == "srpt_state" else "M0"
        self.state_policy = StateSRPT(
            cost_model, self.tables, self.board, self.cfg["evidence_policy"], use=use,
            shadow=bool(s.get("state_shadow", True)), state_meta=self.state_meta,
        )  # fmt: skip
        return SchedulingGate(self.state_policy, slots=s.get("slots", 1),
                              engine_priority=s.get("engine_priority", "none"))  # fmt: skip

    def _run_pipeline(self, ex, llm, grader_llm, q, web=None):
        if self.state_policy is not None:
            self.board.start(q.request_key, len(ex.question))
            grader_llm = GradeTap(grader_llm, self.board, q.request_key)
        return super()._run_pipeline(ex, llm, grader_llm, q, web)

    def run(self) -> Path:
        run_dir = super().run()
        if self.state_policy is None:
            return run_dir
        log = self.state_policy.log
        with open(run_dir / "state_estimates.jsonl", "x", encoding="utf-8") as f:
            for e in log:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        calls = [json.loads(x) for x in (run_dir / "calls.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]  # fmt: skip
        wall = sum(c.get("wall") or 0.0 for c in calls if not c.get("warmup"))
        ov = overhead_summary(log, wall)
        ov["policy"] = self.state_policy.name
        (run_dir / "scheduler_overhead.json").open("x", encoding="utf-8").write(
            json.dumps(ov, indent=2)
        )
        (run_dir / "state_check.json").open("x", encoding="utf-8").write(
            json.dumps(self.state_check(calls), indent=2, ensure_ascii=False)
        )
        return run_dir

    def state_check(self, calls: list[dict]) -> dict:
        """tap 점수 = 결과의 grader 점수, tap idx = recorder idx, 로그 수 = gate dispatch 수."""
        board = self.board.items()
        n_rows = mism = idx_bad = 0
        for r in self.rows:
            if r.get("error"):
                continue
            n_rows += 1
            tap = board.get(f"{r['request_index']}:{r['query_id']}") or {"scores": {}}
            got = [tap["scores"].get(i) for i in range(len(tap["scores"]))]
            ref = list(((r.get("extra") or {}).get("grading") or {}).get("scores_full") or [])
            mism += got != ref
        by_key: dict = {}
        for c in calls:
            if c["kind"] == "grade":
                by_key.setdefault(f"{c['request_index']}:{c['query_id']}", []).append(c["idx"])
        for k, idxs in by_key.items():
            tap = board.get(k) or {"scores": {}}
            idx_bad += sorted(idxs) != sorted(tap["scores"])
        log = self.state_policy.log
        return {"rows_checked": n_rows, "score_mismatch_rows": mism,
                "grade_idx_mismatch_rows": idx_bad, "estimates_logged": len(log),
                "gate_dispatches": len(self.gate.log),
                "m0_eq_m2_share_generate": _share(
                    [e["m0"] == e["m2"] for e in log if e["kind"] == "generate"
                     and e["m2"] is not None]),
                }  # fmt: skip


def _share(xs):
    return sum(xs) / len(xs) if xs else None


def main(argv=None) -> Path:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--workload", default=None)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    load_dotenv()
    cfg = load_layered_config(a.config, a.overlay, a.set)
    cfg["exp_id"] = a.exp_id
    if (cfg.get("run") or {}).get("purpose") == "paper":
        raise SystemExit("진단 실험은 run.purpose=paper 로 실행하지 않음")
    workload = load_workload(a.workload) if a.workload else None
    run_dir = StateSchedRunner(cfg, workload=workload, run_id=a.run_id).run()
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
