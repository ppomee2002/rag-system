"""Evidence preservation 정책을 live (또는 sequential) 로 실행하는 별도 runner.
인자는 run.py 와 같다.

기존 CRAGGraph / Runner / grading / gate / scheduler 는 바꾸지 않는다
(하위 클래스로 contexts 만 바꿈).
설정: config 최상위 `evidence_policy:` (필수). 규칙표는 evidence_filter_ablation.CONDITIONS 와 같다.

  crag_rule             기존 C-RAG 그대로 (같은 runner 경로로 실행해 비교 조건을 맞추기 위함)
  no_correct_filter     correct: FAISS top-k
  no_ambiguous_filter   ambiguous: FAISS top-k + web
  no_incorrect_discard  incorrect: FAISS top-k + web
  keep_all              세 규칙 모두
  (vanilla_ctx 는 grading·web 을 하고도 쓰지 않는 조건이라 live 에서는 허용하지 않음.
   Vanilla 기준이 필요하면 run.py --set pipeline=vanilla)

바뀌지 않는 것: grading 호출·점수·action 판정, rewrite / web search 호출, stage 이름과 StageTimer,
LLM 호출 수. 결과 구분: queries.jsonl pipeline = "crag_graph:<grading>+evp_<정책>",
extra.evidence_policy, config.yaml 의 evidence_policy.

    uv run python experiments/crag_system/run_evidence_live.py \
        --config experiments/crag_system/config.yaml \
        --overlay <engine profile> --set evidence_policy=keep_all --exp-id PRE-EVC-SV1-LIVE ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_filter_ablation import CONDITIONS  # noqa: E402
from runner import Runner, load_layered_config  # noqa: E402
from workload import load_workload  # noqa: E402

from rag.graph import CRAGGraph  # noqa: E402

LIVE_POLICIES = tuple(c for c in CONDITIONS if c != "vanilla_ctx")


def parse_policy(cfg: dict) -> str:
    pol = cfg.get("evidence_policy")
    if pol not in LIVE_POLICIES:
        raise ValueError(f"evidence_policy 는 {LIVE_POLICIES} 중 하나 (got {pol!r})")
    if cfg.get("pipeline", "crag_graph") != "crag_graph":
        raise ValueError(f"pipeline=crag_graph 에서만 (got {cfg.get('pipeline')})")
    if "ablation" in cfg:
        raise ValueError("ablation 키와 함께 쓰지 않음 (run_ablation.py 와 별개)")
    return pol


class EvidenceCRAGGraph(CRAGGraph):
    """decide / web_search 노드 결과의 contexts 만 정책대로 바꾼다."""

    def __init__(self, *args, policy: str, **kwargs):
        self.policy = policy
        self.rule = CONDITIONS[policy]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_graph(cls, g: CRAGGraph, policy: str) -> EvidenceCRAGGraph:
        return cls(
            g.retriever, g.llm, g.web_search, grader=g.grader, grader_llm=g.grader_llm,
            top_k=g.top_k, web_top_k=g.web_top_k, upper_threshold=g.upper,
            lower_threshold=g.lower, name=f"{g.name}+evp_{policy}", policy=policy,
        )  # fmt: skip

    def _decide(self, state, config) -> dict:
        out = super()._decide(state, config)
        if out["action"] == "correct" and self.rule["correct"]:
            out["contexts"] = list(state["docs"])
        return out

    def _web_search(self, state, config) -> dict:
        out = super()._web_search(state, config)
        if self.rule.get(state["action"]):  # ambiguous / incorrect: FAISS top-k + web
            out["contexts"] = list(state["docs"]) + list(out["web_docs"])
        return out

    def run(self, *args, **kwargs):
        res = super().run(*args, **kwargs)
        res.extra["evidence_policy"] = self.policy
        return res


class EvidenceRunner(Runner):
    def __init__(self, cfg: dict, *args, **kwargs):
        policy = parse_policy(cfg)  # 결과 폴더를 만들기 전에 검증
        super().__init__(cfg, *args, **kwargs)
        self.graph = EvidenceCRAGGraph.from_graph(self.graph, policy)


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
    run_dir = EvidenceRunner(cfg, workload=workload, run_id=a.run_id).run()
    print(f"saved -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
