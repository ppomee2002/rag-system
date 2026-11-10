"""SYS 공통 실험 runner.

설정으로 바꾸는 축: pipeline / grading / scheduler / llm (engine) / workload / run.mode / seed.

  mode=sequential : 요청을 하나씩 실행 (부하 없음). 정확도·단계별 latency·trace (cost model 입력)
  mode=live       : workload 의 도착 시각대로 open-loop 동시 실행. 스케줄링·동시 부하 실험

실험 구분 (run.purpose, 결과 폴더와 exp_id 접두사로 분리):
  smoke : 환경·기능 확인. exp_id SMOKE-*. 결과를 논문에 쓰지 않음
  pre   : 예비 측정 (엔진 비교, 품질 확인, K calibration). exp_id PRE-*
  paper : 논문 본실험 SYS-1~4. exp_id SYS-<n>*. --paper-approved 와 아래 조건이 모두 필요
          (git 작업트리 clean, 엔진 프로필·launch 기록 지정, smoke launch (--enforce-eager) 아님,
           --limit 로 질문 수를 줄이지 않음)

결과: results/crag_system/<purpose>/<exp_id>/<run_id>/
  config.yaml, run_meta.json   실행 전 기록 (git, 패키지, GPU 상태, 엔진 정보, workload hash)
  queries.jsonl                요청 단위 결과 (warmup 행은 warmup=true)
  calls.jsonl                  LLM 호출 단위 기록 (request_index 로 queries 와 연결)
  scheduler_log.jsonl          gate dispatch 기록 (scheduler 사용 시)
  run_end.json, summary.json   종료 시 GPU 상태·엔진 지표 변화·오류 수 / 요약 통계
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag.config import load_layered_config, parse_set  # noqa: F401 (re-export)
from rag.data.loader import load_qa_jsonl
from rag.eval.metrics import score_all
from rag.grading import build_grader
from rag.graph import CRAGGraph
from rag.llm.engine_llm import EngineLLM
from rag.llm.factory import build_engine
from rag.pipelines.crag import CRAG
from rag.pipelines.vanilla import VanillaRAG
from rag.scheduling import QueryState, SchedulingGate, build_policy
from rag.scheduling.cost_model import (
    cost_model_from_dict,
    fit_cost_model,
    fit_oof_cost_model,
    job_from_row,
)
from rag.scheduling.preflight import (
    check_cost_model_leakage,
    check_launch_meta,
    check_priority_config,
    probe_server_priority,
)
from rag.telemetry.calls import CallRecorder
from rag.telemetry.engine_metrics import MetricsSampler
from rag.telemetry.run_meta import (
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    git_info,
    gpu_snapshot,
    write_new,
    write_run_files,
)
from rag.types import QAExample

PIPELINES = ("crag_graph", "crag", "vanilla")
PURPOSE_PREFIX = {"smoke": "SMOKE-", "pre": "PRE-", "paper": "SYS-"}


class PurposeError(ValueError):
    pass


def check_purpose(cfg: dict, paper_approved: bool = False, git: dict | None = None) -> dict:
    """smoke / pre / paper 구분 검사. paper (SYS-1~4) 는 별도 승인과 재현 조건을 요구한다."""
    run = cfg.get("run") or {}
    purpose = run.get("purpose", "smoke")
    exp_id = cfg.get("exp_id", "")
    if purpose not in PURPOSE_PREFIX:
        raise PurposeError(f"run.purpose: {tuple(PURPOSE_PREFIX)} (got {purpose})")
    prefix = PURPOSE_PREFIX[purpose]
    if not exp_id.startswith(prefix):
        raise PurposeError(f"purpose={purpose} 의 exp_id 는 {prefix}* 이어야 함 (got {exp_id})")
    for other, pre in PURPOSE_PREFIX.items():
        if other != purpose and exp_id.startswith(pre):
            raise PurposeError(f"exp_id {exp_id} 는 purpose={other} 용")
    if purpose != "paper":
        return {"purpose": purpose, "exp_id": exp_id}
    problems = []
    if not paper_approved:
        problems.append("SYS 본실험은 별도 승인 후 --paper-approved 로만 실행")
    git = git if git is not None else git_info()
    if git.get("dirty") is not False:
        problems.append(f"git 작업트리가 clean 하지 않음 (dirty={git.get('dirty')})")
    if run.get("limit_override") is not None:
        problems.append("--limit 로 질문 수를 줄인 run 은 본실험이 될 수 없음")
    llm = cfg.get("llm") or {}
    if llm.get("provider") == "vllm":
        if not cfg.get("engine_profile"):
            problems.append("engine_profile (엔진 프로필 overlay) 이 필요")
        if not llm.get("launch_meta"):
            problems.append("llm.launch_meta (launch 기록) 가 필요")
    if problems:
        raise PurposeError("본실험 조건 미충족: " + "; ".join(problems))
    return {
        "purpose": purpose,
        "exp_id": exp_id,
        "paper_approved": True,
        "git_commit": git.get("commit"),
    }


# ---------------------------------------------------------------- config
# ---------------------------------------------------------------- components
class SharedRetriever:
    """FAISS + encoder 를 여러 thread 가 공유할 때 호출을 직렬화 (원본: correction_scheduling)."""

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()

    def retrieve(self, query, top_k):
        with self.lock:
            return self.inner.retrieve(query, top_k)


class QueryRetriever:
    """기존 CRAG / Vanilla 클래스용: 최초 retrieval 결과를 QueryState 에 남긴다."""

    def __init__(self, inner, state: QueryState):
        self.inner, self.state = inner, state

    def retrieve(self, query, top_k):
        docs = self.inner.retrieve(query, top_k)
        self.state.on_retrieved(docs)
        return docs


@dataclass
class Components:
    retriever: Any
    web_search: Any
    engine: Any
    grader_engine: Any = None
    cost_model: Any = None  # CostModel | OOFCostModel
    closers: list = field(default_factory=list)


def load_examples(cfg: dict) -> list[QAExample]:
    """crag_latency / correction_scheduling 과 같은 seed shuffle + limit (같은 query 셋)."""
    examples = load_qa_jsonl(cfg["dataset"]["path"])
    random.Random(cfg["seed"]).shuffle(examples)
    limit = cfg["dataset"].get("limit")
    return examples[:limit] if limit else examples


def split_warmup(cfg: dict, examples: list[QAExample]) -> tuple[list, list]:
    n = cfg.get("run", {}).get("warmup_queries", 0) or 0
    return examples[:n], examples[n:]


def apply_split(cfg: dict, pool: list[QAExample]) -> list[QAExample]:
    """dataset.split: {part: train | eval | all, train_frac, seed}

    cost model 학습용 (train) 과 스케줄링 평가용 (eval) 질문을 분리한다.
    같은 설정이면 항상 같은 분할.
    """
    sp = (cfg.get("dataset") or {}).get("split") or {}
    part = sp.get("part", "all")
    if part == "all":
        return pool
    if part not in ("train", "eval"):
        raise ValueError("dataset.split.part: train | eval | all")
    ids = sorted(ex.id for ex in pool)
    random.Random(sp.get("seed", 0)).shuffle(ids)
    n_train = int(round(len(ids) * sp.get("train_frac", 0.3)))
    chosen = set(ids[:n_train] if part == "train" else ids[n_train:])
    return [ex for ex in pool if ex.id in chosen]


def load_trace(run_dir: str | Path) -> list[dict]:
    """runner 결과 폴더 -> calls 를 붙인 query 행 (warmup / 오류 제외)."""
    run_dir = Path(run_dir)
    calls: dict[int, list] = {}
    for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        calls.setdefault(c["request_index"], []).append(c)
    rows = []
    for line in (run_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("warmup") or r.get("error"):
            continue
        rows.append({**r, "calls": calls.get(r["request_index"], [])})
    return rows


def load_cost_model(spec: dict | None):
    """scheduler.cost_model:
    {path: model.json}  또는  {trace: <sequential run 폴더>, service_key: wall, oof_folds: null|k}
    """
    if not spec:
        return None
    if spec.get("path"):
        return cost_model_from_dict(json.loads(Path(spec["path"]).read_text(encoding="utf-8")))
    rows = load_trace(spec["trace"])
    jobs = [job_from_row(r, spec.get("service_key", "wall")) for r in rows]
    k = spec.get("oof_folds")
    model = fit_oof_cost_model(jobs, k, spec.get("oof_seed", 0)) if k else fit_cost_model(jobs)
    model.meta.update(trace=str(spec["trace"]), service_key=spec.get("service_key", "wall"))
    return model


def replay_sources(wcfg: dict) -> list[Path]:
    """replay_paths + replay_from_runs (각 run 폴더의 web_cache.jsonl)."""
    paths = [Path(p) for p in wcfg.get("replay_paths") or []]
    paths += [Path(r) / "web_cache.jsonl" for r in wcfg.get("replay_from_runs") or []]
    if not paths:
        raise ValueError("provider=replay 에는 replay_paths 또는 replay_from_runs 가 필요")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"web replay 기록이 없음: {missing}")
    return paths


def build_web_search(cfg: dict, run_dir: Path):
    from rag.retrieval.web_cache import RecordingWebSearch, ReplayWebSearch

    wcfg = cfg["crag"]["web_search"]

    def tavily():
        from rag.retrieval.tavily_search import TavilyWebSearch

        return TavilyWebSearch(search_depth=wcfg.get("search_depth", "basic"))

    provider = wcfg.get("provider", "tavily")
    if provider == "tavily":
        ws = tavily()
        return RecordingWebSearch(ws, run_dir / "web_cache.jsonl") if wcfg.get("record") else ws
    if provider == "replay":
        fallback = tavily() if wcfg.get("replay_fallback") else None
        if fallback is not None and wcfg.get("record"):
            fallback = RecordingWebSearch(fallback, run_dir / "web_cache.jsonl")
        return ReplayWebSearch(
            replay_sources(wcfg), wcfg.get("latency_mode", "recorded"), fallback=fallback
        )
    raise ValueError(f"지원하지 않는 web_search provider: {provider}")


def build_components(cfg: dict, run_dir: Path) -> Components:
    from rag.retrieval.faiss_retriever import FaissRetriever

    rcfg = cfg["retriever"]
    retriever = SharedRetriever(
        FaissRetriever(
            index_path=rcfg["index_path"],
            docs_path=rcfg["docs_path"],
            model_name=cfg["embedding"]["model"],
            query_prefix=rcfg.get("query_prefix", ""),
            device=rcfg.get("device"),
        )
    )
    gcfg = (cfg.get("crag") or {}).get("grader") or {}
    grader_engine = (
        build_engine({**cfg["llm"], **{k: v for k, v in gcfg.items() if v is not None}})
        if gcfg.get("provider")
        else None
    )
    return Components(
        retriever=retriever,
        web_search=build_web_search(cfg, run_dir),
        engine=build_engine(cfg["llm"]),
        grader_engine=grader_engine,
    )


# ---------------------------------------------------------------- runner
def metric_delta(before, after, needle: str) -> dict | None:
    """vLLM /metrics 에서 이름에 needle 이 들어간 counter 의 변화량 (예: prefix cache hit)."""
    if not isinstance(before, dict) or not isinstance(after, dict) or "error" in after:
        return None
    return {k: after[k] - before.get(k, 0.0) for k in after if needle in k}


def _percentiles(x) -> dict:
    if len(x) == 0:
        return {"n": 0}
    a = np.asarray(x, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def summarize(rows: list[dict], mode: str) -> dict:
    ok = [r for r in rows if not r.get("warmup") and not r.get("error")]
    out = {
        "n_measured": len([r for r in rows if not r.get("warmup")]),
        "n_ok": len(ok),
        "n_errors": len([r for r in rows if r.get("error")]),
    }
    if not ok:
        return out
    corr = [r for r in ok if r.get("corrective_action") not in (None, "correct")]
    out.update(
        accuracy=float(np.mean([r["acc"] for r in ok])),
        f1=float(np.mean([r["f1"] for r in ok])),
        latency=_percentiles([r["latency"] for r in ok]),
        latency_corrected=_percentiles([r["latency"] for r in corr]),
        latency_not_corrected=_percentiles([r["latency"] for r in ok if r not in corr]),
        action_counts={
            a: sum(r.get("corrective_action") == a for r in ok)
            for a in ("correct", "ambiguous", "incorrect", None)
        },
        mean_llm_calls=float(np.mean([r["calls_summary"]["n_calls"] for r in ok])),
        mean_prompt_tokens=float(np.mean([r["calls_summary"]["prompt_tokens"] or 0 for r in ok])),
        mean_out_tokens=float(np.mean([r["calls_summary"]["out_tokens"] or 0 for r in ok])),
        mean_gate_wait=float(np.mean([r["query_state"]["cum_wait"] for r in ok])),
    )
    if mode == "live":
        span = max(r["finish"] for r in ok) - min(r["arrival"] for r in ok)
        out["throughput_rps"] = len(ok) / span if span > 0 else None
    return out


class Runner:
    def __init__(
        self,
        cfg: dict,
        components: Components | None = None,
        workload: dict | None = None,
        run_id: str | None = None,
        paper_approved: bool = False,
    ):
        self.cfg = cfg
        self.purpose_check = check_purpose(cfg, paper_approved)
        self.mode = cfg.get("run", {}).get("mode", "sequential")
        if self.mode not in ("sequential", "live"):
            raise ValueError("run.mode: sequential | live")
        self.pipeline_name = cfg.get("pipeline", "crag_graph")
        if self.pipeline_name not in PIPELINES:
            raise ValueError(f"pipeline: {PIPELINES}")
        # 설정 검증을 먼저 끝낸 뒤 결과 폴더를 만든다 (잘못된 설정이 빈 폴더를 남기지 않도록)
        self.grader = build_grader(cfg.get("grading"))
        if self.pipeline_name != "crag_graph" and self.grader.name not in ("full", "full_parallel"):
            raise ValueError(
                f"pipeline={self.pipeline_name} 는 grading 전략을 쓰지 않음 (full 만 가능). "
                "Lazy / Batch / Adaptive 는 pipeline=crag_graph"
            )
        self.examples = load_examples(cfg)
        self.warmup, pool = split_warmup(cfg, self.examples)
        pool = apply_split(cfg, pool)
        self.by_id = {ex.id: ex for ex in self.examples}
        self.workload = workload
        self.requests = self._requests(pool)
        sched = cfg.get("scheduler") or {}
        self.preflight: dict = {"priority": check_priority_config(sched, cfg["llm"])}
        if cfg["llm"].get("launch_meta"):
            self.preflight["launch_meta"] = check_launch_meta(
                cfg["llm"]["launch_meta"], cfg["llm"], cfg.get("server"), cfg.get("engine_profile")
            )
            if self.purpose_check["purpose"] == "paper" and self.preflight["launch_meta"].get(
                "enforce_eager"
            ):
                raise PurposeError("smoke launch (--enforce-eager) 서버로는 본실험을 할 수 없음")
        cost_model = components.cost_model if components else None
        if cost_model is None:
            cost_model = load_cost_model(sched.get("cost_model"))
        if sched.get("policy", "none") not in (None, "none"):
            self.preflight["cost_model_leakage"] = check_cost_model_leakage(
                cost_model,
                {r["query_id"] for r in self.requests},
                allow_overlap=bool(sched.get("allow_cost_model_overlap", False)),
            )
        self.gate = self._build_gate(cost_model)

        purpose, exp_id = self.purpose_check["purpose"], self.purpose_check["exp_id"]
        self.run_dir = create_run_dir(
            f"crag_system/{purpose}/{exp_id}", cfg.get("output_dir", "results"), run_id
        )
        self.comp = components or build_components(cfg, self.run_dir)
        self.comp.cost_model = cost_model
        c = cfg.get("crag", {})
        self.upper = c.get("upper_threshold", 0.7)
        self.lower = c.get("lower_threshold", 0.3)
        self.top_k = cfg["retriever"].get("top_k", 5)
        self.web_top_k = (c.get("web_search") or {}).get("top_k", 5)
        self.graph = None
        if self.pipeline_name == "crag_graph":
            self.graph = CRAGGraph(
                self.comp.retriever,
                None,
                self.comp.web_search,
                grader=self.grader,
                top_k=self.top_k,
                web_top_k=self.web_top_k,
                upper_threshold=self.upper,
                lower_threshold=self.lower,
            )
        self.queries = JsonlWriter(self.run_dir / "queries.jsonl")
        self.calls = JsonlWriter(self.run_dir / "calls.jsonl")
        self.rows: list[dict] = []
        self.rows_lock = threading.Lock()

    def _requests(self, pool: list[QAExample]) -> list[dict]:
        if self.workload is None:
            if self.mode == "live":
                raise ValueError("live 모드는 workload (도착 시각) 가 필요")
            return [{"i": i, "query_id": ex.id, "arrival": None} for i, ex in enumerate(pool)]
        pool_ids = {ex.id for ex in pool}
        missing = {r["query_id"] for r in self.workload["requests"]} - pool_ids
        if missing:
            raise ValueError(
                f"workload query 가 dataset pool 에 없음 (seed/limit/warmup 확인): "
                f"{sorted(missing)[:5]}"
            )
        if self.mode == "live" and any(r["arrival"] is None for r in self.workload["requests"]):
            raise ValueError("live 모드에는 poisson workload 가 필요")
        return self.workload["requests"]

    def _build_gate(self, cost_model) -> SchedulingGate | None:
        s = self.cfg.get("scheduler") or {}
        if s.get("policy", "none") in (None, "none"):
            return None
        policy = build_policy(s, cost_model)
        return SchedulingGate(
            policy, slots=s.get("slots", 1), engine_priority=s.get("engine_priority", "none")
        )

    # ------------------------------------------------------------ one request
    def _execute(self, req: dict, t0: float, warmup: bool) -> dict:
        ex = self.by_id[req["query_id"]]
        # 도착 시각은 live 측정 요청에만 의미가 있다 (sequential / warmup 은 latency = 실행 시간)
        arrival = req["arrival"] if self.mode == "live" and not warmup else None
        t_start = time.perf_counter() - t0
        q = QueryState(
            ex.id,
            arrival=arrival if arrival is not None else t_start,
            s_pop=ex.metadata.get("s_pop"),
            t0=t0,
            request_key=f"{req['i']}:{ex.id}",
        )
        rec = CallRecorder(t0, record_texts=bool(self.cfg.get("run", {}).get("record_texts")))
        kw = {"recorder": rec, "query_state": q}
        if self.gate is not None:
            kw["scheduler"] = self.gate
        llm = EngineLLM(self.comp.engine, **kw)
        grader_llm = EngineLLM(self.comp.grader_engine, **kw) if self.comp.grader_engine else llm
        row: dict = {
            "request_index": req["i"],
            "query_id": ex.id,
            "warmup": warmup,
            "pipeline": self.pipeline_name,
            "grading": self.grader.name,
            "arrival": arrival,
            "t_start": t_start,
            "start_lag": None if arrival is None else t_start - arrival,
        }
        try:
            res = self._run_pipeline(ex, llm, grader_llm, q)
            d = asdict(res)
            d["contexts"] = [c["id"] for c in d["contexts"]]
            row.update(d)
            row.update(score_all(res.answer, ex.answers))
            q.finish(res.corrective_action)
        except Exception as e:  # 기록하고 계속
            row["error"] = f"{type(e).__name__}: {e}"[:1000]
            q.finish()
        finish = time.perf_counter() - t0
        row.update(
            finish=finish,
            latency=finish - (arrival if arrival is not None else t_start),
            gold_answers=ex.answers,
            s_pop=ex.metadata.get("s_pop"),
            retrieval_chars=q.retrieval_chars,
            retrieval_doc_ids=q.retrieval_doc_ids,
            query_state=q.snapshot(),
            calls_summary=rec.summary(),
        )
        for c in rec.to_dicts():
            self.calls.write({"request_index": req["i"], "query_id": ex.id, "warmup": warmup, **c})
        self.queries.write(row)
        with self.rows_lock:
            self.rows.append(row)
        return row

    def _run_pipeline(self, ex, llm, grader_llm, q):
        if self.pipeline_name == "crag_graph":
            return self.graph.run(ex, llm=llm, grader_llm=grader_llm, query_state=q)
        retriever = QueryRetriever(self.comp.retriever, q)
        if self.pipeline_name == "vanilla":
            return VanillaRAG(retriever, llm, top_k=self.top_k).run(ex)
        parallel = (self.cfg.get("grading") or {}).get("parallel", False)
        return CRAG(
            retriever,
            llm,
            self.comp.web_search,
            grader_llm=grader_llm,
            top_k=self.top_k,
            web_top_k=self.web_top_k,
            upper_threshold=self.upper,
            lower_threshold=self.lower,
            parallel_grading=parallel,
        ).run(ex)

    # ------------------------------------------------------------ run
    def _comparability_record(self) -> dict:
        """조건 간 비교에서 같아야 하는 값과 교란 요인 (캐시, 실행 순서) 기록."""
        cfg = self.cfg
        wcfg = (cfg.get("crag") or {}).get("web_search") or {}
        sources = []
        if wcfg.get("provider") == "replay":
            for pth in replay_sources(wcfg):
                digest = hashlib.sha256(pth.read_bytes()).hexdigest()
                sources.append({"path": str(pth), "sha256": digest})
        ids = [r["query_id"] for r in self.requests]
        return {
            "eval_query_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "n_eval_queries": len(set(ids)),
            "warmup_query_ids": [ex.id for ex in self.warmup],
            "dataset": cfg.get("dataset"),
            "retriever": cfg.get("retriever"),
            "embedding": cfg.get("embedding"),
            "llm": {k: v for k, v in cfg["llm"].items() if k != "api_key_env"},
            "thresholds": {"upper": self.upper, "lower": self.lower},
            "web_search": {
                "provider": wcfg.get("provider"),
                "latency_mode": wcfg.get("latency_mode"),
                "record": wcfg.get("record"),
                "sources": sources,
            },
            "prefix_cache_reset_requested": bool(cfg.get("run", {}).get("reset_prefix_cache")),
            "session": cfg.get("run", {}).get("session"),  # {id, order_index}: 조건 실행 순서
        }

    def _start_sampler(self) -> MetricsSampler | None:
        """run.sample_engine_metrics_sec 가 있으면 측정 구간 동안 vLLM /metrics 를 기록."""
        interval = self.cfg.get("run", {}).get("sample_engine_metrics_sec")
        fn = getattr(self.comp.engine, "metrics_text", None)
        if not interval or fn is None:
            return None
        from rag.llm.vllm_engine import parse_prometheus

        return MetricsSampler(lambda: parse_prometheus(fn()), float(interval), t0=self.t0).start()

    def _engine_metrics(self):
        fn = getattr(self.comp.engine, "metrics_text", None)
        if fn is None:
            return None
        try:
            from rag.llm.vllm_engine import parse_prometheus

            return parse_prometheus(fn())
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self) -> Path:
        engines = {"main": self.comp.engine}
        if self.comp.grader_engine is not None:
            engines["grader"] = self.comp.grader_engine
        extra = {
            "mode": self.mode,
            "pipeline": self.pipeline_name,
            "grading": self.grader.name,
            "scheduler": self.gate.describe() if self.gate else None,
            "cost_model": self.comp.cost_model.to_dict() if self.comp.cost_model else None,
            "workload": None
            if self.workload is None
            else {
                "name": self.workload.get("name"),
                "sha256": self.workload["sha256"],
                "params": self.workload["params"],
            },
            "n_warmup": len(self.warmup),
            "n_requests": len(self.requests),
            "purpose": self.purpose_check,
            "preflight": self.preflight,
            "comparability": self._comparability_record(),
        }
        if self.cfg.get("run", {}).get("probe_server_policy"):
            try:
                self.preflight["server_probe"] = probe_server_priority(
                    self.comp.engine, self.cfg["llm"].get("server_scheduling_policy")
                )
            except Exception as e:
                failed = {"error": str(e), "preflight": self.preflight}
                write_new(
                    self.run_dir / "preflight_failed.json",
                    json.dumps(failed, indent=2, default=str),
                )
                raise
        write_run_files(self.run_dir, self.cfg, collect_run_metadata(self.cfg, engines, extra))
        prefix_reset = None
        if self.cfg.get("run", {}).get("reset_prefix_cache"):
            fn = getattr(self.comp.engine, "reset_prefix_cache", None)
            prefix_reset = fn() if fn else {"error": "engine 미지원"}
        metrics_before = self._engine_metrics()
        started = time.perf_counter()

        t0 = time.perf_counter()
        for i, ex in enumerate(self.warmup):  # warmup 은 항상 순차
            self._execute({"i": -1 - i, "query_id": ex.id, "arrival": None}, t0, warmup=True)
        self.t0 = time.perf_counter()
        sampler = self._start_sampler()
        if self.mode == "sequential":
            for req in self.requests:
                self._execute(req, self.t0, warmup=False)
        else:
            self._run_live()
        sampled = sampler.stop() if sampler else None
        if sampler:
            writer = JsonlWriter(self.run_dir / "engine_metrics.jsonl")
            for row in sampler.samples:
                writer.write(row)

        end = {
            "wall_seconds": time.perf_counter() - started,
            "prefix_cache_reset": prefix_reset,
            "gpu_after": gpu_snapshot(),
            "engine_metrics_before": metrics_before,
            "engine_metrics_after": (metrics_after := self._engine_metrics()),
            "prefix_cache_delta": metric_delta(metrics_before, metrics_after, "prefix_cache"),
            "engine_metrics_sampled": sampled,
            "scheduler": None
            if self.gate is None
            else {"max_waiting": self.gate.max_waiting, "dispatches": len(self.gate.log)},
            "web_search": self.comp.web_search.stats()
            if hasattr(self.comp.web_search, "stats")
            else None,
        }
        write_new(self.run_dir / "run_end.json", json.dumps(end, indent=2, default=str))
        if self.gate is not None:
            log = JsonlWriter(self.run_dir / "scheduler_log.jsonl")
            for e in self.gate.log:
                log.write({**e, "t": e["t"] - self.t0})
        write_new(
            self.run_dir / "summary.json",
            json.dumps(summarize(self.rows, self.mode), indent=2, default=str),
        )
        return self.run_dir

    def _run_live(self) -> None:
        threads = []
        t0 = self.t0
        for req in sorted(self.requests, key=lambda r: r["arrival"]):
            delay = t0 + req["arrival"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            th = threading.Thread(target=self._execute, args=(req, t0, False), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
