"""호출 단위 스케줄링 정책 (live gate 용).

규칙은 팀원 simulator 와 같다 (exp/scheduling-optimization:experiments/scheduling_optimization/
simulator.py `fill`):
  - 기본 우선순위 (base) 는 호출이 대기열에 들어갈 때 한 번 계산하고 바꾸지 않는다.
  - Aging       : key = base - alpha x (누적 대기 + 이번 대기). dispatch 할 때마다 다시 계산.
  - 대기 임계값 : 누적 대기 + 이번 대기 > T 인 호출이 있으면 그중 가장 오래 기다린 것,
                  없으면 base 순. (0, -대기, base, 순번) / (1, base, 0, 순번)
  - 실행 중인 호출은 중단하지 않는다 (호출 사이에서만 순서를 정함).

정적 SJF vs 상태 갱신형 SRPT-est
  sjf_pred  : 요청의 첫 LLM 호출 때 (= 검색 직후) query 전체 작업량을 예측하고,
              그 요청의 모든 호출에 같은 값을 쓴다.
              grading 결과·correction 여부가 우선순위에 반영되지 않는다.
  srpt_est  : 호출마다 "현재 stage 에서 남은 작업량" 을 다시 추정한다.
              grading 후 correction 경로면 rewrite + correction 후 generation,
              아니면 generation 비용만 남는다.
  srpt_nopath : (ablation) srpt_est 에서 grading 결과 (실행 경로) 정보만 뺀 것.
              grade / generate 호출은 srpt_est 와 같고, rewrite 호출 (= correction 경로가 확정된 뒤
              첫 호출) 만 경로와 무관한 기대값 pred_post(feats) 를 쓴다.
              sjf_pred -> srpt_nopath : 진행 (남은 grade 수, 현재 호출 prompt 길이) 정보의 효과
              srpt_nopath -> srpt_est : grading 결과 (correction 경로) 정보의 효과

정책은 `CallTicket.query` (= `PolicyView`) 로 정책 입력 필드만 읽는다 (rag.scheduling.state).
cost model 은 `model.for_query(query_id)` 로 고른다 (OOF 모델이면 자기 fold 를 뺀 모델).

모든 정책은 4-tuple key 를 돌려주며 작을수록 먼저 나간다: (tier, primary, secondary, seq).

Correction-aware 정책은 SRPT-est 와의 차이를 먼저 정리한 뒤 구현한다
(experiments/crag_system/README.md "Correction-aware").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.scheduling.cost_model import CostModel, OOFCostModel, features
from rag.scheduling.state import PolicyView, QueryState

POLICY_NAMES = ("fcfs", "fcfs_query", "sjf_pred", "srpt_nopath", "srpt_est")


@dataclass
class CallTicket:
    state: QueryState  # gate 만 쓰기에 사용. 정책은 .query (읽기 전용 view) 만 사용
    kind: str
    idx: int
    prompt_chars: int
    t_enqueue: float  # perf_counter
    seq: int
    base: float = 0.0
    t_grant: float | None = None
    key_at_grant: tuple | None = None
    engine_priority: int | None = None
    granted: bool = field(default=False, repr=False)
    _view: PolicyView | None = field(default=None, repr=False)

    @property
    def query(self) -> PolicyView:
        if self._view is None:
            self._view = self.state.view()
        return self._view

    def waited(self, now: float) -> float:
        """이 query 의 누적 대기 (이미 끝난 호출의 대기 + 이번 호출의 현재 대기)."""
        return self.query.cum_wait + (now - self.t_enqueue)


class Policy:
    name = "base"
    dynamic = False  # True 면 dispatch 때마다 key 재계산이 필요
    base_is_estimate = False  # True 면 base 가 작업량 추정치 (QueryState.est_remaining 에 기록)

    def base(self, t: CallTicket) -> float:
        raise NotImplementedError

    def key(self, t: CallTicket, now: float) -> tuple:
        return (1, t.base, 0.0, t.seq)

    def describe(self) -> dict:
        return {"name": self.name}


class FCFS(Policy):
    """호출 도착 순서 (vLLM 기본과 같은 순서를 client 측에서)."""

    name = "fcfs"

    def base(self, t):
        return float(t.seq)


class FCFSQuery(Policy):
    """query 도착 순서 (먼저 온 query 의 호출 우선)."""

    name = "fcfs_query"

    def base(self, t):
        return t.query.arrival


def _feats(q: PolicyView):
    return features(q.retrieval_scores, q.retrieval_chars, q.s_pop)


def _model_meta(model) -> dict:
    return {k: v for k, v in model.meta.items() if k != "train_query_ids"}


class SJFPred(Policy):
    """정적 SJF (원본 sjf_pred): 요청의 첫 호출 때 예측한 전체 작업량을 끝까지 고정."""

    name = "sjf_pred"
    base_is_estimate = True

    def __init__(self, model: CostModel | OOFCostModel):
        self.model = model
        self._frozen: dict[str, float] = {}  # request_key -> 예측값 (gate lock 안에서만 접근)

    def base(self, t):
        q = t.query
        if q.request_key not in self._frozen:
            m = self.model.for_query(q.query_id)
            self._frozen[q.request_key] = m.pred_total(_feats(q))
        return self._frozen[q.request_key]

    def describe(self):
        return {"name": self.name, "cost_model": _model_meta(self.model)}


class SRPTEst(Policy):
    """상태 갱신형 SRPT-est (원본 srpt_est): 호출마다 현재 stage 기준 남은 작업량을 다시 추정."""

    name = "srpt_est"
    base_is_estimate = True

    def __init__(self, model: CostModel | OOFCostModel):
        self.model = model

    def base(self, t):
        q = t.query
        m = self.model.for_query(q.query_id)
        return m.est_remaining(t.kind, t.idx, t.prompt_chars, q.n_docs, _feats(q))

    def describe(self):
        return {"name": self.name, "cost_model": _model_meta(self.model)}


class SRPTNoPath(SRPTEst):
    """Ablation: srpt_est 와 같되 rewrite 호출의 추정치에서 correction 경로 정보를 뺀다.

    srpt_est 에서 grading 결과가 추정치에 들어가는 곳은 rewrite 호출 하나뿐이다
    (rewrite_mean + gen_corrected_mean). 여기서는 그 값을 grading 전과 같은 기대값
    pred_post(feats) (경로를 모르는 grading 이후 작업량 예측) 로 바꾼다.
    generate 호출은 마지막 호출이라 경로와 관계없이 남은 작업이 generation 하나이므로 그대로 둔다
    (prompt 길이는 호출 시점에 관측되는 값. correction 경로면 web 문서만큼 길어질 수 있음).
    """

    name = "srpt_nopath"

    def base(self, t):
        if t.kind == "rewrite":
            q = t.query
            return self.model.for_query(q.query_id).pred_post(_feats(q))
        return super().base(t)


class _Wrapper(Policy):
    dynamic = True

    def __init__(self, inner: Policy):
        self.inner = inner
        self.base_is_estimate = inner.base_is_estimate

    def base(self, t):
        return self.inner.base(t)


class Aging(_Wrapper):
    def __init__(self, inner: Policy, alpha: float):
        if alpha < 0:
            raise ValueError("alpha >= 0")
        super().__init__(inner)
        self.alpha = alpha
        self.name = f"{inner.name}+aging"

    def key(self, t, now):
        return (1, t.base - self.alpha * t.waited(now), 0.0, t.seq)

    def describe(self):
        return {"name": self.name, "inner": self.inner.describe(), "alpha": self.alpha}


class WaitThreshold(_Wrapper):
    def __init__(self, inner: Policy, threshold: float):
        if threshold < 0:
            raise ValueError("threshold >= 0")
        super().__init__(inner)
        self.threshold = threshold
        self.name = f"{inner.name}+wait{threshold:g}"

    def key(self, t, now):
        waited = t.waited(now)
        if waited > self.threshold:
            return (0, -waited, t.base, t.seq)
        return (1, t.base, 0.0, t.seq)

    def describe(self):
        return {"name": self.name, "inner": self.inner.describe(), "threshold": self.threshold}


def build_policy(cfg: dict, cost_model: CostModel | OOFCostModel | None = None) -> Policy:
    """config `scheduler:` -> 정책.

    policy: fcfs | fcfs_query | sjf_pred | srpt_nopath | srpt_est
    aging_alpha: 0 이 아니면 Aging 적용
    wait_threshold: 지정하면 대기 임계값 적용 (Aging 과 동시 사용 불가, 원본과 같음)
    """
    name = cfg.get("policy", "fcfs")
    if name == "fcfs":
        policy: Policy = FCFS()
    elif name == "fcfs_query":
        policy = FCFSQuery()
    elif name in ("sjf_pred", "srpt_nopath", "srpt_est"):
        if cost_model is None:
            raise ValueError(f"{name} 는 cost_model 이 필요 (scheduler.cost_model)")
        cls = {"sjf_pred": SJFPred, "srpt_nopath": SRPTNoPath, "srpt_est": SRPTEst}[name]
        policy = cls(cost_model)
    elif name == "correction_aware":
        raise NotImplementedError(
            "correction_aware 는 정의 확정 후 구현 예정 (experiments/crag_system/README.md)"
        )
    else:
        raise ValueError(f"지원하지 않는 scheduling policy: {name} ({POLICY_NAMES})")
    alpha = cfg.get("aging_alpha") or 0.0
    threshold = cfg.get("wait_threshold")
    if alpha and threshold is not None:
        raise ValueError("Aging 과 wait_threshold 는 함께 쓰지 않는다 (원본 정의)")
    if alpha:
        policy = Aging(policy, alpha)
    if threshold is not None:
        policy = WaitThreshold(policy, threshold)
    return policy
