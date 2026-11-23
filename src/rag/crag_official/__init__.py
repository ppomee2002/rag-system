"""공식 CRAG (HuskyInSalt/CRAG) 의 Evaluator / Knowledge Refinement / Generation 재구현.

출처·버전·라이선스는 `rag.crag_official.source`. 공식 코드를 복사하지 않고 줄 번호 근거로 다시
구현했다.
"""

from rag.crag_official.evaluator import (
    CachedScoreBackend,
    EvaluatorVerdict,
    MockScoreBackend,
    OfficialRetrievalEvaluator,
    PairScore,
    official_action,
    official_doc_flag,
    process_flag,
)
from rag.crag_official.refinement import (
    RefinementResult,
    ambiguous_knowledge,
    external_knowledge,
    extract_strips,
    internal_knowledge,
)

__all__ = [
    "CachedScoreBackend",
    "EvaluatorVerdict",
    "MockScoreBackend",
    "OfficialRetrievalEvaluator",
    "PairScore",
    "RefinementResult",
    "ambiguous_knowledge",
    "external_knowledge",
    "extract_strips",
    "internal_knowledge",
    "official_action",
    "official_doc_flag",
    "process_flag",
]
