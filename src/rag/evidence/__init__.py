"""C1/C2/C3 공통 연구 인터페이스 (MODE B: Controlled Server Comparison).

graders   문서 단위 grader (공식 T5 / 기존 Llama) 공통 인터페이스
state     C2: 문서별 grading event / snapshot
budget    C3: stop_rule (평가 중단) 과 retention (문서 보존) 분리
policies  C1: evidence 정책 정의 (official, crag_rule, ablation, keep_all, vanilla)
frozen    고정 평가 입력 (질문·정답·Top-k)
pipeline  ControlledCRAG
schema    결과 필드
"""
