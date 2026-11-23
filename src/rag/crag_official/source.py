"""공식 CRAG 출처·버전·상수 (provenance).

공식 저장소   https://github.com/HuskyInSalt/CRAG
기준 commit   de7c2961ae624a1483a138c5798e1f6d0c4fb0e0 (2024-10-08, "Update README.md")
논문          Yan et al., "Corrective Retrieval Augmented Generation", arXiv:2401.15884
라이선스      저장소에 LICENSE 파일이 없고 GitHub API license 필드도 null (2026-11-23 확인).
              -> 별도 허가가 없으므로 공식 코드를 이 레포에 복사(vendoring)하지 않는다.
                 알고리즘은 아래 줄 번호를 근거로 다시 구현했고, 원본은
                 experiments/official_crag/fetch_official.py 가 third_party/CRAG (gitignore) 에
                 위 commit 으로 받는다. 짧은 prompt 문자열은 재현에 필수라 출처와 함께 인용한다.
Evaluator     README 의 Google Drive 폴더 `finetuned_t5_evaluator`
              (https://drive.google.com/drive/folders/1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7):
              config.json, model.safetensors (2.75GB), spiece.model, tokenizer_config.json,
              special_tokens_map.json, added_tokens.json (2023-12-23). 버전 표기 없음 -> 다운로드 후
              파일 sha256 을 기록한다 (fetch_official.py verify --hash).

이 파일의 값은 모두 공식 코드에서 그대로 옮긴 것이다. 위치는 `scripts/<file>:<line>` (기준 commit).
"""

from __future__ import annotations

OFFICIAL_REPO_URL = "https://github.com/HuskyInSalt/CRAG"
OFFICIAL_COMMIT = "de7c2961ae624a1483a138c5798e1f6d0c4fb0e0"
OFFICIAL_PAPER = "https://arxiv.org/abs/2401.15884"
OFFICIAL_LICENSE = None  # LICENSE 파일 없음 (all rights reserved 로 간주)
EVALUATOR_WEIGHTS_URL = "https://drive.google.com/drive/folders/1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7"
EVALUATOR_BASE_MODEL = (
    "t5-large"  # train_evaluator.py:55 (T5ForSequenceClassification, num_labels=1)
)

# ---------------------------------------------------------------- evaluator
# CRAG_Inference.py:165 — tokenizer(c, padding="max_length", max_length=512) (truncation 없음)
# utils.py:284 (select_relevants) — truncation=True, max_length=512
EVAL_MAX_LENGTH = 512
# CRAG_Inference.py:159-162 — "q [SEP] p" 의 p 가 비면 모델을 돌리지 않고 score -1.0
EMPTY_PASSAGE_SCORE = -1.0
# utils.py:280-281 — 단어 4개 미만 strip 은 -1.0 (refinement 전용)
SHORT_STRIP_SCORE = -1.0
SHORT_STRIP_MIN_WORDS = 4
PAIR_SEPARATOR = " [SEP] "  # data_process.py:98, utils.py:283

# run_crag_inference.sh:15 + CRAG_Inference.py:221 (`args.lower_threshold = -args.lower_threshold`)
# 공식 threshold 는 T5 회귀 출력 (logit, 학습 label = ±1) 척도. Llama 0~1 점수와 의미가 다르다.
OFFICIAL_THRESHOLDS = {
    "popqa": {"upper": 0.592, "lower": -0.995},
    "pubqa": {"upper": 0.5, "lower": -0.915},  # run_crag_inference.sh:26 (주석 처리된 예시)
}
OFFICIAL_NDOCS = 10  # run_crag_inference.sh:15 --ndocs 10

# ---------------------------------------------------------------- refinement
# internal_knowledge_preparation.py:61 — selection 이면 3, 그 외 6
INTERNAL_TOP_N = {"selection": 3, "excerption": 6, "fixed_num": 6}
EXTERNAL_TOP_N = 5  # external_knowledge_preparation.py:100
FIXED_NUM_WINDOW = 50  # internal_knowledge_preparation.py:14
EXCERPTION_CONCAT = 3  # internal_knowledge_preparation.py:29
STRIP_JOIN = "; "  # utils.py:299
DEFAULT_DECOMPOSE_MODE = "selection"  # run_knowledge_preparation.sh:10

# ---------------------------------------------------------------- generation
# CRAG_Inference.py:60 (PopQA: instruction == question 인 경우)
POPQA_PROMPT = (
    "Refer to the following documents, follow the instruction and answer the question."
    "\n\nDocuments: {paragraph}\n\nInstruction: Answer the question: {question}"
)
# CRAG_Inference.py:224 — temperature 0, top_p 1, max_tokens 100
GENERATION_PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 100}
RAG_PASSAGE_JOIN = " [sep] "  # CRAG_Inference.py:99 (method=rag 의 문단 결합)
# combined_knowledge_preparation.py:14
AMBIGUOUS_TEMPLATE = "Knowledge1: {internal} [sep] Knowledge2: {external}"
CONTROL_TOKENS = (  # CRAG_Inference.py:32-33
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]",
    "[Retrieval]",
    "[Irrelevant]",
    "[Relevant]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
)

# ---------------------------------------------------------------- external search
# utils.py:231 — PopQA 검색어 추출 few-shot prompt (원본은 gpt-3.5-turbo-16k, temperature 0.1).
# 주의: 원본 템플릿은 `{{question}}` 이라 utils.py:240 의 str.format 후에도 리터럴 `{question}` 이
# 남는다 (질문이 들어가지 않는 결함). 여기서는 의도된 동작대로 질문을 넣는다 (KNOWN_OFFICIAL_ISSUES).  # noqa: E501
POPQA_KEYWORD_PROMPT = (
    "Extract at most three keywords separated by comma from the following dialogues and "
    "questions as queries for the web search, including topic background within dialogues and "
    "main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\nquery: "
    "Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, "
    "Billy Carlson, born\n\nquestion: What is the religion of John Gwynn?\nquery: religion of "
    "John Gwynn\n\nquestion: What sport does Kiribati men's national basketball team play?\n"
    "query: sport, Kiribati men's national basketball team play\n\nquestion: {question}\nquery: "
)
WIKI_QUERY_PREFIX = "Wikipedia, "  # external_knowledge_preparation.py:20 (mode=wiki)
KEYWORD_MODEL = "gpt-3.5-turbo-16k"
SEARCH_PROVIDER = "serper.dev (Google)"

# ---------------------------------------------------------------- 공식 코드의 알려진 결함 (기록)
KNOWN_OFFICIAL_ISSUES = (
    "combined_knowledge_preparation.py: argparse import 누락, ambiguous_lines 미정의, "
    "enumerate(zip(...)) 언팩 오류 -> 그대로는 실행되지 않음. 저장소의 ref/ambiguous 파일은 제공됨",
    "external_knowledge_preparation.py:180 questions[:10] 로 앞 10문항만 처리 (디버그 잔재)",
    "external_knowledge_preparation.py:154 select_relevants(..., mode='cross_encode') 는 utils.py "
    "의 시그니처에 mode 인자가 없어 TypeError",
    "CRAG_Inference.py:inference 는 질문당 정확히 10줄을 가정 (n % 10). 빈 passage 는 n_docs "
    "검사 전에 -1.0 을 추가하므로 ndocs<10 이면 정렬이 어긋남. ndocs=10 에서만 의미가 맞다",
    "CRAG_Inference.py:246-248 internal/external 파일 줄을 strip() 만 하므로 '#' 접두가 generation "
    "입력에 남음 (ambiguous 는 [1:] 로 제거된 뒤 결합되어 '#' 없음)",
    "utils.py:231/240 keyword prompt 의 '{{question}}' 은 format 후 리터럴 '{question}' 이 되어 "
    "질문이 GPT 에 전달되지 않음 (공개된 ref/incorrect 는 질문별 내용이라 다른 버전으로 생성된 것으로 보임)",  # noqa: E501
    "eval.py match: 대소문자 구분 substring (정규화 없음). 이 레포 rag.eval.metrics.accuracy 는 "
    "소문자·구두점·관사 정규화 후 substring",
)
