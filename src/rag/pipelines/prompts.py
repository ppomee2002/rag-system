GENERATE = """Answer the question using the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""

GRADE = """Is the following document relevant to the question? Reply with a single number between 0 and 1.

Question: {question}
Document: {document}
Score:"""

REWRITE = """Rewrite the question into a concise web search query.

Question: {question}
Query:"""


def format_context(docs) -> str:
    return "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))


# Batch grading (SYS-2): 문서 여러 개를 한 번의 호출로 평가.
# 출력은 JSON 으로 받고 rag.grading.strategies.parse_batch_scores 에서 검증한다.
GRADE_BATCH = (
    "For each document below, judge how relevant it is to the question.\n"
    'Reply with only a JSON object of the form {{"scores": [s1, s2, ...]}} containing exactly '
    "{n} numbers between 0 and 1, one per document, in the same order.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "{documents}\n"
    "\n"
    "JSON:"
)


def format_batch_documents(docs) -> str:
    return "\n\n".join(f"Document [{i + 1}]: {d.text}" for i, d in enumerate(docs))
