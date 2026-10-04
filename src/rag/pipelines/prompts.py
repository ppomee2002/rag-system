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
