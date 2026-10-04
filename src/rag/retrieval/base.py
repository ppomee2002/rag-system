from typing import Protocol

from rag.types import Document


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Document]: ...


class WebSearch(Protocol):
    """C-RAG 의 corrective 단계에서 사용하는 외부 검색."""

    def search(self, query: str, top_k: int) -> list[Document]: ...
