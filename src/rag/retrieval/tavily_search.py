from rag.types import Document


class TavilyWebSearch:
    """Tavily 검색 API 기반 `WebSearch` 구현체. TAVILY_API_KEY 환경변수를 사용한다."""

    def __init__(self, api_key: str | None = None, search_depth: str = "basic", client=None):
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
        self.client = client
        self.search_depth = search_depth

    def search(self, query: str, top_k: int) -> list[Document]:
        res = self.client.search(query, max_results=top_k, search_depth=self.search_depth)
        return [
            Document(
                id=r["url"],
                text=r.get("content", ""),
                metadata={"title": r.get("title", ""), "url": r["url"], "source": "web"},
                score=r.get("score"),
            )
            for r in res.get("results", [])[:top_k]
        ]
