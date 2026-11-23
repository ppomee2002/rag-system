from rag.types import Document


def tavily_params(search_depth: str = "basic", include_raw_content: bool = False) -> dict:
    """캐시 기록·호환성 비교에 쓰는 Tavily 검색 parameter (top_k 는 기록 행에 따로)."""
    return {"search_depth": search_depth, "include_raw_content": bool(include_raw_content)}


class TavilyWebSearch:
    """Tavily 검색 API 기반 `WebSearch` 구현체. TAVILY_API_KEY 환경변수를 사용한다.

    search_raw 는 (문서, 원본 응답) 을 돌려준다 (RecordingWebSearch 가 원문 기록에 사용).
    include_raw_content=True 면 Tavily 가 페이지 원문 (raw_content) 도 돌려주며 원본 응답에만
    남는다 (Document.text 는 기존처럼 content 스니펫 -> pipeline 입력은 바뀌지 않음).
    """

    def __init__(
        self,
        api_key: str | None = None,
        search_depth: str = "basic",
        client=None,
        include_raw_content: bool = False,
    ):
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
        self.client = client
        self.search_depth = search_depth
        self.include_raw_content = include_raw_content

    def params(self) -> dict:
        """기록용 검색 parameter (API key 는 포함하지 않음)."""
        return tavily_params(self.search_depth, self.include_raw_content)

    def search_raw(self, query: str, top_k: int) -> tuple[list[Document], dict]:
        kw = {"max_results": top_k, "search_depth": self.search_depth}
        if self.include_raw_content:  # 기존 호출 인자는 그대로 (기본 False 면 추가하지 않음)
            kw["include_raw_content"] = True
        res = self.client.search(query, **kw)
        docs = [
            Document(
                id=r["url"],
                text=r.get("content", ""),
                metadata={"title": r.get("title", ""), "url": r["url"], "source": "web"},
                score=r.get("score"),
            )
            for r in res.get("results", [])[:top_k]
        ]
        return docs, res

    def search(self, query: str, top_k: int) -> list[Document]:
        return self.search_raw(query, top_k)[0]
