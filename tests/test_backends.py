from types import SimpleNamespace

import pytest

from rag.llm.anthropic_llm import AnthropicLLM
from rag.retrieval.tavily_search import TavilyWebSearch


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=" Paris "),
                SimpleNamespace(type="tool_use", text="ignored"),
            ]
        )


def test_anthropic_llm_passes_config_and_joins_text():
    messages = FakeMessages()
    llm = AnthropicLLM(
        "claude-haiku-4-5",
        temperature=0.0,
        max_tokens=64,
        client=SimpleNamespace(messages=messages),
    )
    assert llm.generate("Q?") == "Paris"
    call = messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["extra_body"] == {"temperature": 0.0}
    assert call["max_tokens"] == 64
    assert call["messages"] == [{"role": "user", "content": "Q?"}]


class FakeTavily:
    def search(self, query, max_results, search_depth):
        return {
            "results": [
                {"url": f"https://e.com/{i}", "title": f"t{i}", "content": f"c{i}", "score": 0.5}
                for i in range(max_results + 2)
            ]
        }


def test_tavily_maps_results_to_documents():
    docs = TavilyWebSearch(client=FakeTavily()).search("q", top_k=2)
    assert [d.id for d in docs] == ["https://e.com/0", "https://e.com/1"]
    assert docs[0].text == "c0"
    assert docs[0].metadata["source"] == "web"


def test_faiss_retriever_roundtrip(tmp_path):
    faiss = pytest.importorskip("faiss")
    np = pytest.importorskip("numpy")
    from rag.retrieval.faiss_retriever import FaissRetriever, save_index

    vecs = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vecs)
    save_index(index, tmp_path / "x.faiss")
    (tmp_path / "docs.jsonl").write_text(
        "\n".join(f'{{"id": "d{i}", "text": "doc {i}"}}' for i in range(3)), encoding="utf-8"
    )

    class Encoder:
        def encode(self, texts, **kwargs):
            return vecs[[1]]

    r = FaissRetriever(tmp_path / "x.faiss", tmp_path / "docs.jsonl", "unused", encoder=Encoder())
    docs = r.retrieve("q", top_k=2)
    assert docs[0].id == "d1"
    assert docs[0].score == pytest.approx(1.0)
