import json
from pathlib import Path

from rag.types import Document


class FaissRetriever:
    """sentence-transformers 임베딩 + FAISS inner-product 인덱스 기반 dense `Retriever`.

    인덱스와 문서 파일은 사전에 만들어 두고 로드만 한다.
    docs_path 는 한 줄당 {"id", "text", ...} jsonl 이며 줄 순서가 인덱스 id 와 일치해야 한다.
    """

    def __init__(
        self,
        index_path: str | Path,
        docs_path: str | Path,
        model_name: str,
        query_prefix: str = "",
        device: str | None = None,
        encoder=None,
    ):
        self.index = load_index(index_path)
        self.docs = load_docs(docs_path)
        if self.index.ntotal != len(self.docs):
            raise ValueError(f"index size {self.index.ntotal} != docs {len(self.docs)}")
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name, device=device)
        self.encoder = encoder
        self.query_prefix = query_prefix

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        q = self.encoder.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, ids = self.index.search(q.astype("float32"), top_k)
        out = []
        for score, i in zip(scores[0], ids[0], strict=True):
            if i < 0:
                continue
            d = self.docs[i]
            out.append(Document(id=d.id, text=d.text, metadata=d.metadata, score=float(score)))
        return out


# faiss.read_index / write_index 는 Windows 에서 비 ASCII 경로를 열지 못하므로 bytes 로 직렬화한다
def save_index(index, path: str | Path) -> None:
    import faiss

    Path(path).write_bytes(faiss.serialize_index(index).tobytes())


def load_index(path: str | Path):
    import faiss
    import numpy as np

    return faiss.deserialize_index(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8))


def load_docs(path: str | Path) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(id=str(row.pop("id")), text=row.pop("text"), metadata=row))
    return docs
