"""prepare_data.py 로 모은 Wikipedia 문서를 청킹 -> 임베딩 -> FAISS 인덱스로 저장한다.

    uv run python experiments/crag_latency/build_index.py \
        --config experiments/crag_latency/config.yaml

청킹은 이 실험 전용 고정 크기 방식 (chunking 실험 코드와 분리).
chunk_size / chunk_overlap 단위는 임베딩 모델 tokenizer 의 토큰이며,
원문 문자 offset 으로 잘라 대소문자 등 원문 그대로 보존한다.
임베딩은 shard 단위로 <index_path>.ckpt/ 에 저장되어, 중단 후 재실행하면 이어서 진행한다.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.config import load_config
from rag.retrieval.faiss_retriever import save_index


def chunk_by_tokens(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    step = size - overlap
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + size]
        chunks.append(text[window[0][0] : window[-1][1]].strip())
        if start + size >= len(offsets):
            break
    return [c for c in chunks if c]


def encode_with_checkpoints(
    encoder, texts: list[str], ckpt_dir: Path, key: str, shard_size: int, batch_size: int
) -> np.ndarray:
    """shard 단위로 임베딩해 ckpt_dir 에 저장. 중단 후 재실행하면 완료된 shard 는 건너뛴다."""
    meta = ckpt_dir / "key.txt"
    if ckpt_dir.exists() and (not meta.exists() or meta.read_text() != key):
        shutil.rmtree(ckpt_dir)  # 다른 설정으로 만든 checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta.write_text(key)

    shards = []
    n_shards = (len(texts) + shard_size - 1) // shard_size
    for k in tqdm(range(n_shards), desc="embedding shards"):
        path = ckpt_dir / f"shard_{k:05d}.npy"
        if not path.exists():
            emb = encoder.encode(
                texts[k * shard_size : (k + 1) * shard_size],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, emb)
            tmp.replace(path)  # 저장 도중 중단돼도 깨진 shard 가 남지 않게
        shards.append(np.load(path))
    return np.concatenate(shards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/crag_latency/config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=1024, help="checkpoint 단위 chunk 수")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ck, rcfg = cfg["chunking"], cfg["retriever"]
    if ck["strategy"] != "fixed":
        raise ValueError(f"build_index.py 는 fixed 청킹만 지원: {ck['strategy']}")

    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(cfg["embedding"]["model"], device=rcfg.get("device"))
    # 제목 prefix 만큼 여유를 두고 모델 최대 길이를 넘지 않게 자른다
    size = min(ck["chunk_size"], encoder.max_seq_length - 32)

    docs = []
    with open(cfg["corpus"]["pages_path"], encoding="utf-8") as f:
        for line in tqdm(f, desc="chunking"):
            page = json.loads(line)
            for i, chunk in enumerate(
                chunk_by_tokens(page["text"], encoder.tokenizer, size, ck["chunk_overlap"])
            ):
                docs.append(
                    {
                        "id": f"{page['title']}#{i}",
                        "text": f"{page['title']}\n{chunk}",
                        "title": page["title"],
                    }
                )
    print(f"{len(docs)} chunks (chunk_size={size}, overlap={ck['chunk_overlap']})")

    docs_path, index_path = Path(rcfg["docs_path"]), Path(rcfg["index_path"])
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 모델 / 청킹 / chunk 내용이 같을 때만 기존 shard 를 재사용
    key = hashlib.sha1(cfg["embedding"]["model"].encode())
    for d in docs:
        key.update((d["id"] + d["text"]).encode())
    emb = encode_with_checkpoints(
        encoder,
        [d["text"] for d in docs],
        ckpt_dir=index_path.with_suffix(".ckpt"),
        key=key.hexdigest(),
        shard_size=args.shard_size,
        batch_size=args.batch_size,
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    save_index(index, index_path)
    shutil.rmtree(index_path.with_suffix(".ckpt"), ignore_errors=True)
    print(f"saved -> {index_path}, {docs_path}")


if __name__ == "__main__":
    main()
