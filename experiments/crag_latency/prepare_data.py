"""PopQA long-tail 질의셋 + 검색 corpus 용 Wikipedia 문서를 준비한다.

    uv run python experiments/crag_latency/prepare_data.py \
        --config experiments/crag_latency/config.yaml

1. HF `akariasai/PopQA` 에서 s_pop < popularity_threshold 인 long-tail subset 추출
   (C-RAG / Self-RAG 와 같은 기준: s_pop < 100 -> 1,399 문항)
2. 각 문항의 subject / object Wikipedia 문서 본문을 MediaWiki API 로 수집
   - subject 문서: 정답 근거 / object 문서: 자연스러운 distractor
   - 응답은 data/raw/wiki/ 에 캐시 (재실행 시 재요청 안 함)
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from rag.config import load_config

load_dotenv()  # WIKIMEDIA_CONTACT

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia 는 연락처 없는 User-Agent 를 강하게 rate limit 한다
# -> WIKIMEDIA_CONTACT 환경변수에 이메일/URL 지정 권장
USER_AGENT = f"crag-latency-experiment/0.1 ({os.getenv('WIKIMEDIA_CONTACT', 'research')})"
MAX_RETRIES = 6


def load_popqa_longtail(threshold: int) -> list[dict]:
    from datasets import load_dataset

    rows = []
    for r in load_dataset("akariasai/PopQA", split="test"):
        if r["s_pop"] >= threshold:
            continue
        rows.append(
            {
                "id": str(r["id"]),
                "question": r["question"],
                "answers": json.loads(r["possible_answers"]),
                "s_pop": r["s_pop"],
                "prop": r["prop"],
                "s_wiki_title": r["s_wiki_title"],
                "o_wiki_title": r["o_wiki_title"],
            }
        )
    return rows


def _get_json(url: str) -> dict:
    """429 / 5xx 는 Retry-After (없으면 지수 backoff) 만큼 기다린 뒤 재시도."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == MAX_RETRIES - 1:
                raise
            retry_after = e.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2**attempt
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_wiki_page(title: str, cache_dir: Path, sleep: float) -> dict | None:
    cache = cache_dir / (hashlib.sha1(title.encode("utf-8")).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
        "titles": title,
    }
    data = _get_json(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
    time.sleep(sleep)

    pages = data.get("query", {}).get("pages", [])
    page = pages[0] if pages else {}
    out = None
    if not page.get("missing") and page.get("extract"):
        out = {"title": page["title"], "text": page["extract"]}
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/crag_latency/config.yaml")
    parser.add_argument("--sleep", type=float, default=1.0, help="Wikipedia API 요청 간격 (초)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_cfg = cfg["corpus"]

    rows = load_popqa_longtail(cfg["dataset"]["popularity_threshold"])
    qa_path = Path(cfg["dataset"]["path"])
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions -> {qa_path}")

    titles = sorted({t for r in rows for t in (r["s_wiki_title"], r["o_wiki_title"]) if t})
    cache_dir = Path(corpus_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen, n_missing = set(), 0
    pages_path = Path(corpus_cfg["pages_path"])
    with open(pages_path, "w", encoding="utf-8") as f:
        for title in tqdm(titles, desc="wikipedia"):
            page = fetch_wiki_page(title, cache_dir, args.sleep)
            if page is None:
                n_missing += 1
                continue
            if page["title"] in seen:  # redirect 로 같은 문서에 도달한 경우
                continue
            seen.add(page["title"])
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
    print(f"{len(seen)} pages ({n_missing} missing) -> {pages_path}")


if __name__ == "__main__":
    main()
