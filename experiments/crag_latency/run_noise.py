"""Bad evidence 주입 실험: gold 문서를 고정하고 bad 문서의 종류 / 개수만 바꿔 답변 품질을 비교.

    uv run python experiments/crag_latency/run_noise.py --limit 203 [--dry-run]

The Power of Noise (Cuconasu et al., SIGIR 2024) 방식. query 마다 아래 context 로 각각 generation:

  gold               정답 chunk 1개 (완벽히 제거했을 때의 상한)
  gold+dist{n}       + distracting n개 (제거 전)
  gold+rand{n}       + random n개
  dist{N}            distracting 만 (gold 없음)
  filter@{t}         gold+dist{N} 중 grader score >= t 인 문서만 (실제 제거 후)

문서 종류 (정답 문자열은 chunk 본문에서만 찾는다. 제목 줄은 제외):
  gold         subject 문서의 chunk 중 정답 문자열 포함
  distracting  그 query 의 검색 상위 chunk 중 정답 근거가 아닌 것
               (정답 문자열이 없거나, 있어도 subject 를 언급하지 않음)
  random       다른 문서에서 무작위 추출한 chunk (정답 문자열 없음)

latency 는 측정하지 않는다 (품질 비교 전용).
출력: results/crag_noise/<timestamp>/ 에 results.jsonl, selection.jsonl, skipped.jsonl, report.md
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from analyze import QUALITY, paired_bootstrap, to_markdown
from dotenv import load_dotenv
from run import apply_overrides, build_llm
from tqdm import tqdm

from rag.config import load_config
from rag.data.loader import load_qa_jsonl
from rag.eval.metrics import normalize_answer, score_all
from rag.pipelines import prompts
from rag.pipelines.crag import CRAG
from rag.retrieval.faiss_retriever import FaissRetriever
from rag.types import Document, QAExample


def _contains(norm_text: str, phrase: str) -> bool:
    """정규화된 text 에 phrase 가 토큰 경계 단위로 들어 있는지 ("pol" 이 "policy" 에 안 걸리게)."""
    p = normalize_answer(phrase)
    return bool(p) and f" {p} " in f" {norm_text} "


def subject_title(ex: QAExample, cache_dir: Path) -> str:
    """s_wiki_title 을 redirect 이후의 실제 문서 제목 (= chunk title) 으로 바꾼다."""
    title = ex.metadata["s_wiki_title"]
    cache = cache_dir / (hashlib.sha1(title.encode("utf-8")).hexdigest() + ".json")
    page = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    return page["title"] if page else title


def classify(doc: Document, ex: QAExample, s_title: str) -> str:
    """gold / support (subject 문서 밖의 정답 근거, 어느 쪽에도 쓰지 않음) / distracting."""
    body = normalize_answer(doc.text.partition("\n")[2])
    if not any(_contains(body, a) for a in ex.answers):
        return "distracting"
    if doc.metadata.get("title") == s_title:
        return "gold"
    subject = re.sub(r"\s*\([^)]*\)$", "", ex.metadata["s_wiki_title"])
    return "support" if _contains(normalize_answer(doc.text), subject) else "distracting"


def select(
    ex: QAExample, retriever: FaissRetriever, by_title: dict, ncfg: dict, cache_dir: Path, seed: int
) -> dict | str:
    """query 의 gold / distracting / random 문서를 고른다. 실험에 쓸 수 없으면 사유(str)를 반환."""
    n_max = max(ncfg["n_bad"])
    s_title = subject_title(ex, cache_dir)
    retrieved = retriever.retrieve(ex.question, ncfg["candidate_k"])
    kinds = [classify(d, ex, s_title) for d in retrieved]

    gold_rank = next((i for i, k in enumerate(kinds) if k == "gold"), None)
    if gold_rank is not None:
        gold = retrieved[gold_rank]
    else:  # 검색 상위에 없으면 subject 문서 전체에서 찾는다
        gold = next((d for d in by_title[s_title] if classify(d, ex, s_title) == "gold"), None)
    if gold is None:
        return "no_gold"

    dist = [d for d, k in zip(retrieved, kinds, strict=True) if k == "distracting"][:n_max]
    if len(dist) < n_max:
        return "not_enough_distracting"

    # query 별 seed: 실행 순서 / resume 과 무관하게 같은 random 문서가 뽑힌다
    rng = random.Random(f"{seed}:{ex.id}")
    exclude = {s_title, ex.metadata.get("o_wiki_title")} | {d.id for d in retrieved}
    rand: list[Document] = []
    while len(rand) < n_max:
        d = rng.choice(retriever.docs)
        if d.metadata.get("title") in exclude or d.id in exclude:
            continue
        if classify(d, ex, s_title) != "distracting":  # 정답 문자열이 우연히 들어간 chunk
            continue
        exclude.add(d.id)
        rand.append(d)
    return {"gold": gold, "gold_rank": gold_rank, "dist": dist, "rand": rand}


def build_conditions(sel: dict, scores: list[float], ncfg: dict) -> dict[str, list[Document]]:
    """condition 이름 -> context 문서. scores 는 [gold, *dist] 순서의 grader score."""
    gold, dist, rand = sel["gold"], sel["dist"], sel["rand"]
    n_max = max(ncfg["n_bad"])

    def with_gold(bad: list[Document]) -> list[Document]:
        return [gold, *bad] if ncfg["gold_position"] == "first" else [*bad, gold]

    conds = {"gold": [gold]}
    for n in ncfg["n_bad"]:
        conds[f"gold+dist{n}"] = with_gold(dist[:n])
    for n in ncfg["n_bad"]:
        conds[f"gold+rand{n}"] = with_gold(rand[:n])
    conds[f"dist{n_max}"] = dist
    score_of = dict(zip([gold.id, *(d.id for d in dist)], scores, strict=True))
    for t in ncfg["filter_thresholds"]:
        conds[f"filter@{t}"] = [d for d in with_gold(dist) if score_of[d.id] >= t]
    return {c: docs for c, docs in conds.items() if _selected(c, ncfg)}


def _selected(cond: str, ncfg: dict) -> bool:
    """noise.conditions 에 있는 condition 만 실행 / 집계한다 ("filter" 는 모든 threshold)."""
    only = ncfg.get("conditions")
    return not only or cond in only or cond.split("@")[0] in only


def write_report(out_dir: Path, ncfg: dict) -> None:
    df = pd.read_json(out_dir / "results.jsonl", lines=True, dtype={"query_id": str})
    df = df[df["condition"].map(lambda c: _selected(c, ncfg))]
    if df.empty:
        return
    order = list(dict.fromkeys(df["condition"]))
    g = df.groupby("condition")
    summary = g[QUALITY].mean()
    summary.insert(0, "n", g.size())
    summary.insert(1, "n_contexts", g["contexts"].agg(lambda s: s.map(len).mean()))
    summary = summary.reindex(order)

    # analyze.paired_bootstrap 은 pipeline / repeat 컬럼을 기준으로 짝짓는다
    paired = df.rename(columns={"condition": "pipeline"}).assign(repeat=0)
    cis = pd.concat(
        paired_bootstrap(paired, "gold", c, ["acc", "f1"]) for c in order if c != "gold"
    )
    before = f"gold+dist{max(ncfg['n_bad'])}"
    filters = [c for c in order if c.startswith("filter@")]
    removal = (
        pd.concat(paired_bootstrap(paired, before, c, ["acc", "f1"]) for c in filters)
        if before in order and filters
        else None
    )

    sel = pd.read_json(out_dir / "selection.jsonl", lines=True, dtype={"query_id": str})
    sel = sel[sel["query_id"].isin(df["query_id"])]
    gold_s = sel["grader_scores"].map(lambda s: s[0])
    dist_s = sel["grader_scores"].map(lambda s: s[1:]).explode().astype(float)
    grader = pd.DataFrame(
        {
            "gold_kept": [(gold_s >= t).mean() for t in ncfg["filter_thresholds"]],
            "distracting_removed": [(dist_s < t).mean() for t in ncfg["filter_thresholds"]],
        },
        index=pd.Index(ncfg["filter_thresholds"], name="threshold"),
    )

    skipped = pd.read_json(out_dir / "skipped.jsonl", lines=True)
    n_max = max(ncfg["n_bad"])
    s = [
        f"# Bad evidence 주입 실험 ({out_dir.name})",
        "",
        f"query {df['query_id'].nunique()}개, gold 위치: {ncfg['gold_position']}",
        "",
        "## 1. condition 별 답변 품질",
        "",
        to_markdown(summary),
        "",
        "## 2. gold 대비 차이 (같은 query 끼리 짝지은 paired bootstrap 95% CI)",
        "",
        to_markdown(cis),
        "",
        f"### 제거 전 ({before}) 대비 grader 필터 후",
        "",
        to_markdown(removal) if removal is not None else "해당 condition 없음",
        "",
        f"## 3. Grader 가 oracle 라벨을 맞힌 비율 (gold+dist{n_max} 의 문서 기준)",
        "",
        to_markdown(grader),
        "",
        "## 4. 제외된 query",
        "",
        to_markdown(skipped["reason"].value_counts().rename_axis("reason").to_frame("n"))
        if len(skipped)
        else "없음",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(s), encoding="utf-8")
    print("\n".join(s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/crag_latency/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="dataset.limit override")
    parser.add_argument("--resume", type=Path, default=None, help="중단된 run 폴더")
    parser.add_argument(
        "--dry-run", action="store_true", help="문서 선택만 하고 LLM 호출 없이 종료"
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="config override"
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["dataset"]["limit"] = args.limit
    apply_overrides(cfg, args.set)
    ncfg, rcfg = cfg["noise"], cfg["retriever"]

    # run.py 와 같은 query 셋 (같은 shuffle, limit, warmup 제외) 을 써서 결과를 비교할 수 있게 한다
    examples = load_qa_jsonl(cfg["dataset"]["path"])
    random.Random(cfg["seed"]).shuffle(examples)
    examples = examples[: cfg["dataset"]["limit"]][cfg["run"]["warmup_queries"] :]

    retriever = FaissRetriever(
        index_path=rcfg["index_path"],
        docs_path=rcfg["docs_path"],
        model_name=cfg["embedding"]["model"],
        query_prefix=rcfg.get("query_prefix", ""),
        device=rcfg.get("device"),
    )
    by_title = defaultdict(list)
    for d in retriever.docs:
        by_title[d.metadata.get("title")].append(d)
    cache_dir = Path(cfg["corpus"]["cache_dir"])

    selections, skipped = {}, []
    for ex in tqdm(examples, desc="select"):
        sel = select(ex, retriever, by_title, ncfg, cache_dir, cfg["seed"])
        if isinstance(sel, str):
            skipped.append({"query_id": ex.id, "reason": sel})
        else:
            selections[ex.id] = sel
    reasons = pd.Series([r["reason"] for r in skipped]).value_counts().to_dict()
    in_top_k = sum(
        s["gold_rank"] is not None and s["gold_rank"] < rcfg["top_k"] for s in selections.values()
    )
    print(f"usable {len(selections)} / {len(examples)} queries, skipped: {reasons}")
    print(f"gold in retrieval top-{rcfg['top_k']}: {in_top_k} / {len(selections)}")
    if args.dry_run:
        return

    if args.resume:
        out_dir = args.resume
        rows = []
        for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        (out_dir / "results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        graded = {}
        for line in (out_dir / "selection.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                graded[row["query_id"]] = row["grader_scores"]
            except json.JSONDecodeError:
                pass
        print(f"resume {out_dir}: {len(rows)} rows already done")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["output_dir"]) / "crag_noise" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, graded = [], {}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (out_dir / "skipped.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in skipped), encoding="utf-8"
    )

    done = {(r["query_id"], r["condition"]) for r in rows}
    # 같은 query 에서 context 가 같으면 답변을 재사용한다 (예: filter 결과가 gold 만 남은 경우)
    answers = {(r["query_id"], tuple(r["contexts"])): r["answer"] for r in rows}

    llm = build_llm(cfg["llm"])
    grader_cfg = cfg["crag"]["grader"]
    grader_llm = build_llm({**cfg["llm"], **grader_cfg}) if grader_cfg.get("model") else None
    # grader 프롬프트 / score 파싱을 C-RAG 와 똑같이 쓰기 위해 CRAG.grade_all 만 빌려 쓴다
    grader = CRAG(retriever, llm, web_search=None, grader_llm=grader_llm, parallel_grading=True)

    by_id = {ex.id: ex for ex in examples}
    n_errors = 0
    with (
        open(out_dir / "results.jsonl", "a", encoding="utf-8") as f,
        open(out_dir / "selection.jsonl", "a", encoding="utf-8") as fsel,
        open(out_dir / "errors.jsonl", "a", encoding="utf-8") as ferr,
    ):
        for qid, sel in tqdm(selections.items(), desc="generate"):
            ex = by_id[qid]
            try:
                if qid not in graded:
                    graded[qid] = grader.grade_all(ex.question, [sel["gold"], *sel["dist"]])
                    sel_row = {
                        "query_id": qid,
                        "gold": sel["gold"].id,
                        "gold_rank": sel["gold_rank"],
                        "dist": [d.id for d in sel["dist"]],
                        "rand": [d.id for d in sel["rand"]],
                        "grader_scores": graded[qid],
                    }
                    fsel.write(json.dumps(sel_row, ensure_ascii=False) + "\n")
                    fsel.flush()
                for cond, docs in build_conditions(sel, graded[qid], ncfg).items():
                    if (qid, cond) in done:
                        continue
                    key = (qid, tuple(d.id for d in docs))
                    if key not in answers:
                        answers[key] = llm.generate(
                            prompts.GENERATE.format(
                                context=prompts.format_context(docs), question=ex.question
                            )
                        )
                    row = {
                        "query_id": qid,
                        "condition": cond,
                        "question": ex.question,
                        "answer": answers[key],
                        "contexts": [d.id for d in docs],
                        "gold_in_context": sel["gold"].id in key[1],
                        "gold_answers": ex.answers,
                        **score_all(answers[key], ex.answers),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:  # API 오류 등: 기록 후 계속. --resume 으로 재시도
                n_errors += 1
                ferr.write(json.dumps({"query_id": qid, "error": repr(e)}) + "\n")
                ferr.flush()
            f.flush()

    if n_errors:
        print(f"{n_errors} errors -> {out_dir / 'errors.jsonl'} (--resume 으로 재시도)")
    write_report(out_dir, ncfg)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
