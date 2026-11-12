"""Grader chat template 비교 진단 (probe 실행 + 분석). 코어 코드는 바꾸지 않는다.

같은 grade 입력 (기존 Full run 의 질문 x 검색 문서, prompts.GRADE) 을 렌더링만 바꿔 raw 로 보낸다.
  template hf    : HF Llama 3.1 chat template (vLLM 기본) = system 블록
                   (Cutting Knowledge Date / Today Date: 26 Jul 2024, 25 토큰) + user + assistant
  template plain : Ollama llama3.1 template (system/tools 없음) = user + assistant
BOS: vLLM 은 렌더링에 <|begin_of_text|> 를 넣고 add_special_tokens=false (chat API 와 같은 토큰열),
     Ollama 는 렌더링에 넣지 않는다 (runner 가 붙임, chat API 와 같은 토큰열).
엔진 호출: vLLM POST /v1/completions, Ollama POST /api/generate (raw=true).
temperature 0, seed 0, max_tokens 는 기존 Full config (512), Ollama num_ctx 6144.

실행 (조건 하나 = run 하나, 새 폴더 results/crag_system/diag/template_probe/<label>/<ts>/):
    uv run python experiments/crag_system/diag_template_probe.py run --engine vllm \
        --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --template plain \
        --label S-AWQ-PLAIN --preset a1-smoke \
        --workload results/crag_system/workloads/diag_grader_diff_20.json

분석 (새 폴더 results/crag_system/analysis/template_probe/<name>/<ts>/):
    uv run python experiments/crag_system/diag_template_probe.py analyze --preset a1-smoke \
        --run S-AWQ-HF=<dir> --run S-AWQ-PLAIN=<dir> --run L-Q4-PLAIN=<dir> --run L-Q4-HF=<dir> \
        --pair S-AWQ-HF:S-AWQ-PLAIN --pair L-Q4-PLAIN:L-Q4-HF \
        [--local-profile results/integrated/profile_20261102_235932/profile.jsonl]

검사 (gates): 조건 간 입력 (prompt sha) 동일, prompt_tokens 가 기존 로그와 맞는지
  (hf = 서버 로그, plain = 로컬 로그), 기존 엔진 기본 조건의 점수 재현 (vLLM+hf = 기존 Full,
  Ollama+plain = 로컬 integrated).
지표: grader 점수 분포, 질문 단위 action (decide_action) 과 조건 간 변화,
  정답 문자열 문서 보존율 ([추정] 정답 문자열 = 정규화 후 단어 경계 일치).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_grade_rerun import output_class  # noqa: E402
from analyze_regression import (  # noqa: E402
    PRESETS,
    ROOT,
    full_view,
    gold_norms,
    has_gold,
    load_chunk_texts,
    resolve,
)
from analyze_runs import load_run, md_table, read_jsonl, write_csv  # noqa: E402
from diag_crag_logs import grade_calls_for_docs  # noqa: E402
from workload import load_workload  # noqa: E402

from rag.grading.base import decide_action, parse_score  # noqa: E402
from rag.pipelines import prompts  # noqa: E402

BOS = "<|begin_of_text|>"
SYSTEM_HF = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
)
USER_HEAD = "<|start_header_id|>user<|end_header_id|>\n\n"
USER_TAIL = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
TEMPLATES = {"hf": SYSTEM_HF, "plain": ""}
ENGINES = ("vllm", "ollama")


def render(template: str, content: str, engine: str) -> str:
    """content 는 format 하지 않고 이어 붙인다 (문서 안의 중괄호 보존)."""
    body = TEMPLATES[template] + USER_HEAD + content + USER_TAIL
    return BOS + body if engine == "vllm" else body


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 엔진 (raw)
Post = Callable[[str, dict, dict], dict]


def http_post(url: str, body: dict, headers: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def http_get(url: str, headers: dict | None = None, timeout: float = 30):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # 기록용
        return {"error": repr(e)}


class VLLMRaw:
    def __init__(self, base_url: str, model: str, api_key_env: str = "VLLM_API_KEY", post=None):
        self.base_url, self.model = base_url.rstrip("/"), model
        key = os.getenv(api_key_env or "")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.post: Post = post or http_post

    def complete(self, prompt: str, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "add_special_tokens": False,  # 렌더링에 BOS 포함 (chat API 와 같은 토큰열)
            "stream": False,
        }
        r = self.post(f"{self.base_url}/completions", body, self.headers)
        ch, usage = r["choices"][0], r.get("usage") or {}
        return {
            "raw_text": ch.get("text") or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "out_tokens": usage.get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"),
        }

    def describe(self) -> dict:
        root = self.base_url.removesuffix("/v1")
        return {
            "engine": "vllm",
            "model": self.model,
            "version": http_get(f"{root}/version", self.headers),
            "models": http_get(f"{self.base_url}/models", self.headers),
        }


class OllamaRaw:
    def __init__(self, host: str, model: str, num_ctx: int = 6144, post=None):
        self.host, self.model, self.num_ctx = host.rstrip("/"), model, num_ctx
        self.post: Post = post or http_post

    def complete(self, prompt: str, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "seed": 0,
            },
        }
        r = self.post(f"{self.host}/api/generate", body, {})
        return {
            "raw_text": r.get("response") or "",
            "prompt_tokens": r.get("prompt_eval_count"),
            "out_tokens": r.get("eval_count"),
            "finish_reason": r.get("done_reason"),
        }

    def describe(self) -> dict:
        show = {}
        try:
            show = self.post(f"{self.host}/api/show", {"model": self.model}, {})
        except Exception as e:
            show = {"error": repr(e)}
        return {
            "engine": "ollama",
            "model": self.model,
            "num_ctx": self.num_ctx,
            "version": http_get(f"{self.host}/api/version"),
            "details": show.get("details"),
            "template_sha": sha(show.get("template") or "") if "template" in show else None,
            "parameters": show.get("parameters"),
        }


# ---------------------------------------------------------------- 입력
def build_inputs(full: dict, docs_path: Path, query_ids: list[str] | None) -> list[dict]:
    """기존 Full run 의 질문 x 검색 문서 -> grade prompt (검색 순서). 기존 점수·로그 토큰 포함."""
    qids = query_ids if query_ids is not None else list(full["rows"])
    missing = [q for q in qids if q not in full["rows"]]
    if missing:
        raise SystemExit(f"기존 Full run 에 없는 질문: {missing[:10]}")
    need = {d for q in qids for d in full_view(full["rows"][q])["ret_ids"]}
    texts = load_chunk_texts(docs_path, need)
    seq = (full["cfg"].get("grading") or {}).get("strategy", "full") == "full" and not (
        full["cfg"].get("grading") or {}
    ).get("parallel")
    out = []
    for q in qids:
        r = full["rows"][q]
        fv = full_view(r)
        fc = grade_calls_for_docs(full["calls_by_query"].get(q, []), fv["scores"], seq)
        for i, doc in enumerate(fv["ret_ids"]):
            if doc not in texts:
                raise SystemExit(f"chunk 본문 없음: {doc} ({docs_path})")
            content = prompts.GRADE.format(question=r["question"], document=texts[doc])
            c = fc[i] or {}
            if c.get("prompt_chars") is not None and c["prompt_chars"] != len(content):
                raise SystemExit(f"재구성한 prompt 길이가 기존 로그와 다름: {q} rank {i}")
            out.append(
                {
                    "query_id": q,
                    "rank": i,
                    "doc_id": doc,
                    "content": content,
                    "content_sha": sha(content),
                    "full_score": fv["scores"][i],
                    "full_prompt_tokens": c.get("prompt_tokens"),
                    "full_out_tokens": c.get("out_tokens"),
                }
            )
    return out


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_run(a) -> Path:
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    f_dir = resolve(a.full or preset.get("full"))
    if not f_dir:
        raise SystemExit("--preset 또는 --full 필요")
    full = load_run("full", f_dir)
    docs_path = resolve(a.docs_path or (full["cfg"].get("retriever") or {}).get("docs_path"))
    if a.workload:
        wl = load_workload(resolve(a.workload))  # sha256 검증
        qids = list(wl["query_pool"])
    elif a.all:
        qids = None
    else:
        raise SystemExit("--workload 또는 --all 필요")
    inputs = build_inputs(full, docs_path, qids)
    if a.limit_calls:
        inputs = inputs[: a.limit_calls]
    max_tokens = a.max_tokens or (full["cfg"].get("llm") or {}).get("max_tokens", 512)
    if a.engine == "vllm":
        eng = VLLMRaw(a.base_url, a.model, a.api_key_env)
    else:
        eng = OllamaRaw(a.host, a.model, a.num_ctx)
    out = Path(a.out_root) / a.label / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        "label": a.label,
        "engine": a.engine,
        "template": a.template,
        "template_prefix": TEMPLATES[a.template],
        "bos_in_prompt": a.engine == "vllm",
        "max_tokens": max_tokens,
        "full": str(f_dir),
        "docs_path": str(docs_path),
        "docs_sha256": file_sha(docs_path),
        "n_inputs": len(inputs),
        "inputs_sha": sha("\n".join(x["content_sha"] for x in inputs)),
        "argv": sys.argv,
        "started": datetime.now().isoformat(),
        "engine_info": eng.describe(),
    }
    with open(out / "meta.json", "x", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    n_err = 0
    with open(out / "calls.jsonl", "x", encoding="utf-8") as f:
        for k, x in enumerate(inputs):
            prompt = render(a.template, x["content"], a.engine)
            t0 = time.perf_counter()
            rec = {kk: v for kk, v in x.items() if kk != "content"}
            rec.update(template=a.template, prompt_sha=sha(prompt))
            try:
                res = eng.complete(prompt, max_tokens)
                text = res["raw_text"].strip()  # 기존 엔진 adapter 와 같이 strip
                rec.update(
                    res,
                    output_text=text,
                    score=parse_score(text),
                    output_class=output_class(text),
                )
            except Exception as e:  # 기록하고 계속
                n_err += 1
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            rec["wall"] = time.perf_counter() - t0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(inputs)}", flush=True)
    print(f"saved -> {out} (errors {n_err})")
    return out


# ---------------------------------------------------------------- 분석
def band(s) -> str:
    if s is None:
        return "none"
    return "0.0" if s == 0 else "(0,0.3)" if s < 0.3 else "[0.3,0.7)" if s < 0.7 else ">=0.7"


BANDS = ("0.0", "(0,0.3)", "[0.3,0.7)", ">=0.7")


def load_probe(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    calls = [c for c in read_jsonl(d / "calls.jsonl")]
    return {"dir": d, "meta": meta, "calls": {(c["query_id"], c["rank"]): c for c in calls}}


def kept_docs(scores: list, action: str, upper: float, lower: float) -> list[int]:
    if action == "incorrect":
        return []
    cut = upper if action == "correct" else lower
    return [i for i, s in enumerate(scores) if s is not None and s >= cut]


def query_view(run: dict, gold: dict, upper: float, lower: float) -> dict[str, dict]:
    by_q: dict[str, dict] = defaultdict(dict)
    for (q, i), c in run["calls"].items():
        by_q[q][i] = c
    out = {}
    for q, cs in by_q.items():
        ranks = sorted(cs)
        if ranks != list(range(len(ranks))) or any("error" in cs[i] for i in ranks):
            continue  # 일부 문서가 빠지거나 오류면 action 계산 불가
        scores = [cs[i]["score"] for i in ranks]
        act = decide_action(scores, upper, lower)
        g = [gold.get((q, i)) for i in ranks]
        kept = kept_docs(scores, act, upper, lower)
        out[q] = {
            "scores": scores,
            "action": act,
            "has_gold_doc": any(g),
            "gold_kept": any(g[i] for i in kept),
            "n_gold_docs": sum(bool(x) for x in g),
            "n_gold_high": sum(bool(g[i]) and scores[i] >= upper for i in ranks),
        }
    return out


def dist_row(label: str, run: dict, qv: dict) -> dict:
    cs = [c for c in run["calls"].values() if "error" not in c]
    sc = [c["score"] for c in cs]
    b = Counter(band(s) for s in sc)
    n = len(sc) or 1
    cls = Counter(c.get("output_class") for c in cs)
    acts = Counter(v["action"] for v in qv.values())
    gq = [v for v in qv.values() if v["has_gold_doc"]]
    gd = sum(v["n_gold_docs"] for v in qv.values())
    return {
        "run": label,
        "engine": run["meta"]["engine"],
        "model": run["meta"]["engine_info"].get("model"),
        "template": run["meta"]["template"],
        "n_calls": len(run["calls"]),
        "n_errors": len(run["calls"]) - len(cs),
        **{f"share_{k}": b.get(k, 0) / n for k in BANDS},
        "mean_score": float(np.mean(sc)) if sc else None,
        "share_numeric_only": (cls.get("numeric", 0) + cls.get("numeric_zero", 0)) / n,
        "share_parse_failure": sum(v for k, v in cls.items() if k and k.startswith("no_digit")) / n,
        "n_finish_length": sum(c.get("finish_reason") == "length" for c in cs),
        "n_queries": len(qv),
        **{f"n_{a}": acts.get(a, 0) for a in ("correct", "ambiguous", "incorrect")},
        "gold_doc_high_rate": (sum(v["n_gold_high"] for v in qv.values()) / gd) if gd else None,
        "gold_kept_query_rate": (sum(v["gold_kept"] for v in gq) / len(gq)) if gq else None,
        "n_queries_with_gold_doc": len(gq),
    }


def boot_ci(x: np.ndarray, b: int, seed: int) -> tuple:
    if len(x) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pair_rows(name_a, name_b, ra, rb, qa, qb, upper, b, seed) -> tuple[dict, list[dict]]:
    keys = sorted(set(ra["calls"]) & set(rb["calls"]))
    keys = [k for k in keys if "error" not in ra["calls"][k] and "error" not in rb["calls"][k]]
    ha = np.array([ra["calls"][k]["score"] >= upper for k in keys])
    hb = np.array([rb["calls"][k]["score"] >= upper for k in keys])
    qs = sorted(set(qa) & set(qb))
    # 질문 단위 (cluster) 차이: 고득점 문서 비율, incorrect 여부, 정답 문서 보존
    per_q_high = []
    for q in qs:
        ks = [k for k in keys if k[0] == q]
        per_q_high.append(
            np.mean([rb["calls"][k]["score"] >= upper for k in ks])
            - np.mean([ra["calls"][k]["score"] >= upper for k in ks])
        )
    d_high = np.array(per_q_high, dtype=float)
    d_inc = np.array(
        [(qb[q]["action"] == "incorrect") - (qa[q]["action"] == "incorrect") for q in qs], float
    )
    gq = [q for q in qs if qa[q]["has_gold_doc"]]
    d_gold = np.array([qb[q]["gold_kept"] - qa[q]["gold_kept"] for q in gq], dtype=float)
    trans = Counter((qa[q]["action"], qb[q]["action"]) for q in qs)
    row = {
        "pair": f"{name_b} - {name_a}",
        "n_docs": len(keys),
        "docs_same_score": sum(ra["calls"][k]["score"] == rb["calls"][k]["score"] for k in keys),
        "docs_same_text": sum(
            ra["calls"][k].get("output_text") == rb["calls"][k].get("output_text") for k in keys
        ),
        "high_a": int(ha.sum()),
        "high_b": int(hb.sum()),
        "high_only_a": int((ha & ~hb).sum()),
        "high_only_b": int((~ha & hb).sum()),
        "n_queries": len(qs),
        "action_same": sum(qa[q]["action"] == qb[q]["action"] for q in qs),
        "share_high_diff_mean_per_query": float(d_high.mean()) if len(d_high) else None,
        "share_high_diff_ci95": boot_ci(d_high, b, seed),
        "incorrect_rate_diff": float(d_inc.mean()) if len(d_inc) else None,
        "incorrect_rate_diff_ci95": boot_ci(d_inc, b, seed),
        "n_queries_with_gold_doc": len(gq),
        "gold_kept_rate_diff": float(d_gold.mean()) if len(d_gold) else None,
        "gold_kept_rate_diff_ci95": boot_ci(d_gold, b, seed),
    }
    tr = [
        {"pair": row["pair"], "from_a": x, "to_b": y, "n_queries": n}
        for (x, y), n in sorted(trans.items())
    ]
    return row, tr


def gates(runs: dict, local: dict) -> list[dict]:
    rows = []
    shas = {lbl: {k: c["content_sha"] for k, c in r["calls"].items()} for lbl, r in runs.items()}
    ref = next(iter(shas.values()))
    for lbl, r in runs.items():
        m = r["meta"]
        cs = {k: c for k, c in r["calls"].items() if "error" not in c}
        d_srv = Counter(
            c["prompt_tokens"] - c["full_prompt_tokens"]
            for c in cs.values()
            if c.get("prompt_tokens") is not None and c.get("full_prompt_tokens") is not None
        )
        loc_pt = {
            k: v["prompt_tokens"] for k, v in local.items() if v.get("prompt_tokens") is not None
        }
        d_loc = Counter(c["prompt_tokens"] - loc_pt[k] for k, c in cs.items() if k in loc_pt)
        same_full = sum(c["score"] == c["full_score"] for c in cs.values())
        same_loc = sum(c["score"] == local[k]["score"] for k, c in cs.items() if k in local)
        rows.append(
            {
                "run": lbl,
                "engine": m["engine"],
                "template": m["template"],
                "inputs_same_as_first": sum(shas[lbl].get(k) == v for k, v in ref.items()),
                "n_inputs": len(ref),
                "prompt_tokens_minus_server_log": dict(d_srv.most_common(3)),
                "expect_server": "0 (hf)" if m["template"] == "hf" else "-25 (plain)",
                "prompt_tokens_minus_local_log": dict(d_loc.most_common(3)),
                "expect_local": "0 (plain)" if m["template"] == "plain" else "+25 (hf)",
                "score_same_as_server_full": f"{same_full}/{len(cs)}",
                "score_same_as_local": f"{same_loc}/{sum(k in local for k in cs)}",
                "docs_sha256": m.get("docs_sha256", "")[:16],
            }
        )
    return rows


def read_local(path: Path | None) -> dict:
    """로컬 integrated crag_full repeat 0 -> {(qid, rank): {score, prompt_tokens}}."""
    out = {}
    if not path or not path.exists():
        return out
    for r in read_jsonl(path):
        if r.get("pipeline") != "crag_full" or r.get("repeat", 0) != 0:
            continue
        g = [c for c in r.get("calls") or [] if c.get("kind") == "grade"]
        for i, s in enumerate(r.get("grader_scores") or []):
            out[(r["query_id"], i)] = {
                "score": s,
                "prompt_tokens": g[i].get("prompt_tokens") if i < len(g) else None,
            }
    return out


def cmd_analyze(a) -> Path:
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    f_dir = resolve(a.full or preset.get("full"))
    full = load_run("full", f_dir)
    crag = full["cfg"].get("crag") or {}
    upper, lower = crag.get("upper_threshold", 0.7), crag.get("lower_threshold", 0.3)
    runs = {}
    for spec in a.run:
        lbl, _, d = spec.partition("=")
        runs[lbl] = load_probe(resolve(d))
    # 정답 문자열 문서 [추정]
    keys = {k for r in runs.values() for k in r["calls"]}
    doc_of = {k: c["doc_id"] for r in runs.values() for k, c in r["calls"].items()}
    docs_path = resolve(a.docs_path or (full["cfg"].get("retriever") or {}).get("docs_path"))
    texts = load_chunk_texts(docs_path, set(doc_of.values()))
    gold = {}
    for q, i in keys:
        g = gold_norms((full["rows"].get(q) or {}).get("gold_answers") or [])
        gold[(q, i)] = bool(has_gold(texts.get(doc_of[(q, i)]), g))
    qv = {lbl: query_view(r, gold, upper, lower) for lbl, r in runs.items()}
    # 기존 서버 Full 과 로컬 integrated 도 같은 지표로 (기준선)
    local = read_local(resolve(a.local_profile))
    dist = [dist_row(lbl, r, qv[lbl]) for lbl, r in runs.items()]
    pairs, trans = [], []
    for p in a.pair:
        x, _, y = p.partition(":")
        row, tr = pair_rows(x, y, runs[x], runs[y], qv[x], qv[y], upper, a.bootstrap, a.seed)
        pairs.append(row)
        trans += tr
    gt = gates(runs, local)
    per_doc = []
    for lbl, r in runs.items():
        for (q, i), c in sorted(r["calls"].items()):
            per_doc.append(
                {
                    "run": lbl,
                    "query_id": q,
                    "rank": i,
                    "doc_id": c["doc_id"],
                    "gold_string": gold.get((q, i)),
                    "full_score": c.get("full_score"),
                    "local_score": (local.get((q, i)) or {}).get("score"),
                    "score": c.get("score"),
                    "output_text": c.get("output_text"),
                    "out_tokens": c.get("out_tokens"),
                    "prompt_tokens": c.get("prompt_tokens"),
                    "error": c.get("error"),
                }
            )
    per_q = [
        {"run": lbl, "query_id": q, **{k: v for k, v in d.items() if k != "scores"}}
        for lbl, m in qv.items()
        for q, d in sorted(m.items())
    ]
    out = Path(a.out_root) / a.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "gates.csv", gt)
    write_csv(out / "score_actions.csv", dist)
    write_csv(out / "pairs.csv", pairs)
    write_csv(out / "action_transitions.csv", trans)
    write_csv(out / "per_doc.csv", per_doc)
    write_csv(out / "per_query.csv", per_q)
    lines = [
        f"# Grader template 비교 ({a.name})",
        "",
        f"- 기존 Full: `{f_dir}`, 임계값 upper={upper}, lower={lower}",
        *[f"- {lbl}: `{r['dir']}`" for lbl, r in runs.items()],
        "",
        "## 검사 (gates)",
        "입력 동일 (inputs_same_as_first = n_inputs), prompt_tokens 가 기대값과 같은지, "
        "엔진 기본 조건의 점수 재현 (vLLM+hf -> 서버 Full, Ollama+plain -> 로컬).",
        "",
        md_table(gt),
        "## 점수 분포, action, 정답 문자열 문서 [추정]",
        "gold_doc_high_rate = 정답 문자열 문서 중 점수 >= upper 비율. "
        "gold_kept_query_rate = 정답 문서가 있는 질문 중 판정 후 context 에 정답 문서가 남는 비율.",
        "",
        md_table(dist),
        "## 조건 쌍 (b - a, 질문 단위 bootstrap 95% CI)",
        md_table(pairs),
        md_table(trans),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=ENGINES, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--template", choices=sorted(TEMPLATES), required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--preset", choices=sorted(PRESETS))
    r.add_argument("--full")
    r.add_argument("--docs-path")
    r.add_argument("--workload")
    r.add_argument("--all", action="store_true", help="기존 Full 의 전체 질문 (200 x 5)")
    r.add_argument("--limit-calls", type=int, default=None, help="smoke 용")
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--api-key-env", default="VLLM_API_KEY")
    r.add_argument("--host", default="http://127.0.0.1:11434")
    r.add_argument("--num-ctx", type=int, default=6144)
    r.add_argument(
        "--out-root", default=str(ROOT / "results" / "crag_system" / "diag" / "template_probe")
    )
    z = sub.add_parser("analyze")
    z.add_argument("--run", action="append", required=True, help="label=probe_dir")
    z.add_argument("--pair", action="append", default=[], help="a:b (b - a)")
    z.add_argument("--preset", choices=sorted(PRESETS))
    z.add_argument("--full")
    z.add_argument("--docs-path")
    z.add_argument("--local-profile")
    z.add_argument("--bootstrap", type=int, default=10000)
    z.add_argument("--seed", type=int, default=0)
    z.add_argument("--name", default="template")
    z.add_argument(
        "--out-root",
        default=str(ROOT / "results" / "crag_system" / "analysis" / "template_probe"),
    )
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    main()
