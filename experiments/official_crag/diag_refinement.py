"""MODE A Refinement 불일치 진단 (verify-refinement exact_match 0/10 원인 분리).

배경: mode_a.py verify-refinement 는 공식 internal_knowledge_preparation.py (selection, 상위 3,
'; ' 결합) 를 다시 계산해 저자 ref/correct 와 비교한다. exact_match 가 0/10 이었고, ref/correct 는
문단 순서·개수·결합 방식이 다르다. 이 스크립트는 그 원인을 아래 다섯 가지로 나눈다.

  score      같은 입력에서 공식 select_relevants 와 재구현의 T5 점수가 다르다
  selection  상위 3 집합이 다르다 (재구현 vs 공식 코드, 또는 공식 코드 vs ref/correct)
  ordering   같은 문단 집합인데 순서가 다르다 (점수순 vs 검색 순서)
  format     같은 문단·같은 순서인데 문자열 (결합자 '; ' vs ' ', 공백) 이 다르다
  source     ref/correct 의 문단이 입력 (test_popqa.txt 의 10문서) 에 없다

subcommand
  provenance  (가중치 불필요, 로컬 가능) 공식 저장소 git 이력과 ref/correct 전체 1,399줄의
              구성 규칙.
              각 줄을 '검색 순서를 유지한 문서 부분집합의 공백 결합' 으로 정확히 다시 만들 수 있는지
              (공백 정규화 후) 확인하고 문서 수 분포, top-3/'; ' 형식 여부를 센다.
  compare     (서버, T5 가중치 필요) 같은 질문·문서·가중치로
                A. 공식 utils.select_relevants (third_party/CRAG 원본 함수를 그대로 import)
                B. 재구현 rag.crag_official.refinement.internal_knowledge
              를 실행해 strip 점수, ranking, Top-3 index, 최종 문자열을 비교하고, ref/correct 와의
              차이를 위 다섯 가지로 분류한다. 문서 점수 (판정용, truncation 없음) 로 ref/correct 의
              문서 선택 규칙 가설 (top-3 / top-k / 절대 threshold) 도 검사한다.
  rules       (가중치 불필요) 기존 mode_a.py score 결과 (scores.jsonl, 문서 점수) 로
              1,399문항 전체에서 ref/correct 선택 규칙 가설을 검사한다.

결과: results/official_crag/mode_a/diag_refinement/<run_id>/ (새 폴더, 기존 결과 덮어쓰지 않음)

    uv run python experiments/official_crag/diag_refinement.py provenance
    uv run python experiments/official_crag/diag_refinement.py compare --device cuda:0 --limit 10 \
        --weights models/crag_evaluator/finetuned_t5_evaluator
    uv run python experiments/official_crag/diag_refinement.py rules \
        --scores results/official_crag/mode_a/score/<run_id>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.crag_official import source  # noqa: E402
from rag.crag_official.official_data import (  # noqa: E402
    load_official_queries,
    load_ref,
    retrieved_psgs_line,
)
from rag.crag_official.refinement import (  # noqa: E402
    internal_knowledge,
    official_passages_from_line,
)
from rag.telemetry.run_meta import (  # noqa: E402
    JsonlWriter,
    collect_run_metadata,
    create_run_dir,
    write_new,
    write_run_files,
)

DEFAULT_REPO = ROOT / "third_party" / "CRAG"
SCORE_TOL = 1e-4  # 점수 차이 판정 (fp32 같은 하드웨어면 0, 배치·커널 차이로 1e-6 수준)
CATEGORIES = ("score", "selection", "ordering", "format", "source")


# ---------------------------------------------------------------- 순수 함수 (테스트 대상)
def ws(text: str) -> str:
    """공백 정규화 (연속 공백 -> 한 칸, 앞뒤 제거)."""
    return " ".join(text.split())


def strip_hash(line: str) -> str:
    return line[1:] if line.startswith("#") else line


def ordered_subset(passages: list[str], ref: str) -> list[int] | None:
    """ref == ' '.join(passages[i] for i in S) (공백 정규화) 인 검색 순서 부분집합 S. 없으면 None.

    빈 문서는 건너뛴다. 앞에서부터 prefix 로 맞춰 가는 DFS (문서 10개라 충분히 빠르다).
    """
    target = ws(ref)
    norm = [ws(p) for p in passages]

    def dfs(start: int, acc: str, chosen: list[int]) -> list[int] | None:
        if chosen and acc == target:
            return chosen
        for i in range(start, len(norm)):
            if not norm[i]:
                continue
            nxt = norm[i] if not chosen else f"{acc} {norm[i]}"
            if target == nxt or target.startswith(nxt + " "):
                got = dfs(i + 1, nxt, chosen + [i])
                if got is not None:
                    return got
        return None

    return dfs(0, "", [])


def semicolon_parts(passages: list[str], ref: str) -> list[int] | None:
    """ref 가 공식 select_relevants 형식 ('; ' 결합, 문서 전체) 이면 각 조각의 문서 index."""
    norm = [ws(p) for p in passages]
    out = []
    for part in ws(ref).split("; "):
        hits = [i for i, p in enumerate(norm) if p and p == part]
        if not hits:
            return None
        out.append(hits[0])
    return out


def classify(
    *,
    ours_text: str,
    official_text: str,
    ours_top: list[int],
    official_top: list[int],
    score_max_abs_diff: float | None,
    ref_subset: list[int] | None,
    ref_text: str,
    tol: float = SCORE_TOL,
) -> dict:
    """불일치 유형. 'impl' = 재구현 vs 공식 코드, 'ref' = 공식 코드 vs 저자 ref/correct."""
    impl: list[str] = []
    if score_max_abs_diff is not None and score_max_abs_diff > tol:
        impl.append("score")
    if set(ours_top) != set(official_top):
        impl.append("selection")
    elif ours_top != official_top:
        impl.append("ordering")
    elif ours_text != official_text:
        impl.append("format")

    ref: list[str] = []
    if ref_subset is None:
        ref.append("source")
    else:
        if set(ref_subset) != set(official_top):
            ref.append("selection")
        elif ref_subset != official_top:
            ref.append("ordering")
        elif official_text.strip() != ref_text.strip():
            # 같은 문단·같은 순서인데 문자열만 다르면 format (결합자 '; ' vs ' ', 공백)
            ref.append("format")
    return {
        "impl": impl,
        "ref": ref,
        "impl_exact": official_text == ours_text,
        "ref_exact": official_text.strip() == ref_text.strip(),
    }


def rule_check(rows: list[dict]) -> dict:
    """rows: {scores: [10 문서 점수], kept: [ref 문서 index], nonempty: [...]}.

    가설
      top3_set        kept == 점수 상위 3 (문서 단위)
      topk_set        kept == 점수 상위 |kept| (k 가변)
      thr_upper/lower/zero  kept == {score >= t}
      global_threshold  모든 문항에서 min(kept 점수) > max(버린 비어있지 않은 문서 점수) 를
                        동시에 만족하는 t 가 있는지 (있으면 구간)
    """
    th = source.OFFICIAL_THRESHOLDS["popqa"]
    c: Counter = Counter()
    lo, hi = float("-inf"), float("inf")  # t 는 (lo, hi] 안에 있어야 한다
    sep_violations = 0
    for r in rows:
        s, kept, ne = r["scores"], r["kept"], r["nonempty"]
        rank = sorted(ne, key=lambda i: s[i], reverse=True)
        c["top3_set"] += set(rank[:3]) == set(kept)
        c["topk_set"] += set(rank[: len(kept)]) == set(kept)
        for name, t in (("thr_upper", th["upper"]), ("thr_lower", th["lower"]), ("thr_zero", 0.0)):
            c[name] += {i for i in ne if s[i] >= t} == set(kept)
        dropped = [i for i in ne if i not in kept]
        kmin = min(s[i] for i in kept)
        dmax = max((s[i] for i in dropped), default=float("-inf"))
        c["separable"] += kmin > dmax
        if kmin > dmax:
            lo, hi = max(lo, dmax), min(hi, kmin)
        else:
            sep_violations += 1
    n = len(rows)
    return {
        "n": n,
        "match_counts": dict(c),
        "match_rates": {k: v / n for k, v in c.items()} if n else {},
        "global_threshold": {
            "exists": sep_violations == 0 and lo < hi,
            "interval_open_closed": [lo, hi] if lo < hi else None,
            "n_questions_not_separable": sep_violations,
        },
    }


# ---------------------------------------------------------------- 공식 코드 import
def _stub(name: str, attrs: dict) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:  # noqa: BLE001 - 설치 안 됨 -> 빈 모듈 (refinement 경로는 쓰지 않음)
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod


def load_official_utils(repo: Path):
    """third_party/CRAG/scripts/utils.py 를 원본 그대로 import.

    openai, jsonlines 는 설치돼 있지 않으면 빈 모듈 (refinement 경로에서 쓰지 않음).
    """
    for name, attrs in {"openai": {}, "jsonlines": {}}.items():
        _stub(name, attrs)
    path = repo / "scripts" / "utils.py"
    spec = importlib.util.spec_from_file_location("crag_official_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class RecordingModel:
    """공식 select_relevants 가 부르는 model 을 감싸 logit 을 기록 (계산은 그대로)."""

    def __init__(self, model):
        self.model = model
        self.logits: list[float] = []

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        self.logits.append(float(out["logits"].float().cpu().reshape(-1)[0]))
        return out


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _paths(repo: Path) -> dict[str, Path]:
    d = repo / "data" / "popqa"
    return {
        "test": d / "test_popqa.txt",
        "sources": d / "sources",
        "correct": d / "ref" / "correct",
        "ambiguous": d / "ref" / "ambiguous",
    }


def _run_dir(run_id: str | None) -> Path:
    return create_run_dir("official_crag/mode_a/diag_refinement", "results", run_id)


# ---------------------------------------------------------------- provenance
def cmd_provenance(a) -> Path:
    repo = Path(a.repo)
    p = _paths(repo)
    queries = load_official_queries(p["test"], source.OFFICIAL_NDOCS, p["sources"])
    ref_raw = load_ref(p["correct"])
    amb = load_ref(p["ambiguous"])
    run_dir = _run_dir(a.run_id)
    hist = {
        rel: _git(repo, "log", "--format=%h %ad %s", "--date=iso", "--", rel).splitlines()
        for rel in (
            "data/popqa/ref/correct",
            "data/popqa/ref/ambiguous",
            "data/popqa/test_popqa.txt",
            "scripts/internal_knowledge_preparation.py",
            "scripts/utils.py",
            "run_knowledge_preparation.sh",
        )
    }
    retrieved_published = (repo / "data" / "popqa" / "retrieved_psgs").exists()
    cfg = {"experiment": "official_crag", "step": "diag-refinement/provenance"}
    write_run_files(
        run_dir,
        cfg,
        collect_run_metadata(cfg, None, {"repo_head": _git(repo, "rev-parse", "HEAD")}),
    )
    out = JsonlWriter(run_dir / "ref_structure.jsonl")
    c: Counter = Counter()
    n_kept: Counter = Counter()
    for q, r in zip(queries, ref_raw, strict=True):
        body = strip_hash(r)
        kept = ordered_subset(q.passages, body)
        semi = semicolon_parts(q.passages, body)
        ne = [i for i, x in enumerate(q.passages) if x.strip()]
        c["hash_prefix"] += r.startswith("#")
        if kept is None:
            c["not_ordered_subset"] += 1
        else:
            c["ordered_subset_space_join"] += 1
            n_kept[len(kept)] += 1
            c["kept_eq_all_nonempty"] += kept == ne
            c["kept_len_3"] += len(kept) == 3
        c["semicolon_format_whole_docs"] += semi is not None and len(semi) > 1
        out.write(
            {
                "index": q.index,
                "kept": kept,
                "n_nonempty": len(ne),
                "semicolon_parts": semi,
                "ref_chars": len(body),
            }
        )
    amb_eq = sum(
        x.startswith(f"Knowledge1: {strip_hash(r)} [sep] Knowledge2:")
        for x, r in zip(amb, ref_raw, strict=True)
    )
    summary = {
        "n": len(queries),
        "counts": dict(c),
        "n_kept_distribution": dict(sorted(n_kept.items())),
        "ambiguous_knowledge1_eq_correct": amb_eq,
        "popqa_retrieved_psgs_published": retrieved_published,
        "git_history": hist,
        "interpretation": (
            "ref/correct 가 모두 '검색 순서 부분집합의 공백 결합' 이고 문서 수가 가변이면, "
            "공개 코드 (selection: 점수순 상위 3, '; ' 결합) 로 만든 파일이 아니다. git 이력에서 "
            "ref/correct 가 refinement 코드보다 먼저 올라왔으면 공개 이전 버전으로 생성된 "
            "사전 생성 파일이다."
        ),
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: summary[k] for k in summary if k != "git_history"}, indent=2, ensure_ascii=False
        )
    )
    return run_dir


# ---------------------------------------------------------------- compare (T5 가중치)
def cmd_compare(a) -> Path:
    import torch
    from transformers import T5ForSequenceClassification, T5Tokenizer

    from rag.crag_official.evaluator import (
        OfficialRetrievalEvaluator,
        T5EvaluatorBackend,
        weights_fingerprint,
    )

    repo = Path(a.repo)
    p = _paths(repo)
    queries = load_official_queries(p["test"], source.OFFICIAL_NDOCS, p["sources"])
    ref_raw = load_ref(p["correct"])
    n = min(a.limit or len(queries), len(queries))
    weights = a.weights
    if not weights:
        raise SystemExit("--weights 필요 (공식 finetuned_t5_evaluator 폴더)")

    utils = load_official_utils(repo)
    # A. 공식 경로: internal_knowledge_preparation.knowledge_refinement 와 같은 로딩
    off_tok = T5Tokenizer.from_pretrained(weights)
    off_model = RecordingModel(
        T5ForSequenceClassification.from_pretrained(weights, output_hidden_states=True)
    )
    # B. 재구현 (batch_size 1 = 공식과 같은 연산 순서; --batch-size 로 batch 영향도 볼 수 있다)
    backend = T5EvaluatorBackend(weights, device=a.device, batch_size=a.batch_size)
    ev = OfficialRetrievalEvaluator(backend)

    run_dir = _run_dir(a.run_id)
    cfg = {
        "experiment": "official_crag",
        "step": "diag-refinement/compare",
        "limit": n,
        "device": a.device,
        "batch_size": a.batch_size,
        "decompose_mode": "selection",
        "score_tol": SCORE_TOL,
    }
    write_run_files(
        run_dir,
        cfg,
        collect_run_metadata(
            cfg,
            None,
            {
                "experiment_mode": "A_official_reproduction",
                "repo_head": _git(repo, "rev-parse", "HEAD"),
                "weights": weights_fingerprint(weights),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "official_utils": str(repo / "scripts" / "utils.py"),
                "input_note": (
                    "popqa retrieved_psgs 는 공식 저장소에 없음 -> test_popqa.txt 의 10문서로 "
                    "data_process.py:97 형식 (' [sep] ' 결합) 을 재구성해 양쪽에 같은 입력으로 사용"
                ),
            },
        ),
    )
    out = JsonlWriter(run_dir / "per_question.jsonl")
    top_n = source.INTERNAL_TOP_N["selection"]
    agg: Counter = Counter()
    rule_rows_doc, rule_rows_strip = [], []
    max_diff_all = 0.0
    for q, r in zip(queries[:n], ref_raw[:n], strict=True):
        psgs = official_passages_from_line(retrieved_psgs_line(q.passages))
        strips = list(psgs)  # selection: 문서 그대로 (공식 extract_strips_from_psg)
        off_model.logits = []
        off_text, off_idx = utils.select_relevants(
            strips=strips,
            query=q.question,
            tokenizer=off_tok,
            model=off_model,
            device=a.device,
            top_n=top_n,
        )
        off_text = off_text.replace("\n", " ")
        off_rank = [int(x) for x in off_idx.split(", ")] if off_idx else []
        # 공식 점수 (4단어 미만 / 예외는 모델을 안 부르거나 -1.0)
        scored = [len(s.split()) >= source.SHORT_STRIP_MIN_WORDS for s in strips]
        if len(off_model.logits) == sum(scored):
            logits = iter(off_model.logits)
            off_scores = [next(logits) if ok else -1.0 for ok in scored]
        else:  # 공식 except 경로 (모델 예외 -> -1.0) 가 있었음: 점수 정렬 불가
            off_scores = [None] * len(strips)
            agg["official_score_unaligned"] += 1
        ours = internal_knowledge(q.question, psgs, ev, "selection")
        our_scores = [s.score for s in ours.strips]
        diffs = [abs(x - y) for x, y in zip(off_scores, our_scores, strict=True) if x is not None]
        md = max(diffs) if diffs else 0.0
        max_diff_all = max(max_diff_all, md)
        ours_top = [s.index for s in ours.selected]
        off_top = off_rank[:top_n]
        body = strip_hash(r)
        kept = ordered_subset(q.passages, body)
        cls = classify(
            ours_text=ours.text,
            official_text=off_text,
            ours_top=ours_top,
            official_top=off_top,
            score_max_abs_diff=md,
            ref_subset=kept,
            ref_text=body,
        )
        for k in cls["impl"]:
            agg[f"impl_{k}"] += 1
        for k in cls["ref"]:
            agg[f"ref_{k}"] += 1
        agg["impl_exact"] += cls["impl_exact"]
        agg["ref_exact"] += cls["ref_exact"]
        doc = ev.score_documents(q.question, q.passages)  # 판정용 (truncation 없음)
        doc_scores = [x.score for x in doc]
        ne = [i for i, x in enumerate(q.passages) if x.strip()]
        if kept:
            rule_rows_doc.append({"scores": doc_scores, "kept": kept, "nonempty": ne})
            rule_rows_strip.append({"scores": our_scores, "kept": kept, "nonempty": ne})
        out.write(
            {
                "index": q.index,
                "question": q.question,
                "official_scores": off_scores,
                "ours_scores": our_scores,
                "score_max_abs_diff": md,
                "official_ranking": off_rank,
                "ours_ranking": ours.ranking,
                "official_top3": off_top,
                "ours_top3": ours_top,
                "ref_kept_docs": kept,
                "doc_scores_no_trunc": doc_scores,
                "doc_flags": ev.flags(doc_scores),
                "classification": cls,
                "official_text": off_text,
                "ours_text": ours.text,
                "ref_text": body,
            }
        )
    summary = {
        "n": n,
        "impl_vs_official_code": {
            "exact_string": agg["impl_exact"],
            "score_max_abs_diff": max_diff_all,
            "official_score_unaligned": agg["official_score_unaligned"],
            **{k: agg[f"impl_{k}"] for k in CATEGORIES},
        },
        "official_code_vs_ref_correct": {
            "exact_string": agg["ref_exact"],
            **{k: agg[f"ref_{k}"] for k in CATEGORIES},
        },
        "ref_selection_rules_doc_scores": rule_check(rule_rows_doc) if rule_rows_doc else None,
        "ref_selection_rules_strip_scores": (
            rule_check(rule_rows_strip) if rule_rows_strip else None
        ),
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    write_new(run_dir / "report.md", _report(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {run_dir}")
    return run_dir


def _report(s: dict) -> str:
    i, r = s["impl_vs_official_code"], s["official_code_vs_ref_correct"]
    lines = [
        "# MODE A Refinement 진단",
        "",
        f"문항 수: {s['n']} (selection, 상위 3)",
        "",
        "## 1. 재구현 vs 공식 코드 (같은 문서·같은 가중치)",
        "",
        f"- 최종 문자열 일치: {i['exact_string']}/{s['n']}",
        f"- strip 점수 최대 절대 차이: {i['score_max_abs_diff']:.3g}",
        "- 불일치 유형 (문항 수): " + ", ".join(f"{k} {i[k]}" for k in CATEGORIES if k != "source"),
        "",
        "## 2. 공식 코드 vs 저자 ref/correct",
        "",
        f"- 최종 문자열 일치: {r['exact_string']}/{s['n']}",
        "- 불일치 유형 (문항 수, 중복 가능): " + ", ".join(f"{k} {r[k]}" for k in CATEGORIES),
        "",
        "## 3. ref/correct 문서 선택 규칙 가설 (문서 점수, truncation 없음)",
        "",
        "```",
        json.dumps(s["ref_selection_rules_doc_scores"], indent=2),
        "```",
        "",
        "판정: 1 이 모두 일치하면 구현 오류 아님. 2 의 불일치가 selection/ordering/format",
        "이고 source 가 0 이면 같은 입력 문서에서 다른 규칙으로 만든 사전 생성 파일 차이.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- rules (score run 재사용)
def cmd_rules(a) -> Path:
    repo = Path(a.repo)
    p = _paths(repo)
    queries = load_official_queries(p["test"], source.OFFICIAL_NDOCS, p["sources"])
    ref_raw = load_ref(p["correct"])
    score_path = Path(a.scores) / "scores.jsonl"
    rows_by_idx = {
        x["index"]: x
        for x in (
            json.loads(line)
            for line in score_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    run_dir = _run_dir(a.run_id)
    cfg = {"experiment": "official_crag", "step": "diag-refinement/rules", "scores": str(a.scores)}
    write_run_files(run_dir, cfg, collect_run_metadata(cfg, None, {}))
    rows, by_action = [], {}
    for q, r in zip(queries, ref_raw, strict=True):
        if q.index not in rows_by_idx:
            continue
        kept = ordered_subset(q.passages, strip_hash(r))
        if not kept:
            continue
        sc = rows_by_idx[q.index]
        row = {
            "scores": sc["scores"],
            "kept": kept,
            "nonempty": [i for i, x in enumerate(q.passages) if x.strip()],
        }
        rows.append(row)
        by_action.setdefault(sc["action"], []).append(row)
    summary = {
        "scores": str(score_path),
        "all": rule_check(rows),
        "by_action": {k: rule_check(v) for k, v in sorted(by_action.items())},
    }
    write_new(run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("provenance", "compare", "rules"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=str(DEFAULT_REPO))
        p.add_argument("--run-id", default=None)
    c = sub.choices["compare"]
    c.add_argument("--weights", default=None)
    c.add_argument("--device", default="cpu")
    c.add_argument("--limit", type=int, default=10)
    c.add_argument("--batch-size", type=int, default=1)
    sub.choices["rules"].add_argument("--scores", required=True, help="mode_a.py score run 폴더")
    a = ap.parse_args(argv)
    return {"provenance": cmd_provenance, "compare": cmd_compare, "rules": cmd_rules}[a.cmd](a)


if __name__ == "__main__":
    main()
