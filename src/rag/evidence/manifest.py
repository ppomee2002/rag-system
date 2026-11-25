"""Experiment Manifest: 같은 데이터·모델·평가 조건을 재현·대조하기 위한 run 단위 기록.

MODE B run 폴더의 manifest.json. run_meta.json (환경·패키지·GPU) 과 겹치는 값도 있지만 manifest 는
"C1/C2/C3 결과를 서로 비교해도 되는가" 를 판단하는 값만 한곳에 모은다.

  comparison  정책 간에 반드시 같아야 하는 조건 (고정 입력, generator, prompt, 평가 기준).
              comparison_sha256 이 같으면 같은 비교 조건이다. evidence 정책·grader·budget 은
              비교 대상 (조건 차이) 이므로 여기에 넣지 않는다.
  condition   이 run 의 처리 조건 (grader, threshold, 판정 규칙, evidence 정책, budget, correction).

experiment_mode 가 다르면 (A_official_reproduction vs B_controlled) 비교하지 않는다
(compare_manifests 가 차이로 보고). MODE A 는 공식 데이터·저자 지식 파일·completions API 를 쓰는
별도 설정이다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.crag_official import source
from rag.pipelines import prompts

MANIFEST_SCHEMA = "official_crag.manifest.v1"
ROOT = Path(__file__).resolve().parents[3]

# 평가 스크립트 (정답 판정) 버전 = 파일 sha256
EVAL_FILES = ("src/rag/eval/metrics.py", "src/rag/crag_official/generation.py")


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_templates() -> dict[str, dict]:
    """prompt template 별 sha256 (template 원문 그대로)."""
    t = {
        "legacy_grade": ("rag.pipelines.prompts.GRADE", prompts.GRADE),
        "legacy_rewrite": ("rag.pipelines.prompts.REWRITE", prompts.REWRITE),
        "legacy_generate": ("rag.pipelines.prompts.GENERATE", prompts.GENERATE),
        "official_popqa_generate": ("rag.crag_official.source.POPQA_PROMPT", source.POPQA_PROMPT),
        "official_popqa_keywords": (
            "rag.crag_official.source.POPQA_KEYWORD_PROMPT",
            source.POPQA_KEYWORD_PROMPT,
        ),
        "official_ambiguous_template": (
            "rag.crag_official.source.AMBIGUOUS_TEMPLATE",
            source.AMBIGUOUS_TEMPLATE,
        ),
    }
    return {k: {"source": s, "sha256": text_sha(v)} for k, (s, v) in t.items()}


def used_prompts(cfg: dict) -> dict[str, str]:
    """이 설정에서 실제로 쓰는 template 이름 (역할 -> 이름)."""
    gen = (cfg.get("generation") or {}).get("prompt", "legacy")
    qmode = (cfg.get("correction") or {}).get("query_mode", "legacy_rewrite")
    gtype = (cfg.get("grader") or {}).get("type", "official_t5")
    out = {
        "generate": "official_popqa_generate" if gen == "official" else "legacy_generate",
        "correction_query": (
            "official_popqa_keywords" if qmode == "official_keywords" else "legacy_rewrite"
        ),
    }
    if cfg.get("evidence_policy") != "vanilla" and gtype == "llama_prompt":
        out["grade"] = "legacy_grade"
    if str(cfg.get("evidence_policy", "")).startswith("official"):  # official + fallback ablation
        out["ambiguous"] = "official_ambiguous_template"
    return out


def eval_version(root: Path = ROOT) -> dict[str, Any]:
    files = {f: file_sha(root / f) for f in EVAL_FILES}
    return {
        "files": files,
        "sha256": text_sha(json.dumps(files, sort_keys=True)),
        "primary_metric": "acc (rag.eval.metrics.accuracy: 정규화 후 포함)",
        "also": "official_match (공식 metrics.py match: 대소문자 구분 포함)",
    }


def generator_view(cfg: dict) -> dict[str, Any]:
    llm = cfg.get("llm") or {}
    gen = cfg.get("generation") or {}
    return {
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "revision": llm.get("revision"),  # null = 미지정 (서버 모델 snapshot 을 config 에 적을 것)
        "api": llm.get("api", "chat"),
        "temperature": llm.get("temperature"),
        "top_p": llm.get("top_p"),  # null = 엔진/서버 기본값 (요청에 넣지 않음)
        # chat 엔진 (VLLMEngine) 은 top_p 를 보내지 않는다 -> 실제로 요청에 들어간 값만
        "top_p_sent": llm.get("top_p") if llm.get("api", "chat") == "completions" else None,
        "max_tokens": llm.get("max_tokens"),
        "seed": llm.get("seed"),
        "generation_prompt": gen.get("prompt", "legacy"),
        "generation_max_tokens": gen.get("max_tokens"),
    }


def comparison_view(cfg: dict, input_info: dict | None, question_ids: list[str] | None) -> dict:
    """정책 간에 같아야 하는 조건.

    오래된 run (manifest 없음) 도 config.yaml + run_meta 로 만든다.
    """
    inputs = input_info or {}
    fm = inputs.get("manifest") or {}
    qids = list(question_ids or [])
    return {
        "experiment_mode": cfg.get("experiment_mode", "B_controlled"),
        "inputs_sha256": inputs.get("sha256"),
        "dataset_sha256": (fm.get("dataset") or {}).get("sha256"),
        "question_ids_sha256": text_sha("\n".join(qids)) if qids else None,
        "n_questions": len(qids) if qids else None,
        "generator": generator_view(cfg),
        "web_top_k": (cfg.get("web_search") or {}).get("top_k", 5),
        "seed": cfg.get("seed"),
    }


def comparison_sha(view: dict) -> str:
    return text_sha(json.dumps(view, sort_keys=True, ensure_ascii=False, default=str))


def build_manifest(
    cfg: dict,
    input_info: dict,
    question_ids: list[str],
    pipeline_desc: dict,
    evaluator_desc: dict | None,
    weights: dict | None,
    web_desc: dict | None,
    git: dict,
    analysis_scripts: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    fm = input_info.get("manifest") or {}
    comp = comparison_view(cfg, input_info, question_ids)
    grader = pipeline_desc.get("grader") or {}
    upper = grader.get("upper")
    lower = grader.get("lower")
    templates = prompt_templates()
    used = used_prompts(cfg)
    return {
        "schema": MANIFEST_SCHEMA,
        "experiment_mode": comp["experiment_mode"],
        "not_comparable_with": (
            "A_official_reproduction (공식 데이터·저자 지식 파일·completions API — 다른 설정)"
            if comp["experiment_mode"] == "B_controlled"
            else "B_controlled"
        ),
        "comparison": comp,
        "comparison_sha256": comparison_sha(comp),
        "dataset": fm.get("dataset"),
        "question_ids": question_ids,
        "question_ids_file": {
            "path": fm.get("question_ids_file"),
            "sha256_of_ids": fm.get("question_ids_sha256"),
        },
        "subset_caveat": fm.get("caveat"),
        "retrieval_inputs": {
            "frozen_path": input_info.get("path"),
            "sha256": input_info.get("sha256"),
            "top_k": fm.get("top_k"),
            "docs_path": fm.get("docs_path"),
            "retriever": fm.get("retriever"),
            "embedding": fm.get("embedding"),
            "reference_runs": fm.get("reference_runs"),
            "limit": (cfg.get("inputs") or {}).get("limit"),
        },
        "prompts": {
            "used": {role: {"name": n, **templates[n]} for role, n in used.items()},
            "all": templates,
            "version_sha256": text_sha(json.dumps(templates, sort_keys=True)),
        },
        "official_t5": (
            None
            if evaluator_desc is None
            else {
                "weights_fingerprint": weights,
                "upper": evaluator_desc.get("upper"),
                "lower": evaluator_desc.get("lower"),
                "normalize_passage": evaluator_desc.get("normalize_passage"),
                "backend": (evaluator_desc.get("backend") or {}).get("backend"),
                "weights_url": source.EVALUATOR_WEIGHTS_URL,
                "official_commit": source.OFFICIAL_COMMIT,
            }
        ),
        "generator": comp["generator"],
        "grader": {
            "type": grader.get("grader_type"),
            "score_scale": grader.get("score_scale") or grader.get("type"),
            "upper": upper,
            "lower": lower,
            "decision_rule": (
                None
                if upper is None
                else (
                    f"doc: score>={upper}->pass, score>={lower}->partial, else fail; "
                    "query: any pass->correct, else any partial->ambiguous, else incorrect"
                )
            ),
        },
        "evidence_policy": (pipeline_desc.get("policy") or {}).get("name"),
        "policy_id": pipeline_desc.get("policy_id"),
        "budget": pipeline_desc.get("budget"),
        "correction": {
            "query_mode": (cfg.get("correction") or {}).get("query_mode", "legacy_rewrite"),
            "wiki_prefix": (cfg.get("correction") or {}).get("wiki_prefix", False),
            "on_web_error": (cfg.get("correction") or {}).get("on_web_error", "fail"),
            "web_search": web_desc,
        },
        "evaluation": eval_version(),
        "analysis_scripts": analysis_scripts or {},
        "git": {
            "commit": git.get("commit"),
            "branch": git.get("branch"),
            "dirty": git.get("dirty"),
        },
    }


def compare_manifests(a: dict, b: dict) -> list[str]:
    """두 run 의 비교 조건 차이 (빈 목록이면 같은 조건).

    None (설정 없음) 과 값이 있는 경우도 차이로 본다 (예: temperature 미지정 vs 0.5).
    """
    ca, cb = a.get("comparison") or a, b.get("comparison") or b
    diffs = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif x != y:
            diffs.append(f"{path}: {x!r} != {y!r}")

    walk(ca, cb, "")
    return diffs
