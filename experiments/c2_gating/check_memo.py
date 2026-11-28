"""공식 재현 환경에서 internal refinement 의 중복 T5 채점 여부와 memo 적용 가능성 확인.

공식 selection 모드: Correct/Ambiguous 의 internal refinement 는 grading 과 같은 10문서를 strip 으로
다시 채점한다. 차이: grading = normalize_passage(문서) + truncation 없음, refinement = '[sep]' 분할
(앞뒤 공백 유지) + truncation=True + 4단어 미만 -1.0. memo 는 두 입력 **텐서가 같은** 쌍에만 적용.

단계
  1. 토큰화만 (GPU 불필요, 기본 1,399문항 전체): 문서별 텐서 동일 여부 / 512 초과 / 규칙 차이 집계
  2. --forward: 앞 --n 문항에서 두 경로를 실제로 따로 계산해 점수 비트 일치, memo 적용 refinement
    결과
     (선택 strip, 텍스트) 와 memo 미적용 결과가 같은지

    CUDA_VISIBLE_DEVICES=1 uv run --extra crag --extra official python \
        experiments/c2_gating/check_memo.py \
        --mode-a-config experiments/official_crag/configs/mode_a_popqa.yaml --forward --n 100
-> results/c2_gating/check_memo/<ts>/{summary.json, docs.jsonl}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_components import build_runner  # noqa: E402
from c2_data import load_items  # noqa: E402

from rag.config import load_layered_config  # noqa: E402
from rag.crag_official import source  # noqa: E402
from rag.crag_official.evaluator import normalize_passage, pair_text  # noqa: E402
from rag.crag_official.refinement import internal_knowledge  # noqa: E402
from rag.crag_official.t5_service import JobScorer, OfficialRules, T5Service  # noqa: E402
from rag.telemetry.run_meta import JsonlWriter, create_run_dir, write_new  # noqa: E402

ROOT = HERE.parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-a-config", required=True)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "CRAG"))
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="--forward 문항 수")
    a = ap.parse_args(argv)
    cfg = load_layered_config(a.mode_a_config, [a.config], a.set)
    items, _ = load_items(a.repo)
    runner = build_runner(cfg)
    d = create_run_dir("c2_gating/check_memo", "results")
    out = JsonlWriter(d / "docs.jsonl")
    cnt: Counter = Counter()
    same_tensor: dict[tuple[int, int], bool] = {}
    for it in items:
        if len(it.refine_passages) != len(it.passages):
            cnt["questions_split_mismatch"] += 1
            continue
        for k, (g, r) in enumerate(zip(it.passages, it.refine_passages, strict=True)):
            gp = normalize_passage(g)
            g_rule = None if gp.strip() else "empty"
            r_rule = "short" if len(r.split()) < source.SHORT_STRIP_MIN_WORDS else None
            row = {"index": it.index, "doc": k, "grading_rule": g_rule, "refine_rule": r_rule}
            cnt["docs"] += 1
            if g_rule or r_rule:
                cnt[f"rule:{g_rule}/{r_rule}"] += 1
                row["same_tensor"] = None
            else:
                eg = runner.encode(pair_text(it.question, gp), truncation=False)
                er = runner.encode(pair_text(it.question, r), truncation=True)
                row.update(
                    n_tokens=eg.n_tokens,
                    over_512=eg.length > source.EVAL_MAX_LENGTH,
                    same_tensor=eg.ids == er.ids,
                    same_text=gp == r,
                )
                cnt["model_docs"] += 1
                cnt["over_512"] += row["over_512"]
                cnt["same_tensor"] += row["same_tensor"]
                cnt["same_text"] += row["same_text"]
                same_tensor[(it.index, k)] = row["same_tensor"]
            out.write(row)
    summary: dict = {"tokenization": dict(cnt), "runner": runner.describe()}
    if a.forward:
        svc = T5Service(runner)  # memo 없이 두 경로를 따로 계산
        rules = OfficialRules(svc)
        memo_svc = T5Service(runner, memo=True, memo_verify=True)
        mrules = OfficialRules(memo_svc)
        f: Counter = Counter()
        diffs = []
        for it in items[: a.n]:
            g = rules.submit_documents(f"g{it.index}", it.question, it.passages).result()
            ref = internal_knowledge(
                it.question, it.refine_passages, JobScorer(rules, f"g{it.index}", "internal_refine")
            )
            mrules.submit_documents(f"m{it.index}", it.question, it.passages).result()
            mem = internal_knowledge(
                it.question,
                it.refine_passages,
                JobScorer(mrules, f"m{it.index}", "internal_refine"),
            )
            memo_svc.release_request(f"m{it.index}")
            for k, (gs, rs) in enumerate(zip(g, ref.strips, strict=True)):
                if gs.source == "model" and rs.score_source == "model":
                    f["both_model"] += 1
                    st = same_tensor.get((it.index, k))
                    key = "same_tensor" if st else "diff_tensor"
                    f[f"{key}_docs"] += 1
                    f[f"{key}_score_bit_equal"] += gs.score == rs.score
                    if st and gs.score != rs.score:
                        diffs.append(
                            {"index": it.index, "doc": k, "grading": gs.score, "refine": rs.score}
                        )
            f["questions"] += 1
            f["refinement_text_equal_with_memo"] += ref.text == mem.text
            f["refinement_ranking_equal_with_memo"] += ref.ranking == mem.ranking
        jobs = [j for j in memo_svc.jobs if j.kind == "internal_refine"]
        summary["forward"] = {
            **dict(f),
            "memo_hits": sum(j.n_memo_hits for j in jobs),
            "memo_mismatch": sum(j.memo_mismatch for j in jobs),
            "max_memo_abs_diff": max([j.max_memo_abs_diff for j in jobs] or [0.0]),
            "refine_busy_no_memo_s": sum(j.busy_s for j in svc.jobs if j.kind == "internal_refine"),
            "refine_forward_pairs_no_memo": sum(
                j.n_forward for j in svc.jobs if j.kind == "internal_refine"
            ),
            "score_diff_examples": diffs[:20],
            "note": "memo_verify: hit 도 다시 계산 -> busy 는 memo 절감량이 아님. "
            "절감 = hit 수 x 쌍당 시간",
        }
        fw = summary["forward"]
        ok = (
            fw["memo_mismatch"] == 0
            and fw.get("same_tensor_docs", 0) == fw.get("same_tensor_score_bit_equal", 0)
            and fw["refinement_text_equal_with_memo"] == fw["questions"]
            and fw["refinement_ranking_equal_with_memo"] == fw["questions"]
        )
        summary["memo_decision"] = {
            "memo_allowed": ok,
            "rule": "같은 텐서 문서의 grading/refinement 점수 비트 일치 + memo_verify mismatch 0 + "
            "memo 적용 refinement 결과 (텍스트·순위) 가 전 문항 동일",
            "next": "memo_allowed=true 이고 memo_verify smoke 도 통과하면 t5_service.memo=true 를 "
            "B/C/Cbn 모두에 적용",
        }
        svc.close()
        memo_svc.close()
    write_new(d / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "runner"},
            indent=2,
            ensure_ascii=False,
            default=str,
        )[:4000]
    )
    print(f"saved -> {d}")


if __name__ == "__main__":
    main()
