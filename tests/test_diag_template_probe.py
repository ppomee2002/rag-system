"""diag_template_probe: 렌더링, raw 엔진 요청, run -> analyze 흐름 (가짜 HTTP)."""

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "crag_system"))
import diag_template_probe as dtp  # noqa: E402
from workload import make_workload, save_workload  # noqa: E402

from rag.pipelines import prompts  # noqa: E402


def test_render_exact():
    c = "Q {x} }{"  # 중괄호는 그대로 보존
    user = "<|start_header_id|>user<|end_header_id|>\n\nQ {x} }{<|eot_id|>"
    asst = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    sysb = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    )
    assert dtp.render("plain", c, "ollama") == user + asst
    assert dtp.render("plain", c, "vllm") == "<|begin_of_text|>" + user + asst
    assert dtp.render("hf", c, "ollama") == sysb + user + asst
    assert dtp.render("hf", c, "vllm") == "<|begin_of_text|>" + sysb + user + asst


def test_raw_clients_request_and_parse():
    seen = {}

    def fake_vllm(url, body, headers):
        seen["v"] = (url, body, headers)
        return {
            "choices": [{"text": " 0.\n", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 542, "completion_tokens": 3},
        }

    def fake_ollama(url, body, headers):
        seen["o"] = (url, body)
        return {"response": "1", "prompt_eval_count": 517, "eval_count": 2, "done_reason": "stop"}

    v = dtp.VLLMRaw("http://h:8000/v1", "m", api_key_env=None, post=fake_vllm)
    r = v.complete("P", 512)
    url, body, _ = seen["v"]
    assert url == "http://h:8000/v1/completions"
    assert body["add_special_tokens"] is False and body["temperature"] == 0.0
    assert body["seed"] == 0 and body["max_tokens"] == 512 and body["prompt"] == "P"
    assert r == {
        "raw_text": " 0.\n",
        "prompt_tokens": 542,
        "out_tokens": 3,
        "finish_reason": "stop",
    }

    o = dtp.OllamaRaw("http://h:11434", "llama3.1:8b", num_ctx=6144, post=fake_ollama)
    r = o.complete("P", 512)
    url, body = seen["o"]
    assert url == "http://h:11434/api/generate" and body["raw"] is True
    assert body["options"] == {"temperature": 0.0, "num_predict": 512, "num_ctx": 6144, "seed": 0}
    assert r["raw_text"] == "1" and r["prompt_tokens"] == 517


# ---------------------------------------------------------------- run -> analyze
GOLD_DOC = "g"


@pytest.fixture
def world(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.jsonl"
    docs = {f"q{q}d{i}": f"text {q} {i}" for q in range(2) for i in range(3)}
    docs["q0d0"] = "the answer is Paris"
    chunks.write_text("".join(json.dumps({"id": k, "text": v}) + "\n" for k, v in docs.items()))
    full = tmp_path / "full"
    full.mkdir()
    cfg = {
        "grading": {"strategy": "full"},
        "llm": {"max_tokens": 512},
        "retriever": {"docs_path": str(chunks)},
        "crag": {"upper_threshold": 0.7, "lower_threshold": 0.3},
    }
    (full / "config.yaml").write_text(yaml.safe_dump(cfg))
    rows, calls = [], []
    for q in range(2):
        ids = [f"q{q}d{i}" for i in range(3)]
        question = f"Where is thing {q}?"
        rows.append(
            {
                "request_index": q,
                "query_id": f"q{q}",
                "warmup": False,
                "question": question,
                "gold_answers": ["Paris"],
                "corrective_action": "incorrect",
                "acc": 0.0,
                "extra": {
                    "retrieval_doc_ids": ids,
                    "grading": {"strategy": "full", "scores_full": [0.0, 0.0, 0.0]},
                },
            }
        )
        for i, d in enumerate(ids):
            content = prompts.GRADE.format(question=question, document=docs[d])
            calls.append(
                {
                    "request_index": q,
                    "query_id": f"q{q}",
                    "warmup": False,
                    "kind": "grade",
                    "idx": i,
                    "prompt_chars": len(content),
                    "prompt_tokens": 125,  # 서버 (hf) 로그
                    "out_tokens": 2,
                }
            )
    for name, data in (("queries.jsonl", rows), ("calls.jsonl", calls)):
        (full / name).write_text("".join(json.dumps(x) + "\n" for x in data))
    wl = save_workload(
        make_workload(["q0", "q1"], arrival="sequential", name="t"), tmp_path / "wl.json"
    )

    def fake_post(url, body, headers):
        if url.endswith("/api/show") or url.endswith("/models"):
            return {}
        p = body["prompt"]
        has_sys = "Cutting Knowledge" in p
        pt = 100 + (25 if has_sys else 0)
        # 가짜 모델: system 블록이 있으면 0, 없으면 정답 문서 (rank 0 of q0) 만 1
        text = "0" if has_sys else ("1" if "Paris" in p else "0.")
        if url.endswith("/completions"):
            return {
                "choices": [{"text": text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": 2},
            }
        return {"response": text, "prompt_eval_count": pt, "eval_count": 2, "done_reason": "stop"}

    monkeypatch.setattr(dtp, "http_post", fake_post)
    monkeypatch.setattr(dtp, "http_get", lambda *a, **k: {})
    return tmp_path, full, wl


def _run(tmp, full, wl, engine, template, label):
    return dtp.main(
        ["run", "--engine", engine, "--model", "m", "--template", template, "--label", label,
         "--full", str(full), "--workload", str(wl), "--out-root", str(tmp / "probe")]
    )  # fmt: skip


def _csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_run_and_analyze(world):
    tmp, full, wl = world
    s_hf = _run(tmp, full, wl, "vllm", "hf", "S-HF")
    s_pl = _run(tmp, full, wl, "vllm", "plain", "S-PLAIN")
    l_pl = _run(tmp, full, wl, "ollama", "plain", "L-PLAIN")
    meta = json.loads((s_hf / "meta.json").read_text())
    assert meta["n_inputs"] == 6 and meta["bos_in_prompt"] is True
    calls = [json.loads(x) for x in (s_hf / "calls.jsonl").read_text().splitlines()]
    assert {c["score"] for c in calls} == {0.0} and all(c["prompt_tokens"] == 125 for c in calls)

    out = dtp.main(
        ["analyze", "--full", str(full), "--run", f"S-HF={s_hf}", "--run", f"S-PLAIN={s_pl}",
         "--run", f"L-PLAIN={l_pl}", "--pair", "S-HF:S-PLAIN", "--bootstrap", "200",
         "--out-root", str(tmp / "an"), "--name", "t"]
    )  # fmt: skip
    gates = {r["run"]: r for r in _csv(out / "gates.csv")}
    assert gates["S-HF"]["inputs_same_as_first"] == "6"
    assert gates["S-HF"]["prompt_tokens_minus_server_log"] == "{0: 6}"
    assert gates["S-PLAIN"]["prompt_tokens_minus_server_log"] == "{-25: 6}"
    assert gates["S-HF"]["score_same_as_server_full"] == "6/6"

    dist = {r["run"]: r for r in _csv(out / "score_actions.csv")}
    assert dist["S-HF"]["n_incorrect"] == "2" and dist["S-PLAIN"]["n_correct"] == "1"
    assert float(dist["S-HF"]["gold_kept_query_rate"]) == 0.0
    assert float(dist["S-PLAIN"]["gold_kept_query_rate"]) == 1.0
    assert float(dist["S-PLAIN"]["gold_doc_high_rate"]) == 1.0

    pair = _csv(out / "pairs.csv")[0]
    assert pair["pair"] == "S-PLAIN - S-HF" and pair["high_only_b"] == "1"
    assert float(pair["incorrect_rate_diff"]) == -0.5
    tr = {(r["from_a"], r["to_b"]): r["n_queries"] for r in _csv(out / "action_transitions.csv")}
    assert tr[("incorrect", "correct")] == "1" and tr[("incorrect", "incorrect")] == "1"


def test_run_refuses_unknown_query(world, tmp_path):
    tmp, full, _ = world
    wl = save_workload(
        make_workload(["q0", "zz"], arrival="sequential", name="u"), tmp_path / "wl2.json"
    )
    with pytest.raises(SystemExit):
        _run(tmp, full, wl, "vllm", "hf", "X")
