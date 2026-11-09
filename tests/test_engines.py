import pytest

from rag.llm.engine import EngineError, LLMRequest
from rag.llm.engine_llm import EngineLLM
from rag.llm.factory import build_engine
from rag.llm.ollama_engine import OllamaEngine
from rag.llm.ollama_llm import OllamaLLM
from rag.llm.vllm_engine import VLLMEngine, parse_prometheus
from rag.pipelines import prompts
from rag.pipelines.crag import CRAG
from rag.telemetry.calls import CallRecorder, infer_call_kind
from rag.types import Document, QAExample


# ---------------------------------------------------------------- vLLM
@pytest.mark.parametrize("stream", [False, True])
def test_vllm_complete_parses_text_usage_and_timing(mock_server, stream):
    srv = mock_server("vllm")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", stream=stream, api_key_env=None)
    res = eng.complete(LLMRequest("What is the capital of France?", max_tokens=16))
    assert res.text == "Paris is the answer"
    assert res.prompt_tokens == 6 and res.completion_tokens == 4
    assert res.finish_reason == "stop"
    assert res.request_id == "chatcmpl-1"
    assert res.wall > 0
    if stream:
        assert res.ttft is not None and 0 <= res.ttft <= res.wall
    else:
        assert res.ttft is None
    body = srv.chat_requests()[0]["body"]
    assert body["model"] == "mock-model"
    assert body["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert body["max_tokens"] == 16 and body["temperature"] == 0.0 and body["seed"] == 0
    assert "priority" not in body
    assert ("stream_options" in body) == stream


def test_vllm_priority_only_sent_when_enabled(mock_server):
    srv = mock_server("vllm", sched_policy="priority")
    off = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None)
    on = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None, send_priority=True)
    assert off.complete(LLMRequest("q", priority=3)).priority_sent is None
    assert on.complete(LLMRequest("q", priority=3)).priority_sent == 3
    bodies = [r["body"] for r in srv.chat_requests()]
    assert "priority" not in bodies[0] and bodies[1]["priority"] == 3


def test_vllm_rejected_priority_raises_engine_error(mock_server):
    srv = mock_server("vllm", sched_policy="fcfs")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None, send_priority=True)
    with pytest.raises(EngineError) as e:
        eng.complete(LLMRequest("q", priority=5))
    assert e.value.status == 400


def test_vllm_api_key_from_env_only(mock_server, monkeypatch):
    srv = mock_server("vllm", api_key="secret-xyz")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env="TEST_VLLM_KEY")
    monkeypatch.delenv("TEST_VLLM_KEY", raising=False)
    with pytest.raises(EngineError) as e:
        eng.complete(LLMRequest("q"))
    assert e.value.status == 401
    monkeypatch.setenv("TEST_VLLM_KEY", "secret-xyz")
    assert eng.complete(LLMRequest("q")).text
    desc = eng.describe()
    assert desc["api_key_configured"] is True
    assert "secret-xyz" not in str(desc)
    assert desc["version"] == {"version": "0.0.0-mock"}
    assert desc["models"]["data"][0]["max_model_len"] == 6144


def test_vllm_connection_error_is_engine_error():
    eng = VLLMEngine("m", base_url="http://127.0.0.1:9/v1", api_key_env=None, timeout=2)
    with pytest.raises(EngineError):
        eng.complete(LLMRequest("q"))
    assert "error" in eng.describe()["version"]


def test_parse_prometheus(mock_server):
    srv = mock_server("vllm")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None)
    m = parse_prometheus(eng.metrics_text())
    assert any(k.startswith("vllm:num_preemptions_total") for k in m)
    assert all(not k.startswith("#") for k in m)


# ---------------------------------------------------------------- Ollama
def test_ollama_engine_sends_same_payload_as_legacy_ollama_llm(mock_server):
    srv = mock_server("ollama", model="llama3.1:8b")
    eng = OllamaEngine("llama3.1:8b", max_tokens=64, num_ctx=6144, host=srv.url)
    res = eng.complete(LLMRequest("Q?"))
    assert res.text == "Paris is the answer"
    assert res.prompt_tokens == 1 and res.completion_tokens == 4
    assert res.server == pytest.approx({"total": 2.0, "load": 0.1, "prompt_eval": 0.5, "eval": 1.4})
    assert res.priority_sent is None

    legacy_payloads = []
    OllamaLLM(
        "llama3.1:8b",
        max_tokens=64,
        num_ctx=6144,
        post=lambda p: legacy_payloads.append(p) or {"message": {"content": "x"}},
    ).generate("Q?")
    assert srv.chat_requests()[0]["body"] == legacy_payloads[0]


def test_ollama_engine_keeps_truncation_check_and_describe(mock_server):
    srv = mock_server("ollama", model="llama3.1:8b")
    eng = OllamaEngine("llama3.1:8b", max_tokens=64, num_ctx=10, host=srv.url)
    with pytest.raises(RuntimeError, match="num_ctx"):
        eng.complete(LLMRequest("Q?"))
    desc = eng.describe()
    assert desc["model_info"]["details"]["quantization_level"] == "Q4_K_M"
    assert desc["placement"]["size_vram"] > 0
    with pytest.raises(ValueError):
        eng.complete(LLMRequest("Q?", response_format={"type": "json_object"}))


def test_factory_builds_both_engines():
    assert isinstance(
        build_engine({"provider": "ollama", "model": "m", "num_ctx": 6144}), OllamaEngine
    )
    v = build_engine({"provider": "vllm", "model": "m", "send_priority": True, "stream": False})
    assert isinstance(v, VLLMEngine) and v.send_priority and not v.stream
    with pytest.raises(ValueError):
        build_engine({"provider": "anthropic", "model": "m"})


# ---------------------------------------------------------------- EngineLLM adapter
def test_infer_call_kind_matches_prompt_templates():
    assert infer_call_kind(prompts.GRADE.format(question="q", document="d")) == "grade"
    assert infer_call_kind(prompts.REWRITE.format(question="q")) == "rewrite"
    assert infer_call_kind(prompts.GENERATE.format(context="c", question="q")) == "generate"
    assert infer_call_kind("hello") == "other"


class FakeGate:
    def __init__(self):
        self.events = []

    def acquire(self, priority):
        self.events.append(("acquire", priority))

    def release(self):
        self.events.append(("release",))


def test_engine_llm_records_calls_and_uses_gate(mock_server):
    srv = mock_server("vllm", sched_policy="priority")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None, send_priority=True)
    rec, gate = CallRecorder(), FakeGate()
    llm = EngineLLM(
        eng,
        recorder=rec,
        gate=gate,
        priority_fn=lambda kind, idx, prompt: 10.0 + idx,
        engine_priority_fn=lambda p: int(p),
    )
    grade = prompts.GRADE.format(question="q", document="d")
    assert llm.generate(grade) == "Paris is the answer"
    llm.generate(grade, max_tokens=8)
    calls = rec.to_dicts()
    assert [(c["kind"], c["idx"], c["priority"], c["engine_priority"]) for c in calls] == [
        ("grade", 0, 10.0, 10),
        ("grade", 1, 11.0, 11),
    ]
    assert gate.events == [("acquire", 10.0), ("release",), ("acquire", 11.0), ("release",)]
    c = calls[0]
    assert c["engine"] == "vllm" and c["prompt_tokens"] and c["out_tokens"] == 4
    assert c["t_enqueue"] <= c["t_start"] <= c["t_first_token"] <= c["t_end"]
    assert c["server_total"] is None and c["error"] is None
    assert srv.chat_requests()[1]["body"]["max_tokens"] == 8
    assert rec.summary()["n_calls_by_kind"] == {"grade": 2}


def test_engine_llm_records_error_and_releases_gate(mock_server):
    srv = mock_server("vllm", sched_policy="fcfs")
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None, send_priority=True)
    rec, gate = CallRecorder(), FakeGate()
    llm = EngineLLM(eng, recorder=rec, gate=gate, engine_priority_fn=lambda p: 7)
    with pytest.raises(EngineError):
        llm.generate("hello")
    assert gate.events[-1] == ("release",)
    (call,) = rec.to_dicts()
    assert call["kind"] == "other" and "EngineError" in call["error"]


class StaticRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k):
        return self.docs[:top_k]

    def search(self, query, top_k):
        return self.docs[:top_k]


def test_existing_crag_pipeline_runs_unchanged_on_engine(mock_server):
    def responder(prompt):
        if infer_call_kind(prompt) == "grade":
            return "0.9" if "doc0" in prompt else "0.1"
        return "Paris"

    srv = mock_server("vllm", responder=responder)
    eng = VLLMEngine("mock-model", base_url=srv.url + "/v1", api_key_env=None)
    rec = CallRecorder()
    docs = [Document(id=str(i), text=f"doc{i}") for i in range(3)]
    crag = CRAG(StaticRetriever(docs), EngineLLM(eng, recorder=rec), StaticRetriever(docs), top_k=3)
    res = crag.run(QAExample(id="q1", question="Capital?", answers=["Paris"]))
    assert res.answer == "Paris"
    assert res.corrective_action == "correct"
    assert res.grader_scores == [0.9, 0.1, 0.1]
    assert [(c["kind"], c["idx"]) for c in rec.to_dicts()] == [
        ("grade", 0),
        ("grade", 1),
        ("grade", 2),
        ("generate", 0),
    ]
