"""config 의 `llm:` 블록 -> `InferenceEngine`.

llm:
  provider: vllm            # vllm | ollama
  model: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://127.0.0.1:8000/v1
  ...
"""

from __future__ import annotations

from rag.llm.engine import InferenceEngine
from rag.llm.ollama_engine import OllamaEngine
from rag.llm.vllm_engine import VLLMEngine


def build_engine(llm_cfg: dict) -> InferenceEngine:
    provider = llm_cfg["provider"]
    common = dict(
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 512),
        timeout=llm_cfg.get("timeout", 600.0),
    )
    if provider == "ollama":
        return OllamaEngine(
            **common,
            num_ctx=llm_cfg.get("num_ctx", 8192),
            host=llm_cfg.get("host", "http://127.0.0.1:11434"),
        )
    if provider == "vllm":
        return VLLMEngine(
            **common,
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            seed=llm_cfg.get("seed", 0),
            stream=llm_cfg.get("stream", True),
            send_priority=llm_cfg.get("send_priority", False),
            api_key_env=llm_cfg.get("api_key_env", "VLLM_API_KEY"),
            extra_body=llm_cfg.get("extra_body"),
        )
    raise ValueError(f"지원하지 않는 engine provider: {provider} (ollama | vllm)")
