import json
import urllib.request


class OllamaLLM:
    """로컬 Ollama 서버 (/api/chat) 기반 `LLM` 구현체. API 비용 없이 실행할 때 사용한다.

    num_ctx 는 prompt + 답변이 들어갈 context 길이. 이보다 긴 prompt 는 Ollama 가 말없이 잘라내므로
    잘렸다고 판단되면 오류를 낸다.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        num_ctx: int = 8192,
        host: str = "http://localhost:11434",
        timeout: float = 600.0,
        post=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.url = f"{host}/api/chat"
        self.timeout = timeout
        self._post = post or self._http_post

    def _http_post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        res = self._post(
            {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                    "seed": 0,
                },
            }
        )
        if res.get("prompt_eval_count", 0) + max_tokens > self.num_ctx:
            raise RuntimeError(
                f"prompt {res['prompt_eval_count']} tokens + max_tokens {max_tokens} "
                f"> num_ctx {self.num_ctx}: prompt 가 잘렸을 수 있음"
            )
        return res["message"]["content"].strip()
