class AnthropicLLM:
    """Anthropic Messages API 기반 `LLM` 구현체.

    temperature 를 지정하므로 sampling 파라미터를 받는 모델(예: claude-haiku-4-5)을 사용해야 한다.
    anthropic SDK 1.x 는 temperature 키워드를 제거했으므로 extra_body 로 요청 JSON 에 직접 넣는다.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 2,
        client=None,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=max_retries)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body={"temperature": kwargs.get("temperature", self.temperature)},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
