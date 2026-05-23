"""OpenAI GPT backend for AI text refinement."""

from __future__ import annotations

import logging

from linux_whispr.ai.base import LLMBackend, RefinementResult

logger = logging.getLogger(__name__)


class OpenAILLMBackend(LLMBackend):
    """LLM backend using OpenAI-compatible API (GPT-4o-mini, LiteLLM, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "") -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or None
        self._client: object | None = None

    def load(self) -> None:
        """Initialize the OpenAI client."""
        from openai import OpenAI

        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        logger.info("OpenAI LLM client initialized (model=%s, base_url=%s)", self._model, self._base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> RefinementResult:
        if self._client is None:
            self.load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return RefinementResult(text=text, model=self._model, tokens_used=tokens)

    def is_available(self) -> bool:
        return bool(self._api_key)
