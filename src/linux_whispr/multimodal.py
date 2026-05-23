"""Multimodal pipeline (SPEC-005).

Glues the file-context loader to a LiteLLM vision-capable model. Mirrors the
shape of :mod:`linux_whispr.edit_selection`: GUI-free, lazy networking, never
raises — failure is reported via :class:`AskResult`.

Why a separate ``call_multimodal`` instead of extending ``OpenAILLMBackend``:
the existing backend takes plain ``system_prompt`` + ``user_prompt`` strings
and is used by the refinement and edit paths. Threading a multi-part content
list through it would complicate code that doesn't need it — so the vision
call lives here, isolated, and uses the same ``openai`` client under the hood.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from linux_whispr.ai.openai_llm import OpenAILLMBackend
from linux_whispr.ai.prompts.vision import VISION_SYSTEM_PROMPT, build_vision_content
from linux_whispr.config import AppConfig
from linux_whispr.input.file_context import FileContext, load_file

logger = logging.getLogger(__name__)

DEFAULT_VISION_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_VISION_MODEL = "vision-gemma3"
DEFAULT_VISION_API_KEY = "sk-litellm-local"


def resolve_vision_api_key(config: AppConfig) -> str:
    """Resolve the LiteLLM key — env first, then config, then local default."""
    return (
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or config.ai.api_key
        or DEFAULT_VISION_API_KEY
    )


def build_vision_llm_backend(
    config: AppConfig,
    *,
    model_override: str | None = None,
) -> OpenAILLMBackend:
    """Create the OpenAI-compatible backend the multimodal mode talks to.

    Picks ``vision-gemma3`` by default — the configured ``ai.model`` is only
    used when it explicitly looks like a vision model (starts with ``vision``).
    Otherwise we'd accidentally send images to ``cerebras-qwen`` (text-only).
    """
    base_url = config.ai.base_url or DEFAULT_VISION_BASE_URL
    if model_override:
        model = model_override
    elif config.ai.model and config.ai.model.startswith("vision"):
        model = config.ai.model
    else:
        model = DEFAULT_VISION_MODEL
    api_key = resolve_vision_api_key(config)
    return OpenAILLMBackend(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True)
class AskResult:
    """Outcome of one multimodal ask.

    Attributes:
        answer: The LLM's reply (empty when ``ok`` is False).
        used_files: Paths of files that were successfully loaded and attached.
        ok: True when the LLM returned usable, non-empty text.
        error: Reason on failure, ``""`` otherwise.
    """

    answer: str
    used_files: list[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""


def _coerce_files(
    files: Iterable[str | Path | FileContext] | None,
) -> tuple[list[FileContext], list[str]]:
    """Load any path inputs, pass through ready :class:`FileContext` objects.

    Returns ``(loaded, skipped_paths)`` so callers can report which inputs
    didn't make it (missing, oversized, undecodable).
    """
    loaded: list[FileContext] = []
    skipped: list[str] = []
    if not files:
        return loaded, skipped
    for item in files:
        if isinstance(item, FileContext):
            loaded.append(item)
            continue
        ctx = load_file(item)
        if ctx is None:
            skipped.append(str(item))
        else:
            loaded.append(ctx)
    return loaded, skipped


def call_multimodal(
    backend: OpenAILLMBackend,
    system_prompt: str,
    content: Sequence[dict],
) -> str:
    """Send a multimodal chat-completion call and return the answer text.

    Lazy-loads the ``openai`` client through ``backend.load()`` so we share the
    same client instance / configuration the rest of the codebase uses. Raises
    on transport errors — callers catch and translate to ``AskResult``.
    """
    if backend._client is None:  # noqa: SLF001 — explicit, intentional reuse
        backend.load()
    client = backend._client  # noqa: SLF001
    assert client is not None  # backend.load() guarantees this

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": list(content)})

    response = client.chat.completions.create(  # type: ignore[union-attr]
        model=backend._model,  # noqa: SLF001
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


class MultimodalPipeline:
    """Headless multimodal Q&A pipeline.

    Construction wires the LLM backend but contacts neither the network nor
    the file system. :meth:`ask` does both work items in one call.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        backend: OpenAILLMBackend | None = None,
        model_override: str | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._backend = backend or build_vision_llm_backend(
            self._config, model_override=model_override
        )

    @property
    def backend(self) -> OpenAILLMBackend:
        """The LLM backend used for the call (test/inspection hook)."""
        return self._backend

    def ask(
        self,
        question: str,
        files: Iterable[str | Path | FileContext] | None = None,
    ) -> AskResult:
        """Ask the model ``question`` with the attached ``files``.

        With no files this is a plain text completion; with files it builds an
        OpenAI multimodal message. Every error path returns ``AskResult(ok=False)``
        with the reason — the caller never has to catch.
        """
        if not question.strip():
            return AskResult(answer="", ok=False, error="no question given")

        loaded, skipped = _coerce_files(files)
        if skipped:
            logger.info("Skipped %d unreadable/oversized inputs: %s", len(skipped), skipped)

        used_paths = [f.path for f in loaded]

        try:
            if not loaded:
                answer = self._ask_text_only(question)
            else:
                content = build_vision_content(question, loaded)
                answer = call_multimodal(self._backend, VISION_SYSTEM_PROMPT, content)
        except Exception as exc:
            logger.exception("Multimodal ask failed")
            return AskResult(answer="", used_files=used_paths, ok=False, error=f"LLM error: {exc}")

        answer = (answer or "").strip()
        if not answer:
            return AskResult(
                answer="",
                used_files=used_paths,
                ok=False,
                error="LLM returned empty text",
            )
        return AskResult(answer=answer, used_files=used_paths, ok=True)

    def _ask_text_only(self, question: str) -> str:
        """Plain text completion path — used when no files are attached."""
        result = self._backend.generate(
            system_prompt=VISION_SYSTEM_PROMPT,
            user_prompt=question.strip(),
        )
        return result.text
