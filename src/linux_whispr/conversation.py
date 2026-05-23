"""Conversation pipeline assembly (SPEC-004).

The conversation mode is the spoken-reply half of 2Key: the user holds the
hotkey, asks a question, and the assistant answers out loud:

    spoken question  ─►  STT(base, local)  ─►  LiteLLM (cerebras-qwen)  ─►  TTS

Like ``dictation.py`` and ``edit_selection.py``, construction is GUI-free and
cheap. No network, no audio device, no model load until a method that needs
them is called. The pipeline never raises across its public surface — every
failure path is wrapped into a :class:`ConversationResult` so the caller can
report a friendly message instead of crashing the assistant.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from linux_whispr.ai.openai_llm import OpenAILLMBackend
from linux_whispr.ai.prompts.conversation import (
    CONVERSATION_SYSTEM_PROMPT,
    build_conversation_prompt,
)
from linux_whispr.config import AppConfig
from linux_whispr.dictation import DictationPipeline
from linux_whispr.output.tts import NullBackend, TtsBackend

logger = logging.getLogger(__name__)

# Same gateway defaults as edit_selection: the local LiteLLM proxy fronts the
# free cerebras-qwen model. Config can override every field.
DEFAULT_CONVERSATION_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_CONVERSATION_MODEL = "cerebras-qwen"
DEFAULT_CONVERSATION_API_KEY = "sk-litellm-local"


def resolve_conversation_api_key(config: AppConfig) -> str:
    """Resolve the LiteLLM key: env first, then config, then the local default.

    Mirrors how edit-selection and the refinement path resolve their key so all
    three call the gateway with the same credential precedence.
    """
    return (
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or config.ai.api_key
        or DEFAULT_CONVERSATION_API_KEY
    )


def build_conversation_llm_backend(config: AppConfig) -> OpenAILLMBackend:
    """Create the OpenAI-compatible backend the conversation mode talks to.

    Uses the configured LiteLLM endpoint/model when set, otherwise the local
    cerebras-qwen defaults. Lazy — no network call happens at construction.
    """
    base_url = config.ai.base_url or DEFAULT_CONVERSATION_BASE_URL
    model = config.ai.model or DEFAULT_CONVERSATION_MODEL
    api_key = resolve_conversation_api_key(config)
    return OpenAILLMBackend(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True)
class ConversationResult:
    """Outcome of a single conversation turn.

    Attributes:
        transcript: What the STT thought the user said (empty when not run).
        reply: The LLM's spoken reply text (empty on failure).
        spoken: True when the TTS backend reported it played the reply.
        ok: True only when both LLM and TTS reported success.
        error: Human-readable reason when ``ok`` is False, else "".
    """

    transcript: str
    reply: str
    spoken: bool
    ok: bool
    error: str = ""


class ConversationPipeline:
    """Headless view of the listen → think → speak path.

    Construction is cheap: it builds the LLM backend and resolves the TTS
    backend but contacts neither the network nor the audio system. The heavy
    work happens in :meth:`process` (LLM + TTS) and :meth:`run` (STT + the
    rest). Both wrap every error into a :class:`ConversationResult` rather
    than raising.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        backend: OpenAILLMBackend | None = None,
        tts: TtsBackend | None = None,
        dictation: DictationPipeline | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._backend = backend or build_conversation_llm_backend(self._config)
        self._tts: TtsBackend = tts or NullBackend()
        # The conversation reuses dictation's STT — same model load, same
        # config, no parallel pipeline.
        self._dictation = dictation or DictationPipeline(self._config)

    @property
    def tts(self) -> TtsBackend:
        """The TTS backend used by this pipeline."""
        return self._tts

    @property
    def dictation(self) -> DictationPipeline:
        """The dictation pipeline used for STT inside conversation mode."""
        return self._dictation

    def _ask_llm(self, transcript: str) -> tuple[str, str]:
        """Return (reply, error). reply is empty when error is set."""
        user_prompt = build_conversation_prompt(transcript)
        try:
            result = self._backend.generate(
                system_prompt=CONVERSATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.exception("Conversation LLM call failed")
            return "", f"LLM error: {exc}"

        reply = (result.text or "").strip()
        if not reply:
            return "", "LLM returned empty reply"

        logger.info(
            "Conversation reply: %d chars (model=%s, tokens=%d)",
            len(reply),
            result.model,
            result.tokens_used,
        )
        return reply, ""

    def process(self, transcript: str) -> ConversationResult:
        """Take an already-transcribed question and produce the spoken answer.

        Pure function-of-text path: empty in → friendly skip; LLM down → ok=False
        with reason; TTS down → ok=False but the reply text is still returned so
        the caller can fall back to showing it.
        """
        cleaned = transcript.strip()
        if not cleaned:
            return ConversationResult(
                transcript="",
                reply="",
                spoken=False,
                ok=False,
                error="no transcript",
            )

        reply, error = self._ask_llm(cleaned)
        if error:
            return ConversationResult(
                transcript=cleaned,
                reply="",
                spoken=False,
                ok=False,
                error=error,
            )

        try:
            spoken = bool(self._tts.speak(reply))
        except Exception as exc:
            logger.exception("TTS backend raised — treating as not spoken")
            spoken = False
            tts_error = f"TTS error: {exc}"
        else:
            tts_error = "" if spoken else "TTS unavailable"

        if not spoken:
            return ConversationResult(
                transcript=cleaned,
                reply=reply,
                spoken=False,
                ok=False,
                error=tts_error,
            )
        return ConversationResult(
            transcript=cleaned,
            reply=reply,
            spoken=True,
            ok=True,
        )

    def run(self, wav_bytes: bytes) -> ConversationResult:
        """Full live flow: STT → LLM → TTS. Returns a :class:`ConversationResult`."""
        if not wav_bytes:
            return ConversationResult(
                transcript="",
                reply="",
                spoken=False,
                ok=False,
                error="no audio",
            )

        try:
            stt_result = self._dictation.transcribe(wav_bytes)
        except Exception as exc:
            logger.exception("Conversation STT failed")
            return ConversationResult(
                transcript="",
                reply="",
                spoken=False,
                ok=False,
                error=f"STT error: {exc}",
            )

        transcript = (stt_result.text or "").strip()
        return self.process(transcript)
