"""Tests for the conversation mode (SPEC-004).

Covers the auto-checkable acceptance criteria without touching the LiteLLM
gateway, STT model, or audio devices:

  * the LLM backend factory honours config + env precedence (mirrors
    edit-selection),
  * the system prompt enforces short, speakable answers,
  * the pipeline survives every error path (empty transcript, LLM down, TTS
    unavailable, STT crash) and never raises across its public surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from linux_whispr.ai.base import RefinementResult
from linux_whispr.ai.prompts.conversation import (
    CONVERSATION_SYSTEM_PROMPT,
    build_conversation_prompt,
)
from linux_whispr.config import AppConfig
from linux_whispr.conversation import (
    DEFAULT_CONVERSATION_API_KEY,
    DEFAULT_CONVERSATION_BASE_URL,
    DEFAULT_CONVERSATION_MODEL,
    ConversationPipeline,
    ConversationResult,
    build_conversation_llm_backend,
    resolve_conversation_api_key,
)
from linux_whispr.output.tts import NullBackend
from linux_whispr.stt.base import TranscriptionResult


def _pipeline(
    *,
    backend: object | None = None,
    tts: object | None = None,
    dictation: object | None = None,
) -> ConversationPipeline:
    """Build a pipeline with all heavy deps mocked by default."""
    return ConversationPipeline(
        AppConfig(),
        backend=backend or MagicMock(),  # type: ignore[arg-type]
        tts=tts or NullBackend(),
        dictation=dictation or MagicMock(),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# LLM backend factory + key resolution
# --------------------------------------------------------------------------- #
class TestConversationBackendFactory:
    def test_defaults_to_local_litellm(self) -> None:
        backend = build_conversation_llm_backend(AppConfig())
        assert backend._model == DEFAULT_CONVERSATION_MODEL
        assert backend._base_url == DEFAULT_CONVERSATION_BASE_URL

    def test_config_overrides_endpoint_and_model(self) -> None:
        config = AppConfig()
        config.ai.base_url = "http://example:9000/v1"
        config.ai.model = "another-model"
        backend = build_conversation_llm_backend(config)
        assert backend._base_url == "http://example:9000/v1"
        assert backend._model == "another-model"

    def test_api_key_env_takes_precedence(self, monkeypatch) -> None:
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        assert resolve_conversation_api_key(AppConfig()) == "sk-from-env"

    def test_api_key_falls_back_to_openai_env(self, monkeypatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai")
        assert resolve_conversation_api_key(AppConfig()) == "sk-from-openai"

    def test_api_key_falls_back_to_config(self, monkeypatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AppConfig()
        config.ai.api_key = "sk-from-config"
        assert resolve_conversation_api_key(config) == "sk-from-config"

    def test_api_key_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert (
            resolve_conversation_api_key(AppConfig()) == DEFAULT_CONVERSATION_API_KEY
        )


# --------------------------------------------------------------------------- #
# Conversation prompt
# --------------------------------------------------------------------------- #
class TestConversationPrompt:
    def test_system_prompt_demands_short_answers(self) -> None:
        assert "KURZ" in CONVERSATION_SYSTEM_PROMPT
        # 2-3 sentence guidance survives in the prompt body.
        assert "2" in CONVERSATION_SYSTEM_PROMPT or "drei" in CONVERSATION_SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_markdown(self) -> None:
        # TTS pronounces markdown noise — the prompt must forbid it.
        lowered = CONVERSATION_SYSTEM_PROMPT.lower()
        assert "markdown" in lowered or "codeblo" in lowered

    def test_user_prompt_trims_whitespace(self) -> None:
        assert build_conversation_prompt("   hallo welt  ") == "hallo welt"

    def test_user_prompt_preserves_internal_spacing(self) -> None:
        assert build_conversation_prompt("was ist 7 mal 8?") == "was ist 7 mal 8?"


# --------------------------------------------------------------------------- #
# ConversationPipeline.process — text-only path
# --------------------------------------------------------------------------- #
class TestConversationProcess:
    def test_empty_transcript_does_not_call_llm(self) -> None:
        backend = MagicMock()
        result = _pipeline(backend=backend).process("   ")
        assert result.ok is False
        assert result.error == "no transcript"
        backend.generate.assert_not_called()

    def test_happy_path_returns_spoken_reply(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(
            text="Wien.", model="cerebras-qwen", tokens_used=12
        )
        tts = MagicMock()
        tts.speak.return_value = True
        result = _pipeline(backend=backend, tts=tts).process(
            "Was ist die Hauptstadt von Oesterreich?"
        )
        assert result == ConversationResult(
            transcript="Was ist die Hauptstadt von Oesterreich?",
            reply="Wien.",
            spoken=True,
            ok=True,
            error="",
        )
        tts.speak.assert_called_once_with("Wien.")

    def test_uses_strict_system_prompt(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="Wien.")
        _pipeline(backend=backend).process("eine Frage")
        _, kwargs = backend.generate.call_args
        assert kwargs["system_prompt"] == CONVERSATION_SYSTEM_PROMPT
        # The user prompt is just the transcript (system prompt does the framing).
        assert kwargs["user_prompt"] == "eine Frage"

    def test_llm_exception_is_caught(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ConnectionError("connection refused")
        result = _pipeline(backend=backend).process("frag was")
        assert result.ok is False
        assert "LLM error" in result.error
        assert "connection refused" in result.error
        assert result.reply == ""

    def test_empty_llm_reply_is_failure(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="   ")
        result = _pipeline(backend=backend).process("frag was")
        assert result.ok is False
        assert "empty reply" in result.error

    def test_tts_failure_is_reported_but_reply_kept(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="Eine Antwort.")
        tts = MagicMock()
        tts.speak.return_value = False
        result = _pipeline(backend=backend, tts=tts).process("frag was")
        assert result.ok is False
        assert result.reply == "Eine Antwort."
        assert result.spoken is False
        assert "TTS" in result.error

    def test_tts_exception_does_not_propagate(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="Eine Antwort.")
        tts = MagicMock()
        tts.speak.side_effect = RuntimeError("audio device gone")
        result = _pipeline(backend=backend, tts=tts).process("frag was")
        assert result.ok is False
        assert "TTS error" in result.error
        assert "audio device gone" in result.error

    def test_reply_is_trimmed(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="  Wien.\n")
        result = _pipeline(backend=backend).process("eine Frage")
        assert result.reply == "Wien."


# --------------------------------------------------------------------------- #
# ConversationPipeline.run — full STT → LLM → TTS path
# --------------------------------------------------------------------------- #
class TestConversationRun:
    def test_no_audio_returns_friendly_error(self) -> None:
        result = _pipeline().run(b"")
        assert result.ok is False
        assert result.error == "no audio"

    def test_run_invokes_dictation_then_process(self) -> None:
        dictation = MagicMock()
        dictation.transcribe.return_value = TranscriptionResult(
            text="Was ist 7 mal 8?", language="de", confidence=0.95
        )
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="56.")
        tts = MagicMock()
        tts.speak.return_value = True
        result = _pipeline(backend=backend, tts=tts, dictation=dictation).run(
            b"FAKE-WAV"
        )
        assert result.ok is True
        assert result.transcript == "Was ist 7 mal 8?"
        assert result.reply == "56."
        assert result.spoken is True
        dictation.transcribe.assert_called_once_with(b"FAKE-WAV")

    def test_run_empty_transcript_skips_llm(self) -> None:
        dictation = MagicMock()
        dictation.transcribe.return_value = TranscriptionResult(text="")
        backend = MagicMock()
        result = _pipeline(backend=backend, dictation=dictation).run(b"FAKE-WAV")
        assert result.ok is False
        assert result.error == "no transcript"
        backend.generate.assert_not_called()

    def test_run_catches_stt_exception(self) -> None:
        dictation = MagicMock()
        dictation.transcribe.side_effect = RuntimeError("ctranslate2 oom")
        result = _pipeline(dictation=dictation).run(b"FAKE-WAV")
        assert result.ok is False
        assert "STT error" in result.error


# --------------------------------------------------------------------------- #
# Construction stays cheap (the SPEC-004 init-must-not-traceback check).
# --------------------------------------------------------------------------- #
class TestPipelineConstruction:
    def test_default_construction_does_not_raise(self) -> None:
        # Build with defaults: STT backend is unloaded, LLM is lazy, TTS is
        # NullBackend. Must not contact network or audio.
        pipeline = ConversationPipeline()
        assert pipeline.tts.name == "null"
        assert pipeline.dictation.is_loaded is False

    def test_tts_property_returns_injected_backend(self) -> None:
        tts = NullBackend()
        pipeline = ConversationPipeline(tts=tts)
        assert pipeline.tts is tts


# --------------------------------------------------------------------------- #
# ConversationResult dataclass shape
# --------------------------------------------------------------------------- #
class TestConversationResult:
    def test_default_error_is_empty_string(self) -> None:
        res = ConversationResult(
            transcript="hi", reply="ja", spoken=True, ok=True
        )
        assert res.error == ""

    def test_dataclass_is_frozen(self) -> None:
        res = ConversationResult(
            transcript="", reply="", spoken=False, ok=False, error="x"
        )
        try:
            res.transcript = "mutated"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("ConversationResult must be frozen")
