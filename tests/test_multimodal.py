"""Tests for the multimodal pipeline (SPEC-005).

Covers:
  * vision-prompt-content builder shape (text first, then image_url blocks),
  * resolve_vision_api_key precedence (env > config > default),
  * build_vision_llm_backend (default model vs config-vision-model vs override),
  * MultimodalPipeline.ask — text-only, image+text, empty question, LLM error,
    empty answer, skipped unreadable file.

No real network: the LLM call goes through a fake backend (``MagicMock``) or a
stub OpenAI client so the OpenAI SDK is never instantiated. No real PDFs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from linux_whispr.ai.base import RefinementResult
from linux_whispr.ai.openai_llm import OpenAILLMBackend
from linux_whispr.ai.prompts.vision import (
    VISION_SYSTEM_PROMPT,
    build_vision_content,
)
from linux_whispr.config import AppConfig
from linux_whispr.input.file_context import FileContext
from linux_whispr.multimodal import (
    DEFAULT_VISION_API_KEY,
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_MODEL,
    AskResult,
    MultimodalPipeline,
    build_vision_llm_backend,
    call_multimodal,
    resolve_vision_api_key,
)


# --------------------------------------------------------------------------- #
# Vision prompt content builder
# --------------------------------------------------------------------------- #
class TestBuildVisionContent:
    def test_text_only_no_files(self) -> None:
        content = build_vision_content("Was passiert?", [])
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "Was passiert?" in content[0]["text"]

    def test_one_image_appended_after_text(self) -> None:
        img = FileContext(path="/x.png", kind="image", mime="image/png", image_b64="ABCD")
        content = build_vision_content("Was ist drauf?", [img])

        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,ABCD"

    def test_multiple_images(self) -> None:
        a = FileContext(path="/a.png", kind="image", mime="image/png", image_b64="AAA")
        b = FileContext(path="/b.jpg", kind="image", mime="image/jpeg", image_b64="BBB")
        content = build_vision_content("Vergleiche", [a, b])

        urls = [c["image_url"]["url"] for c in content if c.get("type") == "image_url"]
        assert urls == ["data:image/png;base64,AAA", "data:image/jpeg;base64,BBB"]

    def test_text_file_embedded_in_text_block(self) -> None:
        txt = FileContext(
            path="/notes.txt", kind="text", mime="text/plain", text="Geheim 42"
        )
        content = build_vision_content("Was steht drin?", [txt])

        assert len(content) == 1  # no image_url
        assert "Geheim 42" in content[0]["text"]
        assert "notes.txt" in content[0]["text"]

    def test_pdf_text_embedded_in_text_block(self) -> None:
        pdf = FileContext(
            path="/r.pdf", kind="pdf-text", mime="application/pdf", text="Kapitel 1"
        )
        content = build_vision_content("Fass zusammen", [pdf])
        assert "PDF" in content[0]["text"]
        assert "Kapitel 1" in content[0]["text"]

    def test_long_text_is_truncated(self) -> None:
        long_text = "x" * 50_000
        txt = FileContext(path="/big.txt", kind="text", mime="text/plain", text=long_text)
        content = build_vision_content("?", [txt])
        # 16000 budget + framing; not the whole 50k.
        assert len(content[0]["text"]) < 20_000
        assert "abgeschnitten" in content[0]["text"]


# --------------------------------------------------------------------------- #
# API-key resolution
# --------------------------------------------------------------------------- #
class TestResolveVisionApiKey:
    def test_env_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LITELLM_API_KEY", "from-env")
        cfg = AppConfig()
        cfg.ai.api_key = "from-config"
        assert resolve_vision_api_key(cfg) == "from-env"

    def test_openai_env_used_when_litellm_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-openai-env")
        cfg = AppConfig()
        cfg.ai.api_key = "from-config"
        assert resolve_vision_api_key(cfg) == "from-openai-env"

    def test_config_used_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = AppConfig()
        cfg.ai.api_key = "from-config"
        assert resolve_vision_api_key(cfg) == "from-config"

    def test_default_when_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_vision_api_key(AppConfig()) == DEFAULT_VISION_API_KEY


# --------------------------------------------------------------------------- #
# Backend factory
# --------------------------------------------------------------------------- #
class TestBuildVisionLlmBackend:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = build_vision_llm_backend(AppConfig())
        assert backend._model == DEFAULT_VISION_MODEL  # noqa: SLF001
        assert backend._base_url == DEFAULT_VISION_BASE_URL  # noqa: SLF001

    def test_config_vision_model_honoured(self) -> None:
        cfg = AppConfig()
        cfg.ai.model = "vision-nemotron"
        backend = build_vision_llm_backend(cfg)
        assert backend._model == "vision-nemotron"  # noqa: SLF001

    def test_config_non_vision_model_ignored(self) -> None:
        # cerebras-qwen is text-only — must not be picked just because it's set
        # for the refinement/edit path.
        cfg = AppConfig()
        cfg.ai.model = "cerebras-qwen"
        backend = build_vision_llm_backend(cfg)
        assert backend._model == DEFAULT_VISION_MODEL  # noqa: SLF001

    def test_model_override_wins(self) -> None:
        cfg = AppConfig()
        cfg.ai.model = "vision-gemma3"
        backend = build_vision_llm_backend(cfg, model_override="vision-nemotron")
        assert backend._model == "vision-nemotron"  # noqa: SLF001

    def test_base_url_from_config(self) -> None:
        cfg = AppConfig()
        cfg.ai.base_url = "http://elsewhere:4000/v1"
        backend = build_vision_llm_backend(cfg)
        assert backend._base_url == "http://elsewhere:4000/v1"  # noqa: SLF001


# --------------------------------------------------------------------------- #
# call_multimodal — multimodal POST shape, no real network
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Minimal stand-in for an OpenAI client — captures the last call."""

    def __init__(self, answer: str = "the answer") -> None:
        self.last_kwargs: dict = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self._answer = answer

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._answer))]
        )


class TestCallMultimodal:
    def test_sends_content_array_as_user_message(self) -> None:
        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        fake = _FakeClient(answer="OK!")
        backend._client = fake  # noqa: SLF001 — bypass network load

        content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
        out = call_multimodal(backend, "sys", content)

        assert out == "OK!"
        assert fake.last_kwargs["model"] == "vision-gemma3"
        messages = fake.last_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == content

    def test_skips_system_when_empty(self) -> None:
        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        fake = _FakeClient()
        backend._client = fake  # noqa: SLF001
        call_multimodal(backend, "", [{"type": "text", "text": "x"}])
        assert fake.last_kwargs["messages"][0]["role"] == "user"

    def test_none_message_content_returns_empty_string(self) -> None:
        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        backend._client = _FakeClient(answer=None)  # type: ignore[arg-type]  # noqa: SLF001
        out = call_multimodal(backend, "sys", [{"type": "text", "text": "x"}])
        assert out == ""


# --------------------------------------------------------------------------- #
# MultimodalPipeline.ask
# --------------------------------------------------------------------------- #
def _backend_returning(text: str) -> MagicMock:
    """A MagicMock backend whose ``generate`` returns ``RefinementResult(text)``."""
    backend = MagicMock(spec=OpenAILLMBackend)
    backend.generate.return_value = RefinementResult(text=text, model="m", tokens_used=1)
    backend._client = None  # noqa: SLF001
    backend._model = "vision-gemma3"  # noqa: SLF001
    return backend


class TestAskTextOnly:
    def test_happy_path_no_files(self) -> None:
        backend = _backend_returning("Hallo!")
        pipe = MultimodalPipeline(AppConfig(), backend=backend)

        result = pipe.ask("Wer bist du?")

        assert result.ok is True
        assert result.answer == "Hallo!"
        assert result.used_files == []
        backend.generate.assert_called_once()
        args, kwargs = backend.generate.call_args
        assert kwargs["system_prompt"] == VISION_SYSTEM_PROMPT
        assert kwargs["user_prompt"] == "Wer bist du?"

    def test_empty_question_short_circuits(self) -> None:
        backend = _backend_returning("never")
        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        result = pipe.ask("   ")
        assert result.ok is False
        assert "no question" in result.error
        backend.generate.assert_not_called()

    def test_llm_returns_empty_text(self) -> None:
        backend = _backend_returning("   ")
        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        result = pipe.ask("hallo?")
        assert result.ok is False
        assert "empty" in result.error.lower()

    def test_llm_raises_is_caught(self) -> None:
        backend = MagicMock(spec=OpenAILLMBackend)
        backend.generate.side_effect = RuntimeError("boom")
        backend._client = None  # noqa: SLF001
        backend._model = "m"  # noqa: SLF001
        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        result = pipe.ask("?")
        assert result.ok is False
        assert "LLM error" in result.error
        assert "boom" in result.error


class TestAskMultimodal:
    def test_image_path_uses_chat_completions_with_content_array(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "shot.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nbinary-pixels")

        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        fake = _FakeClient(answer="Ein PNG.")
        backend._client = fake  # noqa: SLF001

        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        result = pipe.ask("Was ist auf dem Bild?", [p])

        assert result.ok is True
        assert result.answer == "Ein PNG."
        assert str(p) in result.used_files

        messages = fake.last_kwargs["messages"]
        user_content = messages[-1]["content"]
        assert isinstance(user_content, list)
        # Last block must be the image_url block
        assert user_content[-1]["type"] == "image_url"
        assert user_content[-1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_accepts_filecontext_directly(self) -> None:
        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        fake = _FakeClient(answer="ja")
        backend._client = fake  # noqa: SLF001
        pipe = MultimodalPipeline(AppConfig(), backend=backend)

        ctx = FileContext(path="/v.png", kind="image", mime="image/png", image_b64="ZZ")
        result = pipe.ask("?", [ctx])

        assert result.ok is True
        assert "/v.png" in result.used_files

    def test_skips_missing_files_silently(self, tmp_path: Path) -> None:
        backend = _backend_returning("text-only fallback")
        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        # No real file → coerce returns empty → text-only path.
        result = pipe.ask("hallo?", [tmp_path / "ghost.png"])

        assert result.ok is True
        assert result.used_files == []
        backend.generate.assert_called_once()  # text-only path

    def test_image_llm_raises_returns_error_result(self, tmp_path: Path) -> None:
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nbinary-pixels")

        class _BrokenClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kw: (_ for _ in ()).throw(RuntimeError("net dead"))
                    )
                )

        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        backend._client = _BrokenClient()  # noqa: SLF001
        pipe = MultimodalPipeline(AppConfig(), backend=backend)

        result = pipe.ask("?", [p])
        assert result.ok is False
        assert "net dead" in result.error
        assert str(p) in result.used_files


# --------------------------------------------------------------------------- #
# AskResult shape
# --------------------------------------------------------------------------- #
class TestAskResult:
    def test_defaults(self) -> None:
        r = AskResult(answer="x")
        assert r.ok is False
        assert r.error == ""
        assert r.used_files == []


# --------------------------------------------------------------------------- #
# Integration with load_file
# --------------------------------------------------------------------------- #
class TestIntegrationWithLoader:
    def test_pipeline_loads_then_passes_to_backend(self, tmp_path: Path) -> None:
        # txt file → text-only path (no image) — backend.generate is called
        # with a user prompt that embeds the file content.
        p = tmp_path / "notes.txt"
        p.write_text("der Code ist auf Zeile 42", encoding="utf-8")

        backend = OpenAILLMBackend(api_key="k", model="vision-gemma3")
        fake = _FakeClient(answer="Zeile 42")
        backend._client = fake  # noqa: SLF001

        pipe = MultimodalPipeline(AppConfig(), backend=backend)
        result = pipe.ask("Wo ist der Code?", [p])

        assert result.ok is True
        assert str(p) in result.used_files
        # Text file went through multimodal path (no image but still content list)
        user_content = fake.last_kwargs["messages"][-1]["content"]
        assert isinstance(user_content, list)
        text_block = next(c for c in user_content if c["type"] == "text")
        assert "Zeile 42" in text_block["text"]
