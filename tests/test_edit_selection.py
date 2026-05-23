"""Tests for the edit-selection mode (SPEC-003).

Covers the auto-checkable acceptance criteria without touching the X server or
the network:
  * PRIMARY-selection read path (mocked tool calls) incl. empty/no-display edges,
  * the replace mechanic refuses unsafe states and drives xdotool otherwise,
  * the LLM backend factory honours config + env precedence,
  * the edit pipeline handles empty selection/instruction, LLM errors, and the
    happy path — all without raising.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from linux_whispr.ai.base import RefinementResult
from linux_whispr.ai.prompts.edit import EDIT_SYSTEM_PROMPT, build_edit_prompt
from linux_whispr.config import AppConfig
from linux_whispr.edit_selection import (
    DEFAULT_EDIT_API_KEY,
    DEFAULT_EDIT_BASE_URL,
    DEFAULT_EDIT_MODEL,
    EditPipeline,
    EditResult,
    build_edit_llm_backend,
    resolve_edit_api_key,
)
from linux_whispr.output import selection as selection_mod
from linux_whispr.output.selection import SelectionCapture, resolve_primary_tool
from linux_whispr.platform.detect import DisplayServer, PlatformInfo


def _platform(**overrides: object) -> PlatformInfo:
    defaults = dict(
        display_server=DisplayServer.X11,
        desktop=MagicMock(),
        has_xdotool=True,
        has_wtype=False,
        has_ydotool=False,
        has_xclip=True,
        has_xsel=False,
        has_wl_clipboard=False,
    )
    defaults.update(overrides)
    return PlatformInfo(**defaults)  # type: ignore[arg-type]


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> object:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------------- #
# selection.py — tool resolution
# --------------------------------------------------------------------------- #
class TestResolvePrimaryTool:
    def test_prefers_xclip(self) -> None:
        assert resolve_primary_tool(_platform(has_xclip=True, has_xsel=True)) == "xclip"

    def test_falls_back_to_xsel(self) -> None:
        assert resolve_primary_tool(_platform(has_xclip=False, has_xsel=True)) == "xsel"

    def test_none_when_no_tool(self) -> None:
        assert resolve_primary_tool(_platform(has_xclip=False, has_xsel=False)) is None


# --------------------------------------------------------------------------- #
# selection.py — availability
# --------------------------------------------------------------------------- #
class TestSelectionAvailability:
    def test_available_with_display_and_tool(self) -> None:
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.available is True
        assert cap.tool == "xclip"

    def test_unavailable_without_display(self) -> None:
        cap = SelectionCapture(_platform(), display_available=False)
        assert cap.available is False
        assert "display" in cap.unavailable_reason()

    def test_unavailable_without_tool(self) -> None:
        cap = SelectionCapture(
            _platform(has_xclip=False, has_xsel=False), display_available=True
        )
        assert cap.available is False
        assert "tool" in cap.unavailable_reason()


# --------------------------------------------------------------------------- #
# selection.py — read path
# --------------------------------------------------------------------------- #
class TestSelectionRead:
    def test_read_returns_selected_text(self, monkeypatch) -> None:
        monkeypatch.setattr(
            selection_mod.subprocess, "run", lambda *a, **k: _completed(0, "selected!")
        )
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.read() == "selected!"

    def test_empty_selection_returns_empty_string(self, monkeypatch) -> None:
        # xclip exits non-zero when nothing is selected — that is "empty", not error.
        monkeypatch.setattr(
            selection_mod.subprocess, "run", lambda *a, **k: _completed(1, "")
        )
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.read() == ""

    def test_read_none_when_unavailable(self) -> None:
        cap = SelectionCapture(_platform(), display_available=False)
        assert cap.read() is None

    def test_read_none_on_missing_tool_binary(self, monkeypatch) -> None:
        def _raise(*a, **k):
            raise FileNotFoundError

        monkeypatch.setattr(selection_mod.subprocess, "run", _raise)
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.read() is None

    def test_read_none_on_timeout(self, monkeypatch) -> None:
        def _raise(*a, **k):
            raise subprocess.TimeoutExpired(cmd="xclip", timeout=5)

        monkeypatch.setattr(selection_mod.subprocess, "run", _raise)
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.read() is None


# --------------------------------------------------------------------------- #
# selection.py — replace mechanic
# --------------------------------------------------------------------------- #
class TestSelectionReplace:
    def test_replace_refuses_empty_text(self) -> None:
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.replace("") is False

    def test_replace_refuses_without_display(self) -> None:
        cap = SelectionCapture(_platform(), display_available=False)
        assert cap.replace("hi") is False

    def test_replace_refuses_without_xdotool(self) -> None:
        cap = SelectionCapture(_platform(has_xdotool=False), display_available=True)
        assert cap.replace("hi") is False

    def test_replace_types_via_xdotool(self, monkeypatch) -> None:
        calls: list[list[str]] = []

        def _run(args, *a, **k):
            calls.append(args)
            return _completed(0)

        monkeypatch.setattr(selection_mod.subprocess, "run", _run)
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.replace("Hallo Welt") is True
        assert calls and calls[0][0] == "xdotool"
        assert calls[0][1] == "type"
        assert "Hallo Welt" in calls[0]

    def test_replace_reports_tool_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            selection_mod.subprocess, "run", lambda *a, **k: _completed(1, stderr="boom")
        )
        cap = SelectionCapture(_platform(), display_available=True)
        assert cap.replace("Hallo Welt") is False


# --------------------------------------------------------------------------- #
# edit_selection.py — backend factory + key resolution
# --------------------------------------------------------------------------- #
class TestEditBackendFactory:
    def test_defaults_to_local_litellm(self) -> None:
        backend = build_edit_llm_backend(AppConfig())
        assert backend._model == DEFAULT_EDIT_MODEL
        assert backend._base_url == DEFAULT_EDIT_BASE_URL

    def test_config_overrides_endpoint_and_model(self) -> None:
        config = AppConfig()
        config.ai.base_url = "http://example:9000/v1"
        config.ai.model = "some-other-model"
        backend = build_edit_llm_backend(config)
        assert backend._base_url == "http://example:9000/v1"
        assert backend._model == "some-other-model"

    def test_api_key_env_takes_precedence(self, monkeypatch) -> None:
        monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
        assert resolve_edit_api_key(AppConfig()) == "sk-from-env"

    def test_api_key_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_edit_api_key(AppConfig()) == DEFAULT_EDIT_API_KEY


# --------------------------------------------------------------------------- #
# ai/prompts/edit.py
# --------------------------------------------------------------------------- #
class TestEditPrompt:
    def test_system_prompt_demands_only_result(self) -> None:
        assert "AUSSCHLIESSLICH" in EDIT_SYSTEM_PROMPT

    def test_user_prompt_pairs_instruction_and_text(self) -> None:
        prompt = build_edit_prompt("Hallo wlt", "korrigiere")
        assert "korrigiere" in prompt
        assert "Hallo wlt" in prompt
        # instruction comes before the text
        assert prompt.index("korrigiere") < prompt.index("Hallo wlt")


# --------------------------------------------------------------------------- #
# edit_selection.py — pipeline
# --------------------------------------------------------------------------- #
def _pipeline(backend: object, selection: object | None = None) -> EditPipeline:
    return EditPipeline(
        AppConfig(),
        platform=_platform(),
        backend=backend,  # type: ignore[arg-type]
        selection=selection or MagicMock(),  # type: ignore[arg-type]
    )


class TestEditPipeline:
    def test_empty_selection_does_not_call_llm(self) -> None:
        backend = MagicMock()
        result = _pipeline(backend).edit("   ", "korrigiere")
        assert result.ok is False
        assert "no text selected" in result.error
        backend.generate.assert_not_called()

    def test_empty_instruction_does_not_call_llm(self) -> None:
        backend = MagicMock()
        result = _pipeline(backend).edit("Hallo wlt", "  ")
        assert result.ok is False
        assert "no instruction" in result.error
        backend.generate.assert_not_called()

    def test_happy_path_returns_stripped_text(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(
            text="  Hallo Welt  ", model="cerebras-qwen", tokens_used=7
        )
        result = _pipeline(backend).edit("Hallo wlt", "korrigiere")
        assert result == EditResult(text="Hallo Welt", ok=True)
        # strict system prompt is actually used
        _, kwargs = backend.generate.call_args
        assert kwargs["system_prompt"] == EDIT_SYSTEM_PROMPT

    def test_llm_exception_is_caught(self) -> None:
        backend = MagicMock()
        backend.generate.side_effect = ConnectionError("connection refused")
        result = _pipeline(backend).edit("Hallo wlt", "korrigiere")
        assert result.ok is False
        assert "LLM error" in result.error

    def test_empty_llm_response_is_failure(self) -> None:
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="   ")
        result = _pipeline(backend).edit("Hallo wlt", "korrigiere")
        assert result.ok is False
        assert "empty" in result.error

    def test_capture_and_replace_delegate_to_selection(self) -> None:
        selection = MagicMock()
        selection.read.return_value = "highlighted"
        selection.replace.return_value = True
        pipe = _pipeline(MagicMock(), selection=selection)
        assert pipe.capture_selection() == "highlighted"
        assert pipe.replace_selection("new") is True
        selection.replace.assert_called_once_with("new")


class TestEditPipelineRun:
    def test_run_no_display_returns_reason(self) -> None:
        selection = MagicMock()
        selection.read.return_value = None
        selection.unavailable_reason.return_value = "no graphical display"
        result = _pipeline(MagicMock(), selection=selection).run("korrigiere")
        assert result.ok is False
        assert "display" in result.error

    def test_run_empty_selection_returns_friendly_error(self) -> None:
        selection = MagicMock()
        selection.read.return_value = "   "
        result = _pipeline(MagicMock(), selection=selection).run("korrigiere")
        assert result.ok is False
        assert "no text selected" in result.error

    def test_run_happy_path_replaces_selection(self) -> None:
        selection = MagicMock()
        selection.read.return_value = "Hallo wlt"
        selection.replace.return_value = True
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="Hallo Welt")
        result = _pipeline(backend, selection=selection).run("korrigiere")
        assert result.ok is True
        assert result.text == "Hallo Welt"
        selection.replace.assert_called_once_with("Hallo Welt")

    def test_run_reports_replace_failure(self) -> None:
        selection = MagicMock()
        selection.read.return_value = "Hallo wlt"
        selection.replace.return_value = False
        backend = MagicMock()
        backend.generate.return_value = RefinementResult(text="Hallo Welt")
        result = _pipeline(backend, selection=selection).run("korrigiere")
        assert result.ok is False
        assert "replacing the selection failed" in result.error
