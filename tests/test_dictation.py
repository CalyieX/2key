"""Tests for the dictation pipeline assembly (SPEC-002).

Covers the auto-checkable acceptance criteria without touching audio hardware,
the model files, or the network:
  * the pipeline initializes from config without a traceback,
  * injection.method resolution (auto / explicit / clipboard-only / missing),
  * the STT backend factory honours config,
  * empty-audio edge case yields no fabricated text.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from linux_whispr.config import AppConfig
from linux_whispr.dictation import (
    DictationPipeline,
    InjectionMethodResolution,
    build_stt_backend,
    build_text_injector,
    resolve_injection_method,
)
from linux_whispr.events import EventBus
from linux_whispr.platform.detect import DisplayServer, PlatformInfo
from linux_whispr.stt.faster_whisper import FasterWhisperBackend


def _x11_platform(**overrides: object) -> PlatformInfo:
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


def _wayland_platform(**overrides: object) -> PlatformInfo:
    defaults = dict(
        display_server=DisplayServer.WAYLAND,
        desktop=MagicMock(),
        has_xdotool=False,
        has_wtype=True,
        has_ydotool=False,
        has_xclip=False,
        has_xsel=False,
        has_wl_clipboard=True,
    )
    defaults.update(overrides)
    return PlatformInfo(**defaults)  # type: ignore[arg-type]


def _no_tools_platform() -> PlatformInfo:
    return PlatformInfo(
        display_server=DisplayServer.X11,
        desktop=MagicMock(),
        has_xdotool=False,
        has_wtype=False,
        has_ydotool=False,
        has_xclip=False,
        has_xsel=False,
        has_wl_clipboard=False,
    )


class TestBuildSTTBackend:
    def test_default_config_builds_faster_whisper(self) -> None:
        backend = build_stt_backend(AppConfig())
        assert isinstance(backend, FasterWhisperBackend)
        assert not backend.is_loaded

    def test_unknown_backend_falls_back_to_faster_whisper(self) -> None:
        config = AppConfig()
        config.stt.backend = "totally-unknown-backend"
        backend = build_stt_backend(config)
        assert isinstance(backend, FasterWhisperBackend)


class TestResolveInjectionMethod:
    def test_auto_resolves_to_xdotool_on_x11(self) -> None:
        resolution = resolve_injection_method("auto", _x11_platform())
        assert resolution.resolved == "xdotool"
        assert resolution.available is True
        assert resolution.clipboard_only is False

    def test_auto_resolves_to_wtype_on_wayland(self) -> None:
        resolution = resolve_injection_method("auto", _wayland_platform())
        assert resolution.resolved == "wtype"
        assert resolution.available is True

    def test_auto_with_no_tools_is_unavailable(self) -> None:
        resolution = resolve_injection_method("auto", _no_tools_platform())
        assert resolution.resolved is None
        assert resolution.available is False
        assert "no paste tool" in resolution.reason

    def test_clipboard_only_is_intentional_not_failure(self) -> None:
        resolution = resolve_injection_method("clipboard-only", _x11_platform())
        assert resolution.clipboard_only is True
        assert resolution.resolved is None
        assert resolution.available is False

    def test_explicit_installed_tool(self) -> None:
        resolution = resolve_injection_method("xdotool", _x11_platform())
        assert resolution.resolved == "xdotool"
        assert resolution.available is True

    def test_explicit_missing_tool_reports_unavailable(self) -> None:
        resolution = resolve_injection_method("ydotool", _x11_platform())
        assert resolution.resolved == "ydotool"
        assert resolution.available is False
        assert "not installed" in resolution.reason

    def test_unknown_method_is_unavailable(self) -> None:
        resolution = resolve_injection_method("magic", _x11_platform())
        assert resolution.available is False
        assert isinstance(resolution, InjectionMethodResolution)


class TestBuildTextInjector:
    def test_injector_uses_config_method(self) -> None:
        config = AppConfig()
        config.injection.method = "auto"
        injector = build_text_injector(config, _x11_platform(), EventBus())
        # auto on x11 with xdotool present
        assert injector._method == "xdotool"


class TestDictationPipeline:
    def test_init_from_config_no_traceback(self) -> None:
        pipeline = DictationPipeline(AppConfig(), platform=_x11_platform())
        assert pipeline.is_loaded is False
        assert pipeline.injection.resolved == "xdotool"

    def test_init_with_default_config(self) -> None:
        # No config passed → uses AppConfig() defaults; must not raise.
        pipeline = DictationPipeline(platform=_wayland_platform())
        assert pipeline.injection.resolved == "wtype"

    def test_empty_audio_returns_empty_text_without_loading(self) -> None:
        pipeline = DictationPipeline(AppConfig(), platform=_x11_platform())
        result = pipeline.transcribe(b"")
        assert result.text == ""
        # Empty input must not have triggered a model load.
        assert pipeline.is_loaded is False

    def test_transcribe_delegates_to_backend(self) -> None:
        pipeline = DictationPipeline(AppConfig(), platform=_x11_platform())

        fake_backend = MagicMock()
        fake_backend.is_loaded = False
        fake_backend.transcribe.return_value = "RESULT"
        pipeline._stt = fake_backend

        out = pipeline.transcribe(b"RIFFfake-wav-bytes")

        fake_backend.load.assert_called_once()
        fake_backend.transcribe.assert_called_once()
        # language from config is forwarded
        _, kwargs = fake_backend.transcribe.call_args
        assert kwargs["language"] == AppConfig().stt.language
        assert out == "RESULT"

    def test_build_injector_returns_text_injector(self) -> None:
        pipeline = DictationPipeline(AppConfig(), platform=_x11_platform())
        injector = pipeline.build_injector()
        assert injector._method == "xdotool"
