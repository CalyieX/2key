"""Dictation pipeline assembly (SPEC-002).

This module is the single source of truth for how the "2 Key" dictation path
is wired together: Strg+Super → record → STT(base) → inject-at-cursor.

Both the live application (``app.py``) and the offline smoke/check tooling
(``scripts/dictation_smoke.py``, the test-suite) build their STT backend and
text injector through the helpers here. That way what we verify headlessly is
exactly what runs on the desktop — we do not maintain a second, parallel
pipeline (see DECISIONS.md / SPEC-002 decision log).

The actual key-injection is intentionally NOT exercised here: it needs a
focused window and is a human-test. What this module gives us is:
  * a clean, GUI-free way to construct + load the STT backend from our config,
  * a faithful "app transcription path" (`DictationPipeline.transcribe`),
  * deterministic resolution of `injection.method = "auto"` to a concrete tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from linux_whispr.config import AppConfig
from linux_whispr.events import EventBus, event_bus as global_event_bus
from linux_whispr.output.injector import TextInjector
from linux_whispr.platform.detect import PlatformInfo, detect_platform
from linux_whispr.stt.base import STTBackend, TranscriptionResult
from linux_whispr.stt.faster_whisper import FasterWhisperBackend

logger = logging.getLogger(__name__)

# The config value that means "do not simulate a paste, only fill the clipboard".
CLIPBOARD_ONLY_METHOD = "clipboard-only"

# Concrete paste tools the injector knows how to drive.
KNOWN_INJECTION_TOOLS = ("xdotool", "wtype", "ydotool")


def build_stt_backend(config: AppConfig) -> STTBackend:
    """Create the STT backend described by ``config``.

    This mirrors what the running app uses; ``app.py`` delegates here so there
    is exactly one place that decides which backend dictation speaks to.
    """
    backend_name = config.stt.backend

    if backend_name == "faster-whisper":
        return FasterWhisperBackend(
            model_name=config.stt.model,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
        )

    if backend_name == "openai":
        import os

        from linux_whispr.stt.openai_api import OpenAIWhisperBackend

        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAIWhisperBackend(api_key=api_key)

    if backend_name == "groq":
        import os

        from linux_whispr.stt.groq_api import GroqWhisperBackend

        api_key = os.environ.get("GROQ_API_KEY", "")
        return GroqWhisperBackend(api_key=api_key)

    logger.warning(
        "Unknown STT backend '%s', falling back to faster-whisper", backend_name
    )
    return FasterWhisperBackend(model_name=config.stt.model)


@dataclass(frozen=True)
class InjectionMethodResolution:
    """Outcome of resolving the configured ``injection.method``.

    Attributes:
        requested: The raw value from config (e.g. "auto").
        resolved: The concrete tool that will be used, or None when no paste
            tool is available / clipboard-only was requested.
        clipboard_only: True when text is only copied, no paste is simulated.
        available: True when injection at the cursor is actually possible.
        reason: Human-readable explanation, logged + shown in the smoke output.
    """

    requested: str
    resolved: str | None
    clipboard_only: bool
    available: bool
    reason: str


def resolve_injection_method(
    method: str, platform: PlatformInfo
) -> InjectionMethodResolution:
    """Resolve the configured injection method against the detected platform.

    Handles the three shapes the config allows:
      * "auto"          → pick the best tool the platform offers,
      * "clipboard-only"→ never paste, just fill the clipboard,
      * an explicit tool→ honour it, but report if it is not installed.
    """
    normalized = method.strip().lower()

    if normalized == CLIPBOARD_ONLY_METHOD:
        return InjectionMethodResolution(
            requested=method,
            resolved=None,
            clipboard_only=True,
            available=False,
            reason="clipboard-only: text is copied, paste is left to the user",
        )

    if normalized == "auto":
        best = platform.best_injection_tool
        if best is None:
            return InjectionMethodResolution(
                requested=method,
                resolved=None,
                clipboard_only=False,
                available=False,
                reason=(
                    "auto found no paste tool "
                    "(install xdotool, wtype, or ydotool)"
                ),
            )
        return InjectionMethodResolution(
            requested=method,
            resolved=best,
            clipboard_only=False,
            available=True,
            reason=f"auto resolved to '{best}' on {platform.display_server.value}",
        )

    # An explicit tool was named.
    if normalized in KNOWN_INJECTION_TOOLS:
        installed = _tool_is_installed(normalized, platform)
        if installed:
            return InjectionMethodResolution(
                requested=method,
                resolved=normalized,
                clipboard_only=False,
                available=True,
                reason=f"explicit tool '{normalized}' is installed",
            )
        return InjectionMethodResolution(
            requested=method,
            resolved=normalized,
            clipboard_only=False,
            available=False,
            reason=f"explicit tool '{normalized}' is configured but not installed",
        )

    return InjectionMethodResolution(
        requested=method,
        resolved=None,
        clipboard_only=False,
        available=False,
        reason=f"unknown injection method '{method}'",
    )


def _tool_is_installed(tool: str, platform: PlatformInfo) -> bool:
    """Whether a named paste tool was detected on this platform."""
    flags = {
        "xdotool": platform.has_xdotool,
        "wtype": platform.has_wtype,
        "ydotool": platform.has_ydotool,
    }
    return flags.get(tool, False)


def build_text_injector(
    config: AppConfig,
    platform: PlatformInfo,
    event_bus: EventBus,
) -> TextInjector:
    """Create the cursor TextInjector from config — the same one the app uses."""
    return TextInjector(
        event_bus=event_bus,
        platform=platform,
        preserve_clipboard=config.injection.preserve_clipboard,
        restore_delay=config.injection.clipboard_restore_delay,
        method=config.injection.method,
    )


class DictationPipeline:
    """Headless, GUI-free view of the dictation audio→text→inject path.

    Construction is cheap and never touches audio hardware, the model files, or
    the network — that makes it safe for the "init must not traceback" check.
    The heavy model load happens lazily in :meth:`load` (or on first
    :meth:`transcribe`).
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        platform: PlatformInfo | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._platform = platform or detect_platform()
        self._event_bus = event_bus or global_event_bus
        self._stt: STTBackend = build_stt_backend(self._config)
        self._injection = resolve_injection_method(
            self._config.injection.method, self._platform
        )

    @property
    def injection(self) -> InjectionMethodResolution:
        """How the configured injection method resolved on this platform."""
        return self._injection

    @property
    def is_loaded(self) -> bool:
        """Whether the STT model has been loaded."""
        return self._stt.is_loaded

    def load(self) -> None:
        """Load the STT model if it is not already loaded."""
        if not self._stt.is_loaded:
            logger.info("Loading dictation STT backend...")
            self._stt.load()

    def transcribe(self, wav_bytes: bytes) -> TranscriptionResult:
        """Transcribe WAV audio through the app's STT path.

        Uses the configured language (``auto`` lets faster-whisper detect).
        Loads the model on first use. Empty/short audio yields an empty
        transcript rather than a fabricated one — the caller decides whether
        to inject (the live app skips injection on empty text).
        """
        if not wav_bytes:
            logger.warning("Empty audio buffer, nothing to transcribe")
            return TranscriptionResult(text="")

        self.load()
        return self._stt.transcribe(
            wav_bytes,
            language=self._config.stt.language,
        )

    def build_injector(self) -> TextInjector:
        """Build the TextInjector for this pipeline (used by the live app)."""
        return build_text_injector(self._config, self._platform, self._event_bus)
