"""X11 PRIMARY-selection capture and in-place replacement (SPEC-003).

This is the "what is highlighted right now" half of the edit-selection mode:
the user selects text anywhere on screen, and we need to (a) read that text and
(b) replace it with the LLM's edited version.

On X11 the highlighted text lives in the PRIMARY selection (distinct from the
CLIPBOARD that Ctrl+C/Ctrl+V use). We read it with ``xclip``/``xsel`` and replace
it by typing the new text over the still-highlighted region with ``xdotool`` —
typing while a selection is active overwrites it in every standard text widget.

Reading is fully testable headlessly (mock the tool call). The *replace* path
drives the real keyboard, so its end-to-end behaviour is a human-test; here we
implement and document the mechanic and keep it crash-safe.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linux_whispr.platform.detect import PlatformInfo

logger = logging.getLogger(__name__)

# How long we wait for a selection tool before giving up. Reads are local and
# instant; a hang here means the X server is wedged, so fail fast.
_TOOL_TIMEOUT = 5

# xdotool types a touch slower than instant to stay reliable across widgets.
_TYPE_DELAY_MS = 12


def display_is_available() -> bool:
    """Whether a graphical session exists to hold a selection at all.

    No display → no PRIMARY selection. Callers use this to skip cleanly on a
    headless machine instead of shelling out to a tool that will only error.
    """
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_primary_tool(platform: PlatformInfo) -> str | None:
    """Pick the command-line tool that can read the PRIMARY selection.

    Both xclip and xsel speak the PRIMARY selection on X11; we prefer xclip
    because it is the more commonly installed of the two. Returns None when
    neither is present (the caller then reports "no selection tool").
    """
    if platform.has_xclip:
        return "xclip"
    if platform.has_xsel:
        return "xsel"
    return None


def _read_args(tool: str) -> list[str]:
    """Argv that prints the current PRIMARY selection to stdout."""
    if tool == "xclip":
        return ["xclip", "-selection", "primary", "-o"]
    return ["xsel", "--primary", "--output"]


def _write_args(tool: str) -> list[str]:
    """Argv that loads stdin into the PRIMARY selection (used by the smoke test)."""
    if tool == "xclip":
        return ["xclip", "-selection", "primary"]
    return ["xsel", "--primary", "--input"]


class SelectionCapture:
    """Read and replace the X11 PRIMARY selection.

    Construction never touches the X server; it only resolves which tool to use.
    Every method tolerates a missing display or tool by returning a falsy result
    and logging — selecting nothing must never crash the assistant.
    """

    def __init__(
        self,
        platform: PlatformInfo,
        *,
        display_available: bool | None = None,
    ) -> None:
        self._platform = platform
        self._tool = resolve_primary_tool(platform)
        # Allow tests to force the display state; default to real detection.
        self._display = (
            display_is_available() if display_available is None else display_available
        )

    @property
    def tool(self) -> str | None:
        """The selection tool that will be used, or None if none is available."""
        return self._tool

    @property
    def available(self) -> bool:
        """True when a selection can actually be read (display + tool present)."""
        return self._display and self._tool is not None

    def unavailable_reason(self) -> str:
        """Human-readable explanation for why capture is not possible (if so)."""
        if not self._display:
            return "no graphical display (DISPLAY/WAYLAND_DISPLAY unset)"
        if self._tool is None:
            return "no PRIMARY-selection tool found (install xclip or xsel)"
        return ""

    def read(self) -> str | None:
        """Return the currently highlighted text.

        Returns:
            * the selected text (possibly with surrounding whitespace),
            * ``""`` when nothing is selected (a normal, non-error state),
            * ``None`` when capture is impossible (no display/tool) or the tool
              call failed — distinct from "empty" so the caller can warn.
        """
        if not self.available:
            logger.warning("Selection capture unavailable: %s", self.unavailable_reason())
            return None

        assert self._tool is not None
        try:
            result = subprocess.run(
                _read_args(self._tool),
                capture_output=True,
                text=True,
                timeout=_TOOL_TIMEOUT,
            )
        except FileNotFoundError:
            logger.error("Selection tool '%s' not found on PATH", self._tool)
            return None
        except subprocess.TimeoutExpired:
            logger.error("Selection read with '%s' timed out", self._tool)
            return None

        # An empty PRIMARY selection makes xclip exit non-zero — that is "nothing
        # selected", not a failure, so normalise it to an empty string.
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.debug("Selection read returned %d (%s)", result.returncode, stderr)
            return ""

        return result.stdout

    def set_primary(self, text: str) -> bool:
        """Load ``text`` into the PRIMARY selection (test/setup helper).

        Used by ``scripts/selection_smoke.py`` to plant a known selection so the
        read path can be verified headlessly. Not part of the live edit flow.
        """
        if not self.available:
            return False

        assert self._tool is not None
        try:
            result = subprocess.run(
                _write_args(self._tool),
                input=text,
                text=True,
                capture_output=True,
                timeout=_TOOL_TIMEOUT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.exception("Failed to set PRIMARY selection via '%s'", self._tool)
            return False

        return result.returncode == 0

    def replace(self, text: str) -> bool:
        """Replace the highlighted selection with ``text`` by typing over it.

        Mechanic: while text is selected, typing replaces it in every standard
        text widget, so we drive ``xdotool type``. This needs the original window
        focused and is therefore a human-test end-to-end (see SPEC-003); here we
        keep it safe and report success/failure honestly.

        Returns False (without typing) when no display or xdotool is available,
        or when ``text`` is empty — never raises.
        """
        if not text:
            logger.warning("Refusing to replace selection with empty text")
            return False
        if not self._display:
            logger.warning("Cannot replace selection: %s", self.unavailable_reason())
            return False
        if not self._platform.has_xdotool:
            logger.error("Cannot replace selection: xdotool not installed")
            return False

        try:
            result = subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", str(_TYPE_DELAY_MS), text],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            logger.error("xdotool not found on PATH")
            return False
        except subprocess.TimeoutExpired:
            logger.error("xdotool type timed out replacing selection")
            return False

        if result.returncode != 0:
            logger.error("xdotool type failed: %s", result.stderr)
            return False
        return True
