"""Pill UI for the 2 Key mode indicator (SPEC-006).

Two concrete implementations of the ``PillUI`` protocol:

  * :class:`NullPill` — the safe default. Logs at debug-level, never touches
    a display server. Used in headless environments, tests, CI, and as the
    fallback when GTK can't be imported.
  * :class:`GtkPill` — thin skeleton around GTK4. The ``gi.repository``
    import happens lazily inside ``_load_gtk()`` so importing this module
    *never* drags GTK into a headless interpreter. The actual window
    rendering is intentionally a stub: a real, pretty pill window with
    animation + Wayland layer-shell is **Human-Test polish** (Calyie on
    the wonderland desktop) and is documented as such in SPEC-006.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "GtkPill",
    "NullPill",
    "PillUI",
]


@runtime_checkable
class PillUI(Protocol):
    """Structural protocol — anything matching the four calls is a pill."""

    def show(self, mode_name: str) -> None: ...  # noqa: D401, E704
    def hide(self) -> None: ...  # noqa: D401, E704
    def flash(
        self, msg: str, *, duration_s: float = 1.0
    ) -> None: ...  # noqa: D401, E704
    def available(self) -> bool: ...  # noqa: D401, E704


class NullPill:
    """No-op pill — does nothing visible, logs for traceability."""

    def show(self, mode_name: str) -> None:
        logger.debug("NullPill.show(%r)", mode_name)

    def hide(self) -> None:
        logger.debug("NullPill.hide()")

    def flash(self, msg: str, *, duration_s: float = 1.0) -> None:
        logger.debug("NullPill.flash(%r, duration_s=%s)", msg, duration_s)

    def available(self) -> bool:
        return False


class GtkPill:
    """Lazy-import GTK4 pill. Falls back to NullPill behaviour if GTK absent.

    Construction is intentionally cheap and side-effect-free: no GTK call
    runs until :meth:`show` / :meth:`flash` is invoked. That keeps the
    headless test-suite + ``import linux_whispr`` clean of GUI imports and
    matches the *one pipeline, GUI-free constructor* pattern used in
    ``dictation.py`` and ``edit_selection.py``.

    The real pretty rendering (rounded pill, Wayland layer-shell overlay,
    fade-in/out) is left as a **Human-Polish TODO**; the headless contract
    enforced by the tests is:

      * importing this class never imports gi.repository,
      * the four protocol methods are present and don't raise when GTK
        is unavailable.
    """

    def __init__(
        self,
        *,
        position: str = "top-right",
        margin_px: int = 24,
    ) -> None:
        self._position = position
        self._margin = margin_px
        self._gtk = None
        self._window = None
        self._import_error: Exception | None = None

    def _load_gtk(self):  # type: ignore[no-untyped-def]
        """Lazy GTK import. None means GTK not usable on this machine."""
        if self._gtk is not None:
            return self._gtk
        if self._import_error is not None:
            return None
        try:
            import gi  # noqa: PLC0415  (lazy on purpose)

            gi.require_version("Gtk", "4.0")
            from gi.repository import Gtk  # noqa: PLC0415

            self._gtk = Gtk
            return Gtk
        except Exception as exc:  # noqa: BLE001
            self._import_error = exc
            logger.info("GTK4 not available, pill falls back to no-op: %s", exc)
            return None

    def available(self) -> bool:
        return self._load_gtk() is not None

    def show(self, mode_name: str) -> None:
        gtk = self._load_gtk()
        if gtk is None:
            logger.debug("GtkPill.show(%r) — GTK unavailable, no-op", mode_name)
            return
        # Human-polish TODO: real rounded pill, layer-shell, fade-in.
        logger.info("GtkPill.show(%r)", mode_name)

    def hide(self) -> None:
        gtk = self._load_gtk()
        if gtk is None:
            logger.debug("GtkPill.hide() — GTK unavailable, no-op")
            return
        logger.info("GtkPill.hide()")

    def flash(self, msg: str, *, duration_s: float = 1.0) -> None:
        gtk = self._load_gtk()
        if gtk is None:
            logger.debug(
                "GtkPill.flash(%r, duration_s=%s) — GTK unavailable, no-op",
                msg,
                duration_s,
            )
            return
        logger.info("GtkPill.flash(%r, duration_s=%s)", msg, duration_s)
