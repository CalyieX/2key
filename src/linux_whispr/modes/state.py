"""Mode state-machine for the 2 Key hotkey (SPEC-006).

Pure-Python, GTK-free, fully unit-testable. The state-machine cares about
two things only:

  * which mode is currently active (Dictate / Edit / Conversation / Multimodal),
  * whether the next ``Strg+Super`` press is a single tap (= fire the current
    mode's pipeline) or the second half of a double-tap (= cycle the mode).

The double-tap window defaults to 400 ms, matching GTK's
``gtk-double-click-time`` default. After a successful cycle the
``last_tap_ts`` is cleared so a triple-tap counts as "cycle, then action in
the new mode" rather than "cycle twice from one extra press".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "DEFAULT_DOUBLE_TAP_WINDOW_S",
    "HotkeyEventKind",
    "Mode",
    "ModeEvent",
    "ModeManager",
    "ModeState",
]


DEFAULT_DOUBLE_TAP_WINDOW_S: float = 0.4
"""GTK's gtk-double-click-time default, in seconds."""

_HISTORY_MAX = 16
"""Upper bound on the ring of historical modes — keeps the dataclass cheap."""


class Mode(Enum):
    """The four user-facing modes in their canonical cycle order."""

    DICTATE = "dictate"
    EDIT = "edit"
    CONVERSATION = "conversation"
    MULTIMODAL = "multimodal"

    @classmethod
    def cycle_order(cls) -> tuple["Mode", ...]:
        """Return the modes in the order a double-tap cycles through them."""
        return (cls.DICTATE, cls.EDIT, cls.CONVERSATION, cls.MULTIMODAL)

    @classmethod
    def default(cls) -> "Mode":
        """Default mode at app start — classic dictation, no surprises."""
        return cls.DICTATE

    @classmethod
    def next(cls, current: "Mode") -> "Mode":
        """Return the mode that follows ``current`` in the cycle."""
        order = cls.cycle_order()
        idx = order.index(current)
        return order[(idx + 1) % len(order)]


class HotkeyEventKind(Enum):
    """What a press of the 2 Key hotkey ultimately meant."""

    ACTION = "action"
    """Single tap — fire the current mode's pipeline."""

    CYCLE = "cycle"
    """Second tap of a double-tap — advance the mode, do not fire."""


@dataclass(slots=True)
class ModeState:
    """Snapshot of mode-machine state.

    ``history`` is a ring (``deque(maxlen=_HISTORY_MAX)``) of the modes the
    user has been in, oldest first. Bounded on purpose — long runtime should
    not grow memory unboundedly.
    """

    current: Mode = field(default_factory=Mode.default)
    last_tap_ts: float | None = None
    history: deque = field(
        default_factory=lambda: deque(maxlen=_HISTORY_MAX)
    )


@dataclass(frozen=True, slots=True)
class ModeEvent:
    """What ``ModeManager.on_hotkey`` returns.

    ``previous`` is the mode that was active before the press; for an
    ``ACTION`` event ``mode == previous``. ``ts`` is the timestamp the
    caller passed in (we don't read the clock ourselves — tests stay
    deterministic).
    """

    kind: HotkeyEventKind
    mode: Mode
    previous: Mode
    ts: float


class ModeManager:
    """Pure state-machine: tap timestamps in, ``ModeEvent`` out."""

    def __init__(
        self,
        double_tap_window_s: float = DEFAULT_DOUBLE_TAP_WINDOW_S,
        *,
        initial: Mode | None = None,
    ) -> None:
        if double_tap_window_s <= 0:
            raise ValueError("double_tap_window_s must be > 0")
        self._window = double_tap_window_s
        self._state = ModeState(current=initial or Mode.default())
        self._state.history.append(self._state.current)

    @property
    def state(self) -> ModeState:
        """Read-only handle to the underlying state (mutated via on_hotkey)."""
        return self._state

    @property
    def current(self) -> Mode:
        """Currently-active mode."""
        return self._state.current

    @property
    def window_s(self) -> float:
        """Configured double-tap window in seconds."""
        return self._window

    def history(self) -> tuple[Mode, ...]:
        """Bounded history of modes the user has been in (oldest first)."""
        return tuple(self._state.history)

    def reset(self, mode: Mode | None = None) -> None:
        """Reset the state-machine — used by tests and the smoke script."""
        self._state.current = mode or Mode.default()
        self._state.last_tap_ts = None
        self._state.history.clear()
        self._state.history.append(self._state.current)

    def on_hotkey(self, now_ts: float) -> ModeEvent:
        """Process one ``Strg+Super`` press.

        Single tap (no recent prior tap inside the window) → ``ACTION`` in the
        current mode. Second tap within ``window_s`` of the previous one →
        ``CYCLE`` to the next mode, and ``last_tap_ts`` is cleared so a third
        press is treated as a fresh tap (single = ACTION in the new mode).
        """
        previous = self._state.current
        last = self._state.last_tap_ts

        if last is not None and 0.0 < (now_ts - last) <= self._window:
            new_mode = Mode.next(previous)
            self._state.current = new_mode
            self._state.last_tap_ts = None
            self._state.history.append(new_mode)
            return ModeEvent(
                kind=HotkeyEventKind.CYCLE,
                mode=new_mode,
                previous=previous,
                ts=now_ts,
            )

        self._state.last_tap_ts = now_ts
        return ModeEvent(
            kind=HotkeyEventKind.ACTION,
            mode=previous,
            previous=previous,
            ts=now_ts,
        )
