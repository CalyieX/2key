"""Mode dispatcher (SPEC-006).

Wires the state-machine in :mod:`linux_whispr.modes.state` to the per-mode
pipelines (``DictationPipeline`` from SPEC-002, ``EditPipeline`` from
SPEC-003, and the upcoming conversation / multimodal pipelines from
SPEC-004/005). The dispatcher is intentionally tiny — it knows nothing
about audio, the LLM, or the X server. Pipelines are passed in by the
caller, the GTK pill is optional, and everything is fully unit-testable.

Vertrag (UX):
  * ``HotkeyEventKind.ACTION`` → run the active mode's pipeline, then flash
    the pill so the user sees which mode just fired.
  * ``HotkeyEventKind.CYCLE`` → do NOT run anything; only update the pill so
    the user knows the mode has changed. The actual recording happens on the
    next single-tap.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from linux_whispr.modes.state import (
    HotkeyEventKind,
    Mode,
    ModeEvent,
    ModeManager,
)
from linux_whispr.ui.pill import NullPill, PillUI

logger = logging.getLogger(__name__)

__all__ = [
    "DispatchResult",
    "ModeDispatcher",
    "ModePipeline",
]


@runtime_checkable
class ModePipeline(Protocol):
    """Structural type each per-mode pipeline satisfies.

    ``name`` is used for logging + pill text; ``run(payload)`` is the
    GUI-free entrypoint. We deliberately keep this *structural* (Protocol)
    so existing pipelines (``DictationPipeline``, ``EditPipeline``) and
    future ones can plug in without inheritance.
    """

    name: str

    def run(self, payload: Any) -> Any: ...  # noqa: D401, E704


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """What ``ModeDispatcher.dispatch`` and ``on_hotkey`` hand back."""

    ok: bool
    mode: Mode
    kind: HotkeyEventKind
    output: Any = None
    error: str | None = None


class ModeDispatcher:
    """Routes ``ModeEvent`` → registered pipeline + pill update."""

    def __init__(
        self,
        state: ModeManager,
        pipelines: Mapping[Mode, ModePipeline] | None = None,
        pill: PillUI | None = None,
    ) -> None:
        self._state = state
        self._pipelines: dict[Mode, ModePipeline] = dict(pipelines or {})
        self._pill: PillUI = pill or NullPill()

    @property
    def state(self) -> ModeManager:
        """The state-machine the dispatcher is wired to."""
        return self._state

    @property
    def pill(self) -> PillUI:
        """The pill UI used to signal mode changes."""
        return self._pill

    def registered_modes(self) -> tuple[Mode, ...]:
        """Modes that currently have a pipeline registered."""
        return tuple(self._pipelines)

    def register(self, mode: Mode, pipeline: ModePipeline) -> None:
        """Add or replace the pipeline for ``mode``."""
        self._pipelines[mode] = pipeline

    def dispatch(
        self,
        payload: Any = None,
        *,
        mode: Mode | None = None,
    ) -> DispatchResult:
        """Run the pipeline for ``mode`` (or the current mode if omitted).

        Missing pipeline → ``DispatchResult(ok=False, error=...)`` — we never
        let a missing-mode bug crash the hotkey thread. Pipeline raising
        likewise becomes an ``ok=False`` result so the live app can keep
        running.
        """
        target = mode or self._state.current
        pipeline = self._pipelines.get(target)
        if pipeline is None:
            msg = f"no pipeline registered for mode {target.value!r}"
            logger.warning(msg)
            return DispatchResult(
                ok=False,
                mode=target,
                kind=HotkeyEventKind.ACTION,
                error=msg,
            )

        try:
            output = pipeline.run(payload)
        except Exception as exc:
            logger.exception("Pipeline %r raised", pipeline.name)
            return DispatchResult(
                ok=False,
                mode=target,
                kind=HotkeyEventKind.ACTION,
                error=f"pipeline error: {exc}",
            )

        return DispatchResult(
            ok=True,
            mode=target,
            kind=HotkeyEventKind.ACTION,
            output=output,
        )

    def on_hotkey(
        self,
        now_ts: float,
        payload: Any = None,
    ) -> DispatchResult:
        """Hotkey-thread entrypoint.

        Translates the press into a ``ModeEvent`` via the state-machine,
        then either fires the pipeline (single-tap) or just flashes the
        pill (double-tap cycle).
        """
        event: ModeEvent = self._state.on_hotkey(now_ts)

        if event.kind is HotkeyEventKind.CYCLE:
            self._flash_mode(event.mode, prefix="mode: ")
            return DispatchResult(
                ok=True,
                mode=event.mode,
                kind=event.kind,
            )

        self._flash_mode(event.mode, prefix="")
        result = self.dispatch(payload, mode=event.mode)
        # Preserve the ``ACTION`` event-kind on the returned result.
        return DispatchResult(
            ok=result.ok,
            mode=result.mode,
            kind=event.kind,
            output=result.output,
            error=result.error,
        )

    def _flash_mode(self, mode: Mode, *, prefix: str) -> None:
        try:
            self._pill.flash(f"{prefix}{mode.value}")
        except Exception:  # noqa: BLE001
            # Pill must never bring down the hotkey thread.
            logger.exception("Pill flash failed")
