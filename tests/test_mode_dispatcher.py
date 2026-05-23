"""Tests for the mode dispatcher (SPEC-006)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from linux_whispr.modes.dispatcher import DispatchResult, ModeDispatcher, ModePipeline
from linux_whispr.modes.state import HotkeyEventKind, Mode, ModeManager
from linux_whispr.ui.pill import NullPill


class _StubPipeline:
    """Bare-minimum ModePipeline implementation for routing tests."""

    def __init__(self, name: str, result: Any = "ok") -> None:
        self.name = name
        self.result = result
        self.calls: list[Any] = []

    def run(self, payload: Any) -> Any:
        self.calls.append(payload)
        return self.result


class _RaisingPipeline:
    name = "raiser"

    def run(self, payload: Any) -> Any:
        raise RuntimeError("boom")


def _four_pipelines() -> dict[Mode, _StubPipeline]:
    return {
        Mode.DICTATE: _StubPipeline("dictate", result="d-result"),
        Mode.EDIT: _StubPipeline("edit", result="e-result"),
        Mode.CONVERSATION: _StubPipeline("conversation", result="c-result"),
        Mode.MULTIMODAL: _StubPipeline("multimodal", result="m-result"),
    }


def test_dispatcher_constructs_without_pipelines() -> None:
    mgr = ModeManager()
    disp = ModeDispatcher(mgr)
    assert disp.state is mgr
    assert disp.registered_modes() == ()
    assert isinstance(disp.pill, NullPill)


def test_dispatch_with_no_pipeline_returns_error_result() -> None:
    mgr = ModeManager()
    disp = ModeDispatcher(mgr)
    result = disp.dispatch("audio")
    assert isinstance(result, DispatchResult)
    assert result.ok is False
    assert result.mode is Mode.DICTATE
    assert "no pipeline" in (result.error or "")


def test_dispatch_routes_to_current_mode_pipeline() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    disp = ModeDispatcher(mgr, pipelines)
    result = disp.dispatch("audio")
    assert result.ok is True
    assert result.mode is Mode.DICTATE
    assert result.output == "d-result"
    assert pipelines[Mode.DICTATE].calls == ["audio"]
    for other in (Mode.EDIT, Mode.CONVERSATION, Mode.MULTIMODAL):
        assert pipelines[other].calls == []


def test_dispatch_with_explicit_mode_overrides_current() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    disp = ModeDispatcher(mgr, pipelines)
    result = disp.dispatch("x", mode=Mode.MULTIMODAL)
    assert result.ok is True
    assert result.mode is Mode.MULTIMODAL
    assert result.output == "m-result"
    assert pipelines[Mode.MULTIMODAL].calls == ["x"]


def test_dispatch_catches_pipeline_exception() -> None:
    mgr = ModeManager()
    disp = ModeDispatcher(mgr, {Mode.DICTATE: _RaisingPipeline()})
    result = disp.dispatch("y")
    assert result.ok is False
    assert "boom" in (result.error or "")
    assert result.mode is Mode.DICTATE


def test_register_adds_pipeline_for_later_modes() -> None:
    mgr = ModeManager()
    disp = ModeDispatcher(mgr)
    assert disp.registered_modes() == ()
    stub = _StubPipeline("late")
    disp.register(Mode.CONVERSATION, stub)
    assert disp.registered_modes() == (Mode.CONVERSATION,)
    result = disp.dispatch("z", mode=Mode.CONVERSATION)
    assert result.ok is True
    assert stub.calls == ["z"]


def test_on_hotkey_action_runs_current_pipeline_and_flashes_pill() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    pill = MagicMock(spec=["show", "hide", "flash", "available"])
    disp = ModeDispatcher(mgr, pipelines, pill=pill)
    result = disp.on_hotkey(0.0, payload="audio")
    assert result.ok is True
    assert result.kind is HotkeyEventKind.ACTION
    assert result.mode is Mode.DICTATE
    assert pipelines[Mode.DICTATE].calls == ["audio"]
    pill.flash.assert_called_once()
    flash_arg = pill.flash.call_args.args[0]
    assert "dictate" in flash_arg


def test_on_hotkey_cycle_does_not_run_any_pipeline() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    pill = MagicMock(spec=["show", "hide", "flash", "available"])
    disp = ModeDispatcher(mgr, pipelines, pill=pill)
    disp.on_hotkey(0.0)
    pill.flash.reset_mock()
    for p in pipelines.values():
        p.calls.clear()

    result = disp.on_hotkey(0.1, payload="should-be-ignored")
    assert result.ok is True
    assert result.kind is HotkeyEventKind.CYCLE
    assert result.mode is Mode.EDIT
    for p in pipelines.values():
        assert p.calls == []
    pill.flash.assert_called_once()
    flash_arg = pill.flash.call_args.args[0]
    assert "edit" in flash_arg
    assert "mode" in flash_arg.lower()


def test_on_hotkey_cycle_changes_mode_for_next_action() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    disp = ModeDispatcher(mgr, pipelines)
    disp.on_hotkey(0.0)
    disp.on_hotkey(0.1)
    pipelines[Mode.DICTATE].calls.clear()
    pipelines[Mode.EDIT].calls.clear()

    result = disp.on_hotkey(1.0, payload="next")
    assert result.kind is HotkeyEventKind.ACTION
    assert result.mode is Mode.EDIT
    assert pipelines[Mode.EDIT].calls == ["next"]
    assert pipelines[Mode.DICTATE].calls == []


def test_on_hotkey_action_with_missing_pipeline_returns_error() -> None:
    mgr = ModeManager()
    disp = ModeDispatcher(mgr)
    result = disp.on_hotkey(0.0, payload="ignored")
    assert result.kind is HotkeyEventKind.ACTION
    assert result.ok is False
    assert "no pipeline" in (result.error or "")


def test_pill_flash_failure_does_not_crash_dispatch() -> None:
    mgr = ModeManager()
    pill = MagicMock(spec=["show", "hide", "flash", "available"])
    pill.flash.side_effect = RuntimeError("pill blew up")
    disp = ModeDispatcher(mgr, _four_pipelines(), pill=pill)
    result = disp.on_hotkey(0.0, payload="x")
    assert result.ok is True
    assert result.kind is HotkeyEventKind.ACTION
    pill.flash.assert_called_once()


def test_dispatch_result_is_frozen() -> None:
    result = DispatchResult(
        ok=True,
        mode=Mode.DICTATE,
        kind=HotkeyEventKind.ACTION,
        output="x",
    )
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]


def test_mode_pipeline_protocol_accepts_structurally_matching_objects() -> None:
    stub = _StubPipeline("anything")
    assert isinstance(stub, ModePipeline)


def test_full_dispatch_cycle_visits_every_mode() -> None:
    mgr = ModeManager()
    pipelines = _four_pipelines()
    disp = ModeDispatcher(mgr, pipelines)
    expected_modes = [Mode.EDIT, Mode.CONVERSATION, Mode.MULTIMODAL, Mode.DICTATE]
    ts = 0.0
    for want in expected_modes:
        disp.on_hotkey(ts)
        ts += 0.1
        cycle = disp.on_hotkey(ts)
        assert cycle.kind is HotkeyEventKind.CYCLE
        assert cycle.mode is want
        ts += 1.0
        action = disp.on_hotkey(ts, payload=f"p-{want.value}")
        assert action.kind is HotkeyEventKind.ACTION
        assert action.mode is want
        ts += 1.0
    # Each mode pipeline ran exactly once on its tagged payload (DICTATE
    # absorbs the warm-up taps too — ACTION always fires the current mode).
    for mode, pipeline in pipelines.items():
        tagged = [p for p in pipeline.calls if p == f"p-{mode.value}"]
        assert tagged == [f"p-{mode.value}"]
