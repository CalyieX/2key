"""Tests for the mode state-machine (SPEC-006)."""

from __future__ import annotations

import pytest

from linux_whispr.modes.state import (
    DEFAULT_DOUBLE_TAP_WINDOW_S,
    HotkeyEventKind,
    Mode,
    ModeEvent,
    ModeManager,
    ModeState,
)


def test_mode_enum_cycle_order_is_canonical() -> None:
    assert Mode.cycle_order() == (
        Mode.DICTATE,
        Mode.EDIT,
        Mode.CONVERSATION,
        Mode.MULTIMODAL,
    )


def test_mode_default_is_dictate() -> None:
    assert Mode.default() is Mode.DICTATE


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (Mode.DICTATE, Mode.EDIT),
        (Mode.EDIT, Mode.CONVERSATION),
        (Mode.CONVERSATION, Mode.MULTIMODAL),
        (Mode.MULTIMODAL, Mode.DICTATE),
    ],
)
def test_mode_next_wraps(current: Mode, expected: Mode) -> None:
    assert Mode.next(current) is expected


def test_default_double_tap_window_is_400ms() -> None:
    assert DEFAULT_DOUBLE_TAP_WINDOW_S == 0.4


def test_manager_constructs_with_default_state() -> None:
    mgr = ModeManager()
    assert mgr.current is Mode.DICTATE
    assert mgr.window_s == DEFAULT_DOUBLE_TAP_WINDOW_S
    assert mgr.history() == (Mode.DICTATE,)


def test_manager_constructs_with_custom_initial_mode() -> None:
    mgr = ModeManager(initial=Mode.EDIT)
    assert mgr.current is Mode.EDIT
    assert mgr.history() == (Mode.EDIT,)


def test_manager_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        ModeManager(double_tap_window_s=0.0)
    with pytest.raises(ValueError):
        ModeManager(double_tap_window_s=-1.0)


def test_single_tap_is_action_in_current_mode() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    event = mgr.on_hotkey(0.0)
    assert isinstance(event, ModeEvent)
    assert event.kind is HotkeyEventKind.ACTION
    assert event.mode is Mode.DICTATE
    assert event.previous is Mode.DICTATE
    assert event.ts == 0.0
    assert mgr.current is Mode.DICTATE


def test_two_slow_taps_both_action_no_cycle() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    first = mgr.on_hotkey(0.0)
    second = mgr.on_hotkey(1.0)
    assert first.kind is HotkeyEventKind.ACTION
    assert second.kind is HotkeyEventKind.ACTION
    assert mgr.current is Mode.DICTATE


def test_double_tap_inside_window_cycles_mode() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    first = mgr.on_hotkey(0.0)
    second = mgr.on_hotkey(0.1)
    assert first.kind is HotkeyEventKind.ACTION
    assert first.mode is Mode.DICTATE
    assert second.kind is HotkeyEventKind.CYCLE
    assert second.previous is Mode.DICTATE
    assert second.mode is Mode.EDIT
    assert mgr.current is Mode.EDIT


def test_double_tap_at_exact_window_still_cycles() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(0.0)
    event = mgr.on_hotkey(0.4)
    assert event.kind is HotkeyEventKind.CYCLE


def test_tap_just_outside_window_is_action_not_cycle() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(0.0)
    event = mgr.on_hotkey(0.401)
    assert event.kind is HotkeyEventKind.ACTION
    assert mgr.current is Mode.DICTATE


def test_full_cycle_returns_to_dictate() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    expected = [Mode.EDIT, Mode.CONVERSATION, Mode.MULTIMODAL, Mode.DICTATE]
    ts = 0.0
    for want in expected:
        mgr.on_hotkey(ts)
        ts += 0.1
        event = mgr.on_hotkey(ts)
        assert event.kind is HotkeyEventKind.CYCLE
        assert event.mode is want
        ts += 1.0


def test_triple_tap_is_cycle_then_action_in_new_mode() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    e1 = mgr.on_hotkey(0.0)
    e2 = mgr.on_hotkey(0.1)
    e3 = mgr.on_hotkey(0.2)
    assert e1.kind is HotkeyEventKind.ACTION
    assert e2.kind is HotkeyEventKind.CYCLE
    assert e2.mode is Mode.EDIT
    assert e3.kind is HotkeyEventKind.ACTION
    assert e3.mode is Mode.EDIT
    assert mgr.current is Mode.EDIT


def test_quadruple_tap_cycles_twice() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(0.0)
    e2 = mgr.on_hotkey(0.1)
    e3 = mgr.on_hotkey(0.2)
    e4 = mgr.on_hotkey(0.3)
    assert e2.kind is HotkeyEventKind.CYCLE
    assert e3.kind is HotkeyEventKind.ACTION
    assert e4.kind is HotkeyEventKind.CYCLE
    assert mgr.current is Mode.CONVERSATION


def test_five_rapid_taps_within_100ms_dont_crash() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    timestamps = [0.0, 0.02, 0.04, 0.06, 0.08]
    events = [mgr.on_hotkey(t) for t in timestamps]
    kinds = [e.kind for e in events]
    assert kinds[0] is HotkeyEventKind.ACTION
    assert kinds[1] is HotkeyEventKind.CYCLE
    assert kinds[2] is HotkeyEventKind.ACTION
    assert kinds[3] is HotkeyEventKind.CYCLE
    assert kinds[4] is HotkeyEventKind.ACTION
    assert mgr.current is Mode.CONVERSATION


def test_backwards_time_is_treated_as_fresh_tap() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(10.0)
    event = mgr.on_hotkey(5.0)
    assert event.kind is HotkeyEventKind.ACTION
    assert mgr.current is Mode.DICTATE


def test_exact_same_timestamp_twice_is_not_cycle() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(1.0)
    event = mgr.on_hotkey(1.0)
    assert event.kind is HotkeyEventKind.ACTION


def test_history_is_bounded_and_records_cycles() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    ts = 0.0
    for _ in range(20):
        mgr.on_hotkey(ts)
        ts += 0.1
        mgr.on_hotkey(ts)
        ts += 1.0
    history = mgr.history()
    assert len(history) <= 16
    assert history[-1] is mgr.current


def test_reset_returns_to_default_state() -> None:
    mgr = ModeManager(double_tap_window_s=0.4)
    mgr.on_hotkey(0.0)
    mgr.on_hotkey(0.1)
    assert mgr.current is Mode.EDIT
    mgr.reset()
    assert mgr.current is Mode.DICTATE
    assert mgr.history() == (Mode.DICTATE,)
    assert mgr.state.last_tap_ts is None


def test_reset_to_explicit_mode() -> None:
    mgr = ModeManager()
    mgr.reset(Mode.MULTIMODAL)
    assert mgr.current is Mode.MULTIMODAL


def test_state_dataclass_holds_expected_fields() -> None:
    state = ModeState()
    assert state.current is Mode.DICTATE
    assert state.last_tap_ts is None
    assert list(state.history) == []


def test_mode_event_is_immutable_frozen_dataclass() -> None:
    event = ModeEvent(
        kind=HotkeyEventKind.ACTION,
        mode=Mode.DICTATE,
        previous=Mode.DICTATE,
        ts=0.0,
    )
    with pytest.raises(Exception):
        event.kind = HotkeyEventKind.CYCLE  # type: ignore[misc]
