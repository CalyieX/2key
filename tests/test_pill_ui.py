"""Tests for the pill UI (SPEC-006)."""

from __future__ import annotations

import sys

from linux_whispr.ui.pill import GtkPill, NullPill, PillUI


def test_null_pill_satisfies_protocol() -> None:
    assert isinstance(NullPill(), PillUI)


def test_null_pill_show_hide_flash_dont_raise() -> None:
    pill = NullPill()
    pill.show("dictate")
    pill.hide()
    pill.flash("hello")
    pill.flash("with duration", duration_s=2.5)


def test_null_pill_available_is_false() -> None:
    assert NullPill().available() is False


def test_gtk_pill_satisfies_protocol() -> None:
    assert isinstance(GtkPill(), PillUI)


def test_gtk_pill_construction_does_not_import_gi() -> None:
    sys.modules.pop("gi", None)
    sys.modules.pop("gi.repository", None)
    pill = GtkPill()
    # Construction alone must not trigger the heavy GTK import.
    assert "gi" not in sys.modules
    assert pill._gtk is None


def test_gtk_pill_construction_accepts_position_and_margin() -> None:
    pill = GtkPill(position="bottom-left", margin_px=42)
    assert pill._position == "bottom-left"
    assert pill._margin == 42


def test_gtk_pill_methods_no_op_when_gtk_unavailable(monkeypatch) -> None:
    pill = GtkPill()

    def fake_load(self):
        self._import_error = ImportError("simulated")
        return None

    monkeypatch.setattr(GtkPill, "_load_gtk", fake_load)
    pill.show("dictate")
    pill.hide()
    pill.flash("msg")
    assert pill.available() is False


def test_gtk_pill_uses_lazy_loaded_gtk(monkeypatch) -> None:
    pill = GtkPill()
    fake_gtk = object()

    def fake_load(self):
        self._gtk = fake_gtk
        return fake_gtk

    monkeypatch.setattr(GtkPill, "_load_gtk", fake_load)
    assert pill.available() is True
    pill.show("edit")
    pill.flash("flash-msg")
    pill.hide()


def test_gtk_pill_caches_failed_import(monkeypatch) -> None:
    pill = GtkPill()
    calls = {"n": 0}

    real_load = GtkPill._load_gtk

    def counting_load(self):
        calls["n"] += 1
        # Force the failure path by faking gi.require_version raising.
        if calls["n"] == 1:
            self._import_error = ImportError("first attempt failed")
            return None
        return real_load(self)

    monkeypatch.setattr(GtkPill, "_load_gtk", counting_load)
    assert pill.available() is False
    # Second call would also return None from cached error if we used real
    # logic; here we just ensure repeated calls are safe.
    pill.flash("x")
    pill.hide()


def test_null_pill_can_be_used_anywhere_a_pill_is_expected() -> None:
    def accept(p: PillUI) -> None:
        p.show("test")
        p.hide()
        p.flash("yo")

    accept(NullPill())
    accept(GtkPill())


def test_gtk_pill_load_gtk_returns_none_when_import_fails(monkeypatch) -> None:
    """Force the import branch to fail and verify caching."""
    import builtins

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "gi":
            raise ImportError("gi not available in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("gi", None)
    pill = GtkPill()
    assert pill._load_gtk() is None
    # Second call should hit the cached-error branch and still return None.
    assert pill._load_gtk() is None
    assert pill._import_error is not None
