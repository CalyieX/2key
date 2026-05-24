"""Direct evdev hotkey listener — bypasses X11 + Wayland for raw key capture.

Reads /dev/input/event* keyboard devices directly via python-evdev. Works on
both X11 and Wayland sessions because it sees keystrokes at the kernel input
layer, before the display server processes them. Requires the user to be in
the ``input`` group (Ubuntu default for desktop users).

Supports both classic "press combo to trigger" hotkeys and the
push-to-talk "hold modifiers" pattern used by Pixi (Ctrl+Super held while
speaking). The hotkey-string parser is shared with the X11 backend.
"""

from __future__ import annotations

import logging
import select
import threading
from collections.abc import Callable
from pathlib import Path

from linux_whispr.input.hotkey import HotkeyListener
from linux_whispr.input.x11_hotkey import _parse_hotkey

logger = logging.getLogger(__name__)

# Map our modifier names to evdev key codes (left + right variants).
_MOD_TO_EVDEV_CODES: dict[str, set[int]] = {
    "ctrl": {29, 97},     # KEY_LEFTCTRL=29, KEY_RIGHTCTRL=97
    "control": {29, 97},
    "alt": {56, 100},     # KEY_LEFTALT=56, KEY_RIGHTALT=100
    "shift": {42, 54},    # KEY_LEFTSHIFT=42, KEY_RIGHTSHIFT=54
    "super": {125, 126},  # KEY_LEFTMETA=125, KEY_RIGHTMETA=126
    "meta": {125, 126},
    "hyper": {125, 126},
}

# Common non-modifier trigger keys we care about for combo hotkeys.
_KEY_TO_EVDEV_CODE: dict[str, int] = {
    "space": 57,
    "tab": 15,
    "enter": 28,
    "return": 28,
    "escape": 1,
    "esc": 1,
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63, "f6": 64,
    "f7": 65, "f8": 66, "f9": 67, "f10": 68, "f11": 87, "f12": 88,
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
    "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
    "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
    "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "super_l": 125, "control_l": 29, "alt_l": 56, "shift_l": 42,
    "mod1_l": 56, "mod4_l": 125,
}


class _Binding:
    """One registered hotkey + its current pressed-state tracking."""

    def __init__(
        self,
        hotkey: str,
        callback: Callable[[], None],
        name: str,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.hotkey = hotkey
        self.callback = callback
        self.release_callback = release_callback
        self.name = name
        modifiers, key = _parse_hotkey(hotkey)
        self.modifier_codes: list[set[int]] = [
            _MOD_TO_EVDEV_CODES[m] for m in modifiers if m in _MOD_TO_EVDEV_CODES
        ]
        self.trigger_code: int | None = _resolve_key_code(key)
        self.active = False


def _resolve_key_code(key: str) -> int | None:
    """Look up a key name (case-insensitive) in the evdev-code map."""
    k = key.lower()
    if k in _KEY_TO_EVDEV_CODE:
        return _KEY_TO_EVDEV_CODE[k]
    for mod_name, codes in _MOD_TO_EVDEV_CODES.items():
        if k == mod_name or k.startswith(mod_name + "_"):
            return min(codes)
    return None


def _find_keyboard_devices() -> list[str]:
    """Return paths to ALL readable keyboard-capable input devices.

    We probe each /dev/input/event* directly via evdev — the device's
    declared capabilities tell us whether it generates KEY events. This
    catches:

    - real USB keyboards (also visible under /dev/input/by-id/)
    - the persistent ydotoold virtual uinput keyboard (no by-id symlink)
    - any other virtual keyboards (test harnesses, accessibility tools)

    Falling back to /dev/input/by-id/ when the live-probe path can't be
    used (no python-evdev, no permission to open devices) keeps the
    listener working on older / locked-down systems.
    """
    paths: set[str] = set()
    try:
        import evdev  # noqa: PLC0415 — local; only when probing
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                # EV_KEY (=1) means the device emits key events; a real
                # keyboard always has at least KEY_A..Z (codes 30-44).
                if evdev.ecodes.EV_KEY in caps:
                    key_codes = set(caps[evdev.ecodes.EV_KEY])
                    # Accept any device that emits at least one of the
                    # hotkey-relevant keys (modifiers, space, common
                    # letters). Real keyboards always pass the letter
                    # check; minimal virtual test devices may declare
                    # only modifiers + space, which is still useful.
                    relevant = {
                        29, 97,    # ctrl L/R
                        56, 100,   # alt L/R
                        125, 126,  # super L/R
                        57,        # space
                        30, 31, 32, 33, 34,  # a, s, d, f, g
                    }
                    if key_codes & relevant:
                        paths.add(path)
                dev.close()
            except (OSError, PermissionError):
                continue
    except ImportError:
        pass

    if paths:
        return sorted(paths)

    # Fallback: glob /dev/input/by-id for *event-kbd entries.
    by_id = Path("/dev/input/by-id")
    if by_id.is_dir():
        for entry in by_id.iterdir():
            if "event-kbd" in entry.name:
                try:
                    resolved = entry.resolve()
                    if resolved.exists():
                        paths.add(str(resolved))
                except OSError:
                    continue
    return sorted(paths)


class EvdevHotkeyListener(HotkeyListener):
    """Hotkey listener reading /dev/input directly via python-evdev."""

    def __init__(self) -> None:
        self._bindings: list[_Binding] = []
        self._thread: threading.Thread | None = None
        self._running = False
        self._pressed: set[int] = set()

    def register(self, hotkey: str, callback: Callable[[], None], name: str = "") -> None:
        binding = _Binding(hotkey, callback, name or hotkey)
        logger.info(
            "Registered evdev hotkey: %s (mods=%s, trigger=%s)",
            hotkey, binding.modifier_codes, binding.trigger_code,
        )
        self._bindings.append(binding)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="evdev-hotkey",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _listen_loop(self) -> None:
        try:
            import evdev
        except ImportError:
            logger.error("python-evdev not installed")
            return

        device_paths = _find_keyboard_devices()
        if not device_paths:
            logger.error("No keyboard devices found under /dev/input/by-id/")
            return

        devices = []
        for path in device_paths:
            try:
                devices.append(evdev.InputDevice(path))
            except (OSError, PermissionError) as exc:
                logger.warning("Cannot open %s: %s", path, exc)
        if not devices:
            logger.error("No keyboard devices openable; user in `input` group?")
            return
        logger.info(
            "evdev hotkey listener watching %d device(s): %s",
            len(devices), ", ".join(d.name for d in devices),
        )

        fd_to_device = {d.fd: d for d in devices}

        while self._running:
            r, _, _ = select.select(fd_to_device.keys(), [], [], 0.5)
            for fd in r:
                device = fd_to_device.get(fd)
                if device is None:
                    continue
                try:
                    for event in device.read():
                        if event.type == evdev.ecodes.EV_KEY:
                            self._handle_key_event(event.code, event.value)
                except OSError:
                    continue

        logger.info("evdev hotkey listener stopped")

    def _handle_key_event(self, code: int, value: int) -> None:
        if value == 1:
            self._pressed.add(code)
        elif value == 0:
            self._pressed.discard(code)

        for binding in self._bindings:
            self._check_binding(binding)

    def _check_binding(self, binding: _Binding) -> None:
        mods_held = all(
            any(c in self._pressed for c in mod_codes)
            for mod_codes in binding.modifier_codes
        )
        trigger_held = (
            binding.trigger_code is None
            or binding.trigger_code in self._pressed
        )
        active_now = mods_held and trigger_held
        if active_now and not binding.active:
            binding.active = True
            try:
                binding.callback()
            except Exception:
                logger.exception("Error in evdev hotkey callback for %s", binding.name)
        elif binding.active and not active_now:
            # Trailing edge — release detected. Push-to-talk behaviour:
            # if the caller registered a separate release_callback we use
            # it; otherwise we re-fire the SAME callback so a toggle-style
            # handler (state-machine flip) is forced to PTT semantics
            # (press = start, release = stop).
            binding.active = False
            release_cb = binding.release_callback or binding.callback
            try:
                release_cb()
            except Exception:
                logger.exception(
                    "Error in evdev release callback for %s", binding.name
                )
