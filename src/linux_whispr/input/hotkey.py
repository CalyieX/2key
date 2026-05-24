"""Global hotkey listener — dispatches to X11 or Wayland implementation."""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from linux_whispr.platform.detect import PlatformInfo

logger = logging.getLogger(__name__)


class HotkeyListener(abc.ABC):
    """Abstract base class for global hotkey listeners."""

    @abc.abstractmethod
    def register(self, hotkey: str, callback: Callable[[], None], name: str = "") -> None:
        """Register a global hotkey.

        Args:
            hotkey: Hotkey string (e.g., "F12", "<Ctrl><Shift>h").
            callback: Function to call when the hotkey is pressed.
            name: Human-readable name for the hotkey binding.
        """
        ...

    @abc.abstractmethod
    def start(self) -> None:
        """Start listening for hotkeys (blocks or runs in a thread)."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop listening for hotkeys."""
        ...


def create_hotkey_listener(platform: PlatformInfo) -> HotkeyListener:
    """Factory: create the appropriate hotkey listener for the platform."""
    from linux_whispr.platform.detect import DisplayServer

    # Prefer evdev — bypasses X11 / Wayland boundaries (Wayland-native apps
    # do not propagate keystrokes to Xwayland XGrabKey, so the legacy X11
    # listener silently misses every key on a Wayland session). Requires
    # the user to be in the `input` group.
    try:
        from linux_whispr.input.evdev_hotkey import EvdevHotkeyListener
        import evdev as _evdev  # noqa: F401 — probe import
        from pathlib import Path as _P
        _by_id = _P("/dev/input/by-id")
        if _by_id.is_dir():
            for entry in _by_id.iterdir():
                if "event-kbd" in entry.name:
                    try:
                        if entry.resolve().exists():
                            logger.info("Using evdev for hotkeys (kernel-layer, X11+Wayland-agnostic)")
                            return EvdevHotkeyListener()
                    except OSError:
                        continue
    except ImportError:
        pass

    if platform.display_server == DisplayServer.WAYLAND:
        try:
            from linux_whispr.input.wayland_hotkey import WaylandHotkeyListener

            logger.info("Using Wayland D-Bus GlobalShortcuts portal for hotkeys")
            return WaylandHotkeyListener()
        except ImportError:
            logger.warning("Wayland hotkey support unavailable, trying X11 fallback")

    if platform.display_server in (DisplayServer.X11, DisplayServer.WAYLAND):
        # On Wayland we can still try X11 via XWayland
        try:
            from linux_whispr.input.x11_hotkey import X11HotkeyListener

            logger.info("Using X11 XGrabKey for hotkeys")
            return X11HotkeyListener()
        except ImportError:
            logger.error("X11 hotkey support unavailable (python-xlib not installed)")

    # Last resort: pynput (works on both but may need permissions)
    try:
        from linux_whispr.input.pynput_hotkey import PynputHotkeyListener

        logger.info("Using pynput for hotkeys (fallback)")
        return PynputHotkeyListener()
    except ImportError:
        pass

    raise RuntimeError(
        "No hotkey backend available. Install python-xlib (X11) or ensure "
        "D-Bus GlobalShortcuts portal is available (Wayland)."
    )
