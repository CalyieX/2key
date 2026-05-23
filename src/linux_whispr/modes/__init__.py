"""Mode state-machine + dispatcher (SPEC-006).

The four user-facing modes — dictation, edit-selection, conversation,
multimodal — share one hotkey (Strg+Super). A double-tap within the
configured window cycles the active mode; a single tap fires the current
mode's pipeline. The state-machine here is pure Python so it stays fully
headless-testable; the GTK pill that visualises the active mode lives in
``linux_whispr.ui.pill`` and is imported lazily.
"""

from linux_whispr.modes.dispatcher import (
    DispatchResult,
    ModeDispatcher,
    ModePipeline,
)
from linux_whispr.modes.state import (
    HotkeyEventKind,
    Mode,
    ModeEvent,
    ModeManager,
    ModeState,
)

__all__ = [
    "DispatchResult",
    "HotkeyEventKind",
    "Mode",
    "ModeDispatcher",
    "ModeEvent",
    "ModeManager",
    "ModePipeline",
    "ModeState",
]
