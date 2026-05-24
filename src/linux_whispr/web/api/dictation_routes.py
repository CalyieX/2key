"""POST /api/dictation/toggle — external trigger for dictation start/stop.

Lets any HTTP client (curl, browser, custom hotkey, smart-button) toggle
the same state machine the hotkey drives. Useful when:

- the OS-level hotkey conflicts with another app
- a remote machine wants to drive dictation via the network
- you want a click-button on the dashboard

The route emits a ``web.dictation.toggle`` event on the shared event bus.
``LinuxWhispr`` registers a listener for that event in :meth:`setup` and
routes it through the same handler the hotkey does.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from linux_whispr.events import event_bus

logger = logging.getLogger(__name__)

router = APIRouter()


class ToggleResponse(BaseModel):
    """Response body for the toggle route."""

    ok: bool
    detail: str


@router.post("/dictation/toggle", response_model=ToggleResponse)
async def toggle_dictation() -> ToggleResponse:
    """Emit a dictation-toggle event. Same effect as pressing the hotkey."""
    logger.info("Web API: dictation toggle requested")
    event_bus.emit("web.dictation.toggle")
    return ToggleResponse(ok=True, detail="toggle event emitted")
