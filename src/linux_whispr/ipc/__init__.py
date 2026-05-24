"""IPC layer for K&K Voice — Unix-socket bridge for external triggers."""

from linux_whispr.ipc.trigger_socket import (
    CMD_PING,
    CMD_START,
    CMD_STOP,
    CMD_TOGGLE,
    SOCKET_PATH,
    TriggerSocketServer,
    send_trigger,
)

__all__ = [
    "CMD_PING",
    "CMD_START",
    "CMD_STOP",
    "CMD_TOGGLE",
    "SOCKET_PATH",
    "TriggerSocketServer",
    "send_trigger",
]
