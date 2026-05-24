"""Unix-socket trigger bridge.

See ipc/__init__.py for the VOXD-trick context. Socket path is
$XDG_RUNTIME_DIR/linux-whispr-trigger.sock (or /tmp fallback). Server
runs as background thread in the main app; client is short-lived
subprocess from `linux-whispr --trigger-record`.

Protocol: single ASCII line per connection, one of toggle|start|stop|ping.
Authentication is Unix-socket chmod 600 only — same trust boundary as
~/.config files.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

CMD_TOGGLE = "toggle"
CMD_START = "start"
CMD_STOP = "stop"
CMD_PING = "ping"
_VALID_COMMANDS = frozenset({CMD_TOGGLE, CMD_START, CMD_STOP, CMD_PING})


def _default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        base = Path(runtime)
    else:
        base = Path("/tmp") / f"linux-whispr-{os.getuid()}"
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base / "linux-whispr-trigger.sock"


SOCKET_PATH = _default_socket_path()


class TriggerSocketServer:
    """Background-thread Unix-socket server for trigger commands."""

    def __init__(
        self,
        on_command: Callable[[str], None],
        socket_path: Path | None = None,
    ) -> None:
        self._on_command = on_command
        self._socket_path = socket_path or SOCKET_PATH
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            logger.warning("TriggerSocketServer.start called twice")
            return
        # Clean stale socket
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError as exc:
                logger.error("Cannot remove stale socket %s: %s", self._socket_path, exc)
                return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self._socket_path))
            os.chmod(self._socket_path, 0o600)
            sock.listen(4)
            sock.settimeout(0.5)
        except OSError as exc:
            logger.error("Cannot bind trigger socket %s: %s", self._socket_path, exc)
            sock.close()
            return
        self._server_socket = sock
        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="ipc-trigger-socket",
        )
        self._thread.start()
        logger.info("Trigger socket listening: %s", self._socket_path)

    def stop(self) -> None:
        self._running = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        try:
            self._socket_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.info("Trigger socket stopped")

    def _accept_loop(self) -> None:
        assert self._server_socket is not None
        while self._running:
            try:
                client_sock, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle_client(client_sock)
            finally:
                try:
                    client_sock.close()
                except OSError:
                    pass

    def _handle_client(self, client_sock: socket.socket) -> None:
        try:
            client_sock.settimeout(2.0)
            raw = client_sock.recv(64)
        except (socket.timeout, OSError) as exc:
            logger.warning("Trigger client read failed: %s", exc)
            return
        command = raw.decode("ascii", errors="replace").strip().lower()
        if command not in _VALID_COMMANDS:
            logger.warning("Unknown trigger command: %r", command)
            try:
                client_sock.sendall(b"ERR unknown\n")
            except OSError:
                pass
            return
        try:
            client_sock.sendall(b"OK\n")
        except OSError:
            pass
        if command == CMD_PING:
            return
        try:
            self._on_command(command)
        except Exception:
            logger.exception("Trigger callback for command %s failed", command)


def send_trigger(command: str = CMD_TOGGLE, socket_path: Path | None = None) -> bool:
    """Send command to running app socket. Returns True if OK was acknowledged."""
    path = socket_path or SOCKET_PATH
    if not path.exists():
        logger.error("No trigger socket at %s -- is the app running?", path)
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(2.0)
        sock.connect(str(path))
        sock.sendall(f"{command}\n".encode("ascii"))
        response = sock.recv(16).decode("ascii", errors="replace").strip()
        return response == "OK"
    except (OSError, socket.timeout) as exc:
        logger.error("Trigger send failed: %s", exc)
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass
