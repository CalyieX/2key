"""Token middleware for the 2Key server (SPEC-007).

The server lives on the WireGuard/LAN interface; the token is the second layer.
We deliberately keep it boring — a single shared secret in the ``X-2Key-Token``
header, resolved from ``TWO_KEY_TOKEN`` env, falling back to a dev default.
No JWT, no rotation — v1 is single-user inside a private network.
"""

from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

#: Header name clients send the shared secret in. Case-insensitive in HTTP.
TOKEN_HEADER = "X-2Key-Token"

#: Dev-only default. Production must override via ``TWO_KEY_TOKEN``.
DEFAULT_TOKEN = "wonder-secret"


def resolve_token() -> str:
    """Return the token the server should accept.

    Reads ``TWO_KEY_TOKEN`` env; falls back to :data:`DEFAULT_TOKEN`. Logs a
    one-line warning when the dev default is in use so it is visible at boot.
    """
    token = os.environ.get("TWO_KEY_TOKEN", "").strip()
    if token:
        return token
    logger.warning(
        "security: TWO_KEY_TOKEN not set, using dev default (%s)", DEFAULT_TOKEN
    )
    return DEFAULT_TOKEN


def token_required(
    x_2key_token: str | None = Header(default=None, alias=TOKEN_HEADER),
) -> None:
    """FastAPI dependency — raise 401 unless the header matches the active token.

    Two-arg signature is required by FastAPI so it can extract the header by
    its case-insensitive alias. The function returns ``None`` on success — it
    is used purely for its side effect (raise on mismatch).
    """
    expected = resolve_token()
    if not x_2key_token or x_2key_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
        )
