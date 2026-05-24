"""Secrets manager — libsecret/keyring with plain-file fallback.

Profi-standard for desktop apps (Signal, Bitwarden, Standard Notes all
use libsecret). Falls back to the existing ``config.toml`` api_key field
when keyring is unavailable (e.g. headless server, broken D-Bus, missing
gnome-keyring-daemon).

Usage:
    from linux_whispr.secrets import get_api_key, set_api_key, has_keyring

    key = get_api_key("openai", fallback="")
    set_api_key("openai", "sk-xyz...")
    has_keyring()  # bool — True if libsecret usable
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Service name used in libsecret schema — keep stable across versions.
_SERVICE = "linux-whispr"


def _try_import_keyring():
    """Import keyring lazily; returns module or None if unavailable."""
    try:
        import keyring as kr  # type: ignore
        # Probe: instantiate backend; raises if D-Bus broken
        backend = kr.get_keyring()
        if backend is None or "fail" in backend.__class__.__name__.lower():
            logger.info("keyring backend is null/fail (%s)", backend)
            return None
        return kr
    except ImportError:
        logger.info("keyring package not installed")
        return None
    except Exception:
        logger.exception("keyring backend probe failed")
        return None


def has_keyring() -> bool:
    """Return True if keyring/libsecret is functional on this system."""
    return _try_import_keyring() is not None


def get_api_key(provider: str, fallback: Optional[str] = None) -> Optional[str]:
    """Fetch an API key for the given provider.

    Tries keyring first, falls back to the caller-supplied value (usually
    from config.toml). Returns None if both unavailable.

    Provider is a free-form string; canonical values are openai, anthropic,
    groq, google, litellm.
    """
    kr = _try_import_keyring()
    if kr is not None:
        try:
            key = kr.get_password(_SERVICE, provider)
            if key:
                return key
        except Exception:
            logger.exception("keyring.get_password(%s) failed", provider)
    return fallback


def set_api_key(provider: str, api_key: str) -> bool:
    """Store an API key for the given provider in keyring.

    Returns True on success, False if keyring unavailable or write failed.
    Callers should also update config.toml as a record-of-truth (which
    provider is configured) but leave the actual key empty when keyring
    succeeded.
    """
    kr = _try_import_keyring()
    if kr is None:
        logger.warning("Cannot store API key for %s — keyring unavailable", provider)
        return False
    try:
        kr.set_password(_SERVICE, provider, api_key)
        logger.info("Stored API key for %s in keyring", provider)
        return True
    except Exception:
        logger.exception("keyring.set_password(%s) failed", provider)
        return False


def delete_api_key(provider: str) -> bool:
    """Remove a stored API key. Returns True even if it was not present."""
    kr = _try_import_keyring()
    if kr is None:
        return False
    try:
        kr.delete_password(_SERVICE, provider)
        return True
    except kr.errors.PasswordDeleteError:
        # Not present is OK
        return True
    except Exception:
        logger.exception("keyring.delete_password(%s) failed", provider)
        return False
