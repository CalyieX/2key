"""Tests for the server token middleware (SPEC-007)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from linux_whispr.server.auth import (
    DEFAULT_TOKEN,
    TOKEN_HEADER,
    resolve_token,
    token_required,
)


class TestResolveToken:
    def test_returns_env_value_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "prod-secret-xyz")
        assert resolve_token() == "prod-secret-xyz"

    def test_falls_back_to_default_when_env_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("TWO_KEY_TOKEN", raising=False)
        assert resolve_token() == DEFAULT_TOKEN

    def test_falls_back_to_default_when_env_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "")
        assert resolve_token() == DEFAULT_TOKEN

    def test_strips_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "  spaced-secret  ")
        assert resolve_token() == "spaced-secret"

    def test_whitespace_only_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "   ")
        assert resolve_token() == DEFAULT_TOKEN

    def test_default_token_constant_is_documented_dev_value(self) -> None:
        assert DEFAULT_TOKEN == "wonder-secret"

    def test_token_header_name(self) -> None:
        assert TOKEN_HEADER == "X-2Key-Token"


class TestTokenRequired:
    def test_accepts_matching_token(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "abc")
        assert token_required(x_2key_token="abc") is None

    def test_rejects_wrong_token(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "abc")
        with pytest.raises(HTTPException) as excinfo:
            token_required(x_2key_token="wrong")
        assert excinfo.value.status_code == 401
        assert "invalid" in excinfo.value.detail.lower()

    def test_rejects_missing_token(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "abc")
        with pytest.raises(HTTPException) as excinfo:
            token_required(x_2key_token=None)
        assert excinfo.value.status_code == 401

    def test_rejects_empty_token(self, monkeypatch) -> None:
        monkeypatch.setenv("TWO_KEY_TOKEN", "abc")
        with pytest.raises(HTTPException) as excinfo:
            token_required(x_2key_token="")
        assert excinfo.value.status_code == 401

    def test_uses_default_token_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("TWO_KEY_TOKEN", raising=False)
        assert token_required(x_2key_token=DEFAULT_TOKEN) is None
        with pytest.raises(HTTPException):
            token_required(x_2key_token="bogus")
