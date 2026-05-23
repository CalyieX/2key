"""Tests for the FastAPI request/response models (SPEC-007)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from linux_whispr.server.models import (
    ConversationRequest,
    ConversationResponse,
    DictateRequest,
    DictateResponse,
    EditRequest,
    EditResponse,
    FileContextRequest,
    FileContextResponse,
    HealthResponse,
)


class TestDictateRequest:
    def test_accepts_minimal_payload(self) -> None:
        req = DictateRequest(audio_b64="aGVsbG8=")
        assert req.audio_b64 == "aGVsbG8="
        assert req.platform == "unknown"

    def test_rejects_missing_audio(self) -> None:
        with pytest.raises(ValidationError):
            DictateRequest()  # type: ignore[call-arg]

    def test_rejects_empty_audio_string(self) -> None:
        with pytest.raises(ValidationError):
            DictateRequest(audio_b64="")

    def test_honours_explicit_platform(self) -> None:
        req = DictateRequest(audio_b64="x", platform="windows")
        assert req.platform == "windows"


class TestDictateResponse:
    def test_defaults_are_safe(self) -> None:
        resp = DictateResponse()
        assert resp.transcript == ""
        assert resp.language == ""
        assert resp.confidence == 0.0

    def test_round_trip(self) -> None:
        resp = DictateResponse(transcript="hello", language="en", confidence=0.91)
        d = resp.model_dump()
        assert d == {"transcript": "hello", "language": "en", "confidence": 0.91}


class TestEditRequest:
    def test_accepts_valid_payload(self) -> None:
        req = EditRequest(selected_text="hi", instruction="capitalize it")
        assert req.selected_text == "hi"

    def test_rejects_empty_selected_text(self) -> None:
        with pytest.raises(ValidationError):
            EditRequest(selected_text="", instruction="x")

    def test_rejects_empty_instruction(self) -> None:
        with pytest.raises(ValidationError):
            EditRequest(selected_text="x", instruction="")


class TestEditResponse:
    def test_defaults_are_safe(self) -> None:
        resp = EditResponse()
        assert resp.edited_text == ""
        assert resp.ok is False
        assert resp.error == ""

    def test_round_trip(self) -> None:
        resp = EditResponse(edited_text="HI", ok=True, error="")
        assert resp.model_dump()["ok"] is True


class TestConversationRequest:
    def test_accepts_minimal_payload(self) -> None:
        req = ConversationRequest(transcript="hi")
        assert req.transcript == "hi"
        assert req.want_audio is False

    def test_rejects_empty_transcript(self) -> None:
        with pytest.raises(ValidationError):
            ConversationRequest(transcript="")

    def test_want_audio_toggle(self) -> None:
        assert ConversationRequest(transcript="x", want_audio=True).want_audio is True


class TestConversationResponse:
    def test_defaults_are_safe(self) -> None:
        resp = ConversationResponse()
        assert resp.reply == ""
        assert resp.ok is False
        assert resp.audio_b64 == ""

    def test_round_trip(self) -> None:
        resp = ConversationResponse(reply="56", ok=True)
        d = resp.model_dump()
        assert d["reply"] == "56" and d["ok"] is True


class TestFileContextRequest:
    def test_accepts_text_only(self) -> None:
        req = FileContextRequest(question="summarise")
        assert req.question == "summarise"
        assert req.file_b64 == ""

    def test_accepts_with_file(self) -> None:
        req = FileContextRequest(
            question="what is this", file_b64="aGVsbG8=", mime="image/png", filename="a.png"
        )
        assert req.mime == "image/png"
        assert req.filename == "a.png"

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValidationError):
            FileContextRequest(question="")


class TestFileContextResponse:
    def test_defaults_are_safe(self) -> None:
        resp = FileContextResponse()
        assert resp.answer == ""
        assert resp.used_files == []

    def test_used_files_is_independent_per_instance(self) -> None:
        a = FileContextResponse()
        b = FileContextResponse()
        a.used_files.append("/tmp/x")
        assert b.used_files == []


class TestHealthResponse:
    def test_defaults_are_safe(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"

    def test_with_values(self) -> None:
        resp = HealthResponse(status="ok", version="1.2.3", uptime_s=5.5)
        d = resp.model_dump()
        assert d == {"status": "ok", "version": "1.2.3", "uptime_s": 5.5}
