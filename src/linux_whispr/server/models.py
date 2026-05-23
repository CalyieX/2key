"""Pydantic request/response models for the 2Key server API (SPEC-007).

One class per shape, fields documented inline. The models double as input
validation (FastAPI returns 422 on bad payloads) and as the contract clients
on the other platforms (Windows/Android/ChromeOS) follow.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Dictate — POST /v1/dictate
# --------------------------------------------------------------------------- #
class DictateRequest(BaseModel):
    """A push-to-talk recording uploaded for speech-to-text.

    ``audio_b64`` is the base64-encoded WAV captured on the client. ``platform``
    is informational (the server logs it; the response shape does not change).
    """

    audio_b64: str = Field(..., min_length=1, description="base64-encoded WAV")
    platform: str = Field(default="unknown", description="e.g. linux / windows / android")


class DictateResponse(BaseModel):
    """Transcript of a single dictation, plus weak metadata."""

    transcript: str = ""
    language: str = ""
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Edit — POST /v1/edit
# --------------------------------------------------------------------------- #
class EditRequest(BaseModel):
    """Selected text + the user's spoken instruction (already STT-ed)."""

    selected_text: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)


class EditResponse(BaseModel):
    """Edited text, mirroring :class:`linux_whispr.edit_selection.EditResult`."""

    edited_text: str = ""
    ok: bool = False
    error: str = ""


# --------------------------------------------------------------------------- #
# Conversation — POST /v1/conversation
# --------------------------------------------------------------------------- #
class ConversationRequest(BaseModel):
    """Already-transcribed question plus an opt-in flag for synthesized audio."""

    transcript: str = Field(..., min_length=1)
    want_audio: bool = False


class ConversationResponse(BaseModel):
    """Spoken reply text (always) + optional base64-WAV when ``want_audio``."""

    reply: str = ""
    ok: bool = False
    audio_b64: str = ""
    error: str = ""


# --------------------------------------------------------------------------- #
# File context — POST /v1/file-context
# --------------------------------------------------------------------------- #
class FileContextRequest(BaseModel):
    """Multimodal ask: question + (optional) attached file as base64.

    ``file_b64``/``mime``/``filename`` are all optional — with all three empty
    the call becomes a plain text completion against the vision model.
    """

    question: str = Field(..., min_length=1)
    file_b64: str = ""
    mime: str = ""
    filename: str = ""


class FileContextResponse(BaseModel):
    """Answer text + the list of files the pipeline actually loaded."""

    answer: str = ""
    ok: bool = False
    used_files: list[str] = Field(default_factory=list)
    error: str = ""


# --------------------------------------------------------------------------- #
# Health — GET /v1/health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """Liveness probe payload. No auth required."""

    status: str = "ok"
    version: str = ""
    uptime_s: float = 0.0
