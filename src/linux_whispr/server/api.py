"""FastAPI app for the 2Key multi-machine server (SPEC-007).

The single source of truth for how external clients talk to the 2Key brain:
five endpoints (one per mode + a health probe) wrapping the existing
pipelines from SPEC-002-005.

Design notes:
  * Module import is side-effect-free w.r.t. pipelines and network — calling
    ``from linux_whispr.server.api import app`` will NOT build a
    :class:`DictationPipeline` or open a socket. Pipelines are constructed
    lazily on first use and tests inject their own bundle.
  * Every endpoint that is not ``/v1/health`` is gated by the token middleware.
  * Pipeline errors are translated to ``ok=False`` JSON instead of HTTP 5xx so
    the client UX stays uniform across "down" and "wrong".
"""

from __future__ import annotations

import base64
import binascii
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from linux_whispr.constants import VERSION
from linux_whispr.server.auth import token_required
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

logger = logging.getLogger(__name__)


@dataclass
class PipelineBundle:
    """The four pipelines + a clock for tests / health uptime.

    All fields default to ``None`` so an empty bundle does not import or build
    anything. :meth:`ensure_*` methods lazily construct the real pipelines on
    first use; tests skip this by passing pre-built mocks.
    """

    dictation: Any = None
    edit: Any = None
    conversation: Any = None
    multimodal: Any = None
    started_at: float = field(default_factory=time.monotonic)

    def ensure_dictation(self) -> Any:
        """Return the dictation pipeline, building it on first call."""
        if self.dictation is None:
            from linux_whispr.config import AppConfig
            from linux_whispr.dictation import DictationPipeline

            self.dictation = DictationPipeline(AppConfig.load())
        return self.dictation

    def ensure_edit(self) -> Any:
        """Return the edit pipeline, building it on first call."""
        if self.edit is None:
            from linux_whispr.config import AppConfig
            from linux_whispr.edit_selection import EditPipeline

            self.edit = EditPipeline(AppConfig.load())
        return self.edit

    def ensure_conversation(self) -> Any:
        """Return the conversation pipeline, building it on first call."""
        if self.conversation is None:
            from linux_whispr.config import AppConfig
            from linux_whispr.conversation import ConversationPipeline

            self.conversation = ConversationPipeline(AppConfig.load())
        return self.conversation

    def ensure_multimodal(self) -> Any:
        """Return the multimodal pipeline, building it on first call."""
        if self.multimodal is None:
            from linux_whispr.config import AppConfig
            from linux_whispr.multimodal import MultimodalPipeline

            self.multimodal = MultimodalPipeline(AppConfig.load())
        return self.multimodal


def _decode_b64(blob: str, *, label: str) -> bytes:
    """Decode a base64 payload, raising HTTP 400 on garbage input.

    Returns the raw bytes so the caller can stream them into a pipeline. Empty
    input is treated as 400 — Pydantic already enforces ``min_length=1`` but
    the explicit check keeps a tight error message for callers that strip
    padding by accident.
    """
    try:
        return base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid base64 for {label}: {exc}",
        ) from exc


def create_app(bundle: PipelineBundle | None = None) -> FastAPI:
    """Build a fresh FastAPI app, optionally wired to a custom bundle.

    Tests pass a bundle filled with mocks; production calls this without args
    and the bundle is created empty (pipelines built on first request).
    """
    pipelines = bundle if bundle is not None else PipelineBundle()

    app = FastAPI(
        title="2Key Multi-Machine API",
        version=VERSION,
        docs_url="/v1/docs",
        redoc_url=None,
    )

    @app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # FastAPI's default would expose the traceback in DEBUG — we keep the
        # message generic so untrusted clients learn nothing about internals.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal error"},
        )

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=VERSION,
            uptime_s=round(time.monotonic() - pipelines.started_at, 3),
        )

    @app.post(
        "/v1/dictate",
        response_model=DictateResponse,
        dependencies=[Depends(token_required)],
    )
    def dictate(payload: DictateRequest) -> DictateResponse:
        wav_bytes = _decode_b64(payload.audio_b64, label="audio_b64")
        logger.info("dictate: %d bytes from %s", len(wav_bytes), payload.platform)
        result = pipelines.ensure_dictation().transcribe(wav_bytes)
        return DictateResponse(
            transcript=getattr(result, "text", "") or "",
            language=getattr(result, "language", "") or "",
            confidence=float(getattr(result, "confidence", 0.0) or 0.0),
        )

    @app.post(
        "/v1/edit",
        response_model=EditResponse,
        dependencies=[Depends(token_required)],
    )
    def edit(payload: EditRequest) -> EditResponse:
        result = pipelines.ensure_edit().edit(payload.selected_text, payload.instruction)
        return EditResponse(
            edited_text=getattr(result, "text", "") or "",
            ok=bool(getattr(result, "ok", False)),
            error=getattr(result, "error", "") or "",
        )

    @app.post(
        "/v1/conversation",
        response_model=ConversationResponse,
        dependencies=[Depends(token_required)],
    )
    def conversation(payload: ConversationRequest) -> ConversationResponse:
        result = pipelines.ensure_conversation().process(payload.transcript)
        # want_audio is intentionally NOT honoured in v1: see SPEC-007 OOS.
        # We keep the field so clients can opt-in once v2 ships TTS-streaming.
        return ConversationResponse(
            reply=getattr(result, "reply", "") or "",
            ok=bool(getattr(result, "ok", False)),
            audio_b64="",
            error=getattr(result, "error", "") or "",
        )

    @app.post(
        "/v1/file-context",
        response_model=FileContextResponse,
        dependencies=[Depends(token_required)],
    )
    def file_context(payload: FileContextRequest) -> FileContextResponse:
        files: list[str] = []
        tmp_path: Path | None = None
        if payload.file_b64:
            raw = _decode_b64(payload.file_b64, label="file_b64")
            suffix = Path(payload.filename or "").suffix or ""
            with tempfile.NamedTemporaryFile(
                prefix="2key-ctx-", suffix=suffix, delete=False
            ) as fh:
                fh.write(raw)
                tmp_path = Path(fh.name)
            files.append(str(tmp_path))

        try:
            result = pipelines.ensure_multimodal().ask(payload.question, files=files or None)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        return FileContextResponse(
            answer=getattr(result, "answer", "") or "",
            ok=bool(getattr(result, "ok", False)),
            used_files=list(getattr(result, "used_files", []) or []),
            error=getattr(result, "error", "") or "",
        )

    return app


# Module-level singleton for `uvicorn linux_whispr.server.api:app`. Lazy w.r.t.
# pipelines (the bundle is empty until the first request hits an endpoint).
app = create_app()
