"""Tests for the FastAPI multi-machine server (SPEC-007).

Uses ``fastapi.testclient.TestClient`` against a ``create_app`` instance wired
with mock pipelines. No real STT, LLM, TTS or network is touched - the four
pipelines are MagicMocks whose return shape mirrors the real result objects.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from linux_whispr.server.api import PipelineBundle, create_app
from linux_whispr.server.auth import DEFAULT_TOKEN


@dataclass
class FakeDictateResult:
    text: str
    language: str = "en"
    confidence: float = 0.9


@dataclass
class FakeEditResult:
    text: str
    ok: bool
    error: str = ""


@dataclass
class FakeConvResult:
    reply: str
    ok: bool
    spoken: bool = False
    error: str = ""
    transcript: str = ""


@dataclass
class FakeAskResult:
    answer: str
    ok: bool
    used_files: list
    error: str = ""


class _StubPipeline:
    """Generic stub with method->result map for the per-mode test client."""

    def __init__(self, method: str, result: Any) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self._method = method
        self._result = result

        def _fn(*args, **kwargs):
            self.calls.append((args, kwargs))
            return self._result

        setattr(self, method, _fn)


@pytest.fixture
def env_token(monkeypatch) -> str:
    monkeypatch.setenv("TWO_KEY_TOKEN", "test-secret-abc")
    return "test-secret-abc"


def _build_client(
    *,
    dictation: Any = None,
    edit: Any = None,
    conversation: Any = None,
    multimodal: Any = None,
) -> TestClient:
    bundle = PipelineBundle(
        dictation=dictation, edit=edit, conversation=conversation, multimodal=multimodal
    )
    return TestClient(create_app(bundle))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------- #
# /v1/health
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_health_no_auth_required(self) -> None:
        client = _build_client()
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert body["uptime_s"] >= 0.0

    def test_health_works_even_with_wrong_token(self) -> None:
        client = _build_client()
        resp = client.get("/v1/health", headers={"X-2Key-Token": "garbage"})
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /v1/dictate
# --------------------------------------------------------------------------- #
class TestDictate:
    def test_returns_transcript_on_ok(self, env_token) -> None:
        stub = _StubPipeline("transcribe", FakeDictateResult(text="hallo welt"))
        client = _build_client(dictation=stub)
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": env_token},
            json={"audio_b64": _b64(b"riffwave"), "platform": "linux"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["transcript"] == "hallo welt"
        assert body["language"] == "en"
        assert body["confidence"] == pytest.approx(0.9)
        assert stub.calls and stub.calls[0][0][0] == b"riffwave"

    def test_missing_token_is_401(self, env_token) -> None:
        client = _build_client(dictation=_StubPipeline("transcribe", FakeDictateResult("")))
        resp = client.post("/v1/dictate", json={"audio_b64": _b64(b"x")})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "invalid token"

    def test_wrong_token_is_401(self, env_token) -> None:
        client = _build_client(dictation=_StubPipeline("transcribe", FakeDictateResult("")))
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": "wrong"},
            json={"audio_b64": _b64(b"x")},
        )
        assert resp.status_code == 401

    def test_invalid_base64_is_400(self, env_token) -> None:
        client = _build_client(dictation=_StubPipeline("transcribe", FakeDictateResult("")))
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": env_token},
            json={"audio_b64": "!!not-base64!!"},
        )
        assert resp.status_code == 400
        assert "base64" in resp.json()["detail"].lower()

    def test_missing_audio_field_is_422(self, env_token) -> None:
        client = _build_client(dictation=_StubPipeline("transcribe", FakeDictateResult("")))
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": env_token},
            json={"platform": "linux"},
        )
        assert resp.status_code == 422

    def test_empty_audio_string_is_422(self, env_token) -> None:
        client = _build_client(dictation=_StubPipeline("transcribe", FakeDictateResult("")))
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": env_token},
            json={"audio_b64": ""},
        )
        assert resp.status_code == 422

    def test_uses_default_token_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("TWO_KEY_TOKEN", raising=False)
        stub = _StubPipeline("transcribe", FakeDictateResult("hi"))
        client = _build_client(dictation=stub)
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": DEFAULT_TOKEN},
            json={"audio_b64": _b64(b"x")},
        )
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /v1/edit
# --------------------------------------------------------------------------- #
class TestEdit:
    def test_happy_path(self, env_token) -> None:
        stub = _StubPipeline("edit", FakeEditResult(text="HELLO", ok=True))
        client = _build_client(edit=stub)
        resp = client.post(
            "/v1/edit",
            headers={"X-2Key-Token": env_token},
            json={"selected_text": "hello", "instruction": "uppercase"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["edited_text"] == "HELLO"
        assert body["ok"] is True
        assert body["error"] == ""
        assert stub.calls[0][0] == ("hello", "uppercase")

    def test_llm_failure_returns_200_with_ok_false(self, env_token) -> None:
        stub = _StubPipeline("edit", FakeEditResult(text="", ok=False, error="LLM down"))
        client = _build_client(edit=stub)
        resp = client.post(
            "/v1/edit",
            headers={"X-2Key-Token": env_token},
            json={"selected_text": "hi", "instruction": "shout"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["error"] == "LLM down"

    def test_missing_token_is_401(self) -> None:
        client = _build_client(edit=_StubPipeline("edit", FakeEditResult("", False)))
        resp = client.post(
            "/v1/edit",
            json={"selected_text": "hi", "instruction": "x"},
        )
        assert resp.status_code == 401

    def test_empty_fields_are_422(self, env_token) -> None:
        client = _build_client(edit=_StubPipeline("edit", FakeEditResult("", False)))
        resp = client.post(
            "/v1/edit",
            headers={"X-2Key-Token": env_token},
            json={"selected_text": "", "instruction": ""},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /v1/conversation
# --------------------------------------------------------------------------- #
class TestConversation:
    def test_happy_path(self, env_token) -> None:
        stub = _StubPipeline("process", FakeConvResult(reply="42", ok=True))
        client = _build_client(conversation=stub)
        resp = client.post(
            "/v1/conversation",
            headers={"X-2Key-Token": env_token},
            json={"transcript": "wieviel ist 6 mal 7"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "42"
        assert body["ok"] is True
        assert body["audio_b64"] == ""
        assert stub.calls[0][0] == ("wieviel ist 6 mal 7",)

    def test_failure_returns_200_with_error(self, env_token) -> None:
        stub = _StubPipeline(
            "process", FakeConvResult(reply="", ok=False, error="no transcript")
        )
        client = _build_client(conversation=stub)
        resp = client.post(
            "/v1/conversation",
            headers={"X-2Key-Token": env_token},
            json={"transcript": "x"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["error"] == "no transcript"

    def test_missing_token_is_401(self) -> None:
        client = _build_client(conversation=_StubPipeline("process", FakeConvResult("", False)))
        resp = client.post("/v1/conversation", json={"transcript": "hi"})
        assert resp.status_code == 401

    def test_empty_transcript_is_422(self, env_token) -> None:
        client = _build_client(conversation=_StubPipeline("process", FakeConvResult("", False)))
        resp = client.post(
            "/v1/conversation",
            headers={"X-2Key-Token": env_token},
            json={"transcript": ""},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /v1/file-context
# --------------------------------------------------------------------------- #
class TestFileContext:
    def test_text_only_call(self, env_token) -> None:
        stub = _StubPipeline("ask", FakeAskResult(answer="42", ok=True, used_files=[]))
        client = _build_client(multimodal=stub)
        resp = client.post(
            "/v1/file-context",
            headers={"X-2Key-Token": env_token},
            json={"question": "what is the answer"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "42"
        assert body["ok"] is True
        assert body["used_files"] == []
        args, kwargs = stub.calls[0]
        assert args == ("what is the answer",)
        assert kwargs == {"files": None}

    def test_with_file_writes_temp_and_passes_path(self, env_token) -> None:
        captured: dict = {}

        def capture(question, files=None):
            # Snapshot whatever the server passed in; the temp path must exist
            # *now* (before the endpoint cleans it up).
            captured["question"] = question
            captured["files"] = list(files or [])
            captured["readable"] = all(
                __import__("os").path.exists(p) for p in (files or [])
            )
            return FakeAskResult(answer="img is a cat", ok=True, used_files=list(files or []))

        stub = type("S", (), {"ask": staticmethod(capture)})()
        client = _build_client(multimodal=stub)
        resp = client.post(
            "/v1/file-context",
            headers={"X-2Key-Token": env_token},
            json={
                "question": "what is on this image",
                "file_b64": _b64(b"\x89PNG\r\n\x1a\n"),
                "mime": "image/png",
                "filename": "cat.png",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "img is a cat"
        assert captured["readable"] is True
        # And the temp file is unlinked after the handler returns.
        for p in captured["files"]:
            assert not __import__("os").path.exists(p)

    def test_missing_token_is_401(self) -> None:
        client = _build_client(multimodal=_StubPipeline("ask", FakeAskResult("", False, [])))
        resp = client.post("/v1/file-context", json={"question": "hi"})
        assert resp.status_code == 401

    def test_invalid_file_b64_is_400(self, env_token) -> None:
        client = _build_client(multimodal=_StubPipeline("ask", FakeAskResult("", False, [])))
        resp = client.post(
            "/v1/file-context",
            headers={"X-2Key-Token": env_token},
            json={"question": "hi", "file_b64": "!!nope!!", "mime": "image/png"},
        )
        assert resp.status_code == 400

    def test_empty_question_is_422(self, env_token) -> None:
        client = _build_client(multimodal=_StubPipeline("ask", FakeAskResult("", False, [])))
        resp = client.post(
            "/v1/file-context",
            headers={"X-2Key-Token": env_token},
            json={"question": ""},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Global error handling + lazy bundle behaviour
# --------------------------------------------------------------------------- #
class TestGlobalBehaviour:
    def test_pipeline_exception_yields_500(self, env_token) -> None:
        class Boom:
            def transcribe(self, *_a, **_kw):
                raise RuntimeError("kaboom")

        bundle = PipelineBundle(dictation=Boom())
        # raise_server_exceptions=False -> TestClient delivers the error
        # response instead of re-raising, matching real over-the-wire behaviour.
        client = TestClient(create_app(bundle), raise_server_exceptions=False)
        resp = client.post(
            "/v1/dictate",
            headers={"X-2Key-Token": env_token},
            json={"audio_b64": _b64(b"x")},
        )
        assert resp.status_code == 500
        # We deliberately mask internals - no traceback in the body.
        assert "kaboom" not in resp.text
        assert resp.json() == {"detail": "internal error"}

    def test_create_app_with_no_bundle_is_safe(self) -> None:
        # The default-bundle path must NOT construct pipelines at import time.
        app = create_app()
        client = TestClient(app)
        # Only health is safe to call without env wiring; it must not need any
        # pipeline construction.
        assert client.get("/v1/health").status_code == 200

    def test_unknown_route_is_404(self) -> None:
        client = _build_client()
        resp = client.get("/v1/does-not-exist")
        assert resp.status_code == 404

    def test_openapi_docs_available(self) -> None:
        client = _build_client()
        resp = client.get("/v1/docs")
        assert resp.status_code == 200
