"""Tests for the file-context loader (SPEC-005).

Covers PNG/JPG (base64), PDF (gemockt — pymupdf via sys.modules), TXT/MD/JSON,
oversized-reject, missing-file, unknown MIME (fallback text), and the
pymupdf-not-installed path. No real PDFs are read, no network is touched.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import pytest

from linux_whispr.input.file_context import (
    MAX_FILE_BYTES,
    FileContext,
    classify,
    load_file,
)


# --------------------------------------------------------------------------- #
# classify()
# --------------------------------------------------------------------------- #
class TestClassify:
    def test_png_is_image(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "a.png") == "image"

    def test_jpg_is_image(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "a.jpg") == "image"

    def test_jpeg_is_image(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "a.JPEG") == "image"

    def test_pdf(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "doc.pdf") == "pdf"

    def test_txt(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "n.txt") == "text"

    def test_unknown_falls_back_to_text(self, tmp_path: Path) -> None:
        assert classify(tmp_path / "weird.xyz") == "text"


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class TestLoadImage:
    def test_png_returns_base64(self, tmp_path: Path) -> None:
        p = tmp_path / "a.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)

        ctx = load_file(p)

        assert ctx is not None
        assert ctx.kind == "image"
        assert ctx.mime == "image/png"
        assert ctx.image_b64
        assert base64.b64decode(ctx.image_b64) == p.read_bytes()
        assert ctx.text == ""
        assert ctx.bytes_size == p.stat().st_size

    def test_jpg_mime(self, tmp_path: Path) -> None:
        p = tmp_path / "a.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"y" * 8)
        ctx = load_file(p)
        assert ctx is not None
        assert ctx.mime == "image/jpeg"

    def test_jpeg_mime_uppercase(self, tmp_path: Path) -> None:
        p = tmp_path / "a.JPEG"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"y" * 8)
        ctx = load_file(p)
        assert ctx is not None
        assert ctx.mime == "image/jpeg"

    def test_webp_mime(self, tmp_path: Path) -> None:
        p = tmp_path / "a.webp"
        p.write_bytes(b"RIFF....WEBP")
        ctx = load_file(p)
        assert ctx is not None
        assert ctx.mime == "image/webp"


# --------------------------------------------------------------------------- #
# Text files
# --------------------------------------------------------------------------- #
class TestLoadText:
    def test_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "note.txt"
        p.write_text("hello world", encoding="utf-8")

        ctx = load_file(p)
        assert ctx is not None
        assert ctx.kind == "text"
        assert ctx.text == "hello world"
        assert ctx.image_b64 == ""

    def test_md(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("# Hi", encoding="utf-8")
        ctx = load_file(p)
        assert ctx is not None
        assert ctx.text == "# Hi"

    def test_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        ctx = load_file(p)
        assert ctx is not None
        assert "a" in ctx.text

    def test_invalid_utf8_replaced(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        p.write_bytes(b"hello\xff\xfeworld")
        ctx = load_file(p)
        assert ctx is not None
        # errors="replace" gives U+FFFD where bytes are invalid — call survives.
        assert "hello" in ctx.text
        assert "world" in ctx.text

    def test_unknown_extension_falls_back_to_text(self, tmp_path: Path) -> None:
        p = tmp_path / "weird.xyz"
        p.write_text("plain bytes", encoding="utf-8")
        ctx = load_file(p)
        assert ctx is not None
        assert ctx.kind == "text"
        assert ctx.text == "plain bytes"


# --------------------------------------------------------------------------- #
# PDF (pymupdf gemockt)
# --------------------------------------------------------------------------- #
class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakeDoc:
    """Minimal pymupdf doc — iterable + close()."""

    def __init__(self, pages: list[str]) -> None:
        self._pages = [_FakePage(t) for t in pages]
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def close(self) -> None:
        self.closed = True


def _install_fake_fitz(monkeypatch, *, pages: list[str] | None = None, raise_on_open: bool = False) -> None:
    module = types.ModuleType("fitz")

    def fake_open(_path: str):
        if raise_on_open:
            raise RuntimeError("synthetic open failure")
        return _FakeDoc(pages or ["page one text", "page two text"])

    module.open = fake_open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", module)


class TestLoadPdf:
    def test_pdf_extracts_pages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        _install_fake_fitz(monkeypatch, pages=["alpha", "beta"])

        ctx = load_file(p)

        assert ctx is not None
        assert ctx.kind == "pdf-text"
        assert ctx.mime == "application/pdf"
        assert "alpha" in ctx.text
        assert "beta" in ctx.text

    def test_pdf_no_pymupdf_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setitem(sys.modules, "fitz", None)  # forces ImportError

        ctx = load_file(p)
        assert ctx is None

    def test_pdf_open_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        _install_fake_fitz(monkeypatch, raise_on_open=True)

        ctx = load_file(p)
        assert ctx is None

    def test_pdf_extraction_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake")

        class _BrokenPage:
            def get_text(self) -> str:
                raise RuntimeError("synthetic extraction failure")

        class _BrokenDoc:
            def __iter__(self):
                return iter([_BrokenPage()])

            def close(self) -> None:
                pass

        module = types.ModuleType("fitz")
        module.open = lambda _p: _BrokenDoc()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fitz", module)

        ctx = load_file(p)
        assert ctx is None


# --------------------------------------------------------------------------- #
# Guards: missing / oversized / not-a-file
# --------------------------------------------------------------------------- #
class TestGuards:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_file(tmp_path / "does_not_exist.png") is None

    def test_directory_returns_none(self, tmp_path: Path) -> None:
        assert load_file(tmp_path) is None

    def test_oversized_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "big.txt"
        # Write just past the cap — sparse-ish so it doesn't consume real RAM.
        with p.open("wb") as fh:
            fh.seek(MAX_FILE_BYTES + 1)
            fh.write(b"\x00")

        assert load_file(p) is None

    def test_accepts_path_object_or_str(self, tmp_path: Path) -> None:
        p = tmp_path / "n.txt"
        p.write_text("ok", encoding="utf-8")
        assert load_file(p) is not None
        assert load_file(str(p)) is not None


# --------------------------------------------------------------------------- #
# FileContext frozen
# --------------------------------------------------------------------------- #
class TestFileContext:
    def test_is_frozen(self) -> None:
        ctx = FileContext(path="/x", kind="text", mime="text/plain", text="hi")
        with pytest.raises((AttributeError, Exception)):
            ctx.text = "no"  # type: ignore[misc]
