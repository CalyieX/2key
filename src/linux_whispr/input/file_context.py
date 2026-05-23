"""File-context loader for the multimodal mode (SPEC-005).

Turns a path on disk into a :class:`FileContext` the multimodal pipeline can
attach to an LLM call — base64 for images, extracted text for PDFs (via the
*lazy*-imported ``pymupdf``/``fitz``), raw UTF-8 for plain text.

Design notes:
  * Single entry point :func:`load_file` — returns ``None`` on every failure
    path so callers never have to catch.
  * ``pymupdf`` is imported *inside* :func:`_load_pdf` so a headless box without
    the library can still load images and text files.
  * A 10 MB cap protects the LLM token budget; oversized files are rejected
    (with a warning) instead of getting silently truncated.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB — covers screenshots/docs, caps tokens.

_IMAGE_EXTS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_TEXT_EXTS: set[str] = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".sh",
    ".bash",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".csv",
    ".log",
}


@dataclass(frozen=True)
class FileContext:
    """One piece of file content prepared for an LLM call.

    Exactly one of ``image_b64`` / ``text`` carries the payload, depending on
    ``kind``. ``mime`` is filled for image kinds so the data-URL has the right
    media type.
    """

    path: str
    kind: str  # "image" | "pdf-text" | "text"
    mime: str
    text: str = ""
    image_b64: str = ""
    bytes_size: int = 0


def classify(path: Path) -> str:
    """Return ``"image"`` / ``"pdf"`` / ``"text"`` for ``path`` by extension.

    Unknown extensions fall back to ``"text"`` — callers attempt a UTF-8 read
    and gracefully bail out if the bytes are not decodable.
    """
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    return "text"


def _load_image(path: Path) -> FileContext | None:
    mime = _IMAGE_EXTS.get(path.suffix.lower(), "application/octet-stream")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read image %s: %s", path, exc)
        return None
    encoded = base64.b64encode(raw).decode("ascii")
    return FileContext(
        path=str(path),
        kind="image",
        mime=mime,
        image_b64=encoded,
        bytes_size=len(raw),
    )


def _load_pdf(path: Path) -> FileContext | None:
    """Extract text from a PDF using *lazy*-imported pymupdf.

    pymupdf is heavy (and optional). Importing here keeps headless boxes / test
    runs that never touch PDFs from paying that cost — and from crashing when
    the library is not installed.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.warning("PDF requested but pymupdf not installed: %s", exc)
        return None

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # pymupdf raises subclasses of RuntimeError
        logger.warning("Cannot open PDF %s: %s", path, exc)
        return None

    try:
        pages = [page.get_text() for page in doc]
    except Exception as exc:
        logger.warning("PDF text extraction failed for %s: %s", path, exc)
        doc.close()
        return None
    finally:
        # ``doc.close()`` is safe to call twice — pymupdf marks itself closed.
        try:
            doc.close()
        except Exception:
            pass

    text = "\n\n".join(pages).strip()
    return FileContext(
        path=str(path),
        kind="pdf-text",
        mime="application/pdf",
        text=text,
        bytes_size=path.stat().st_size,
    )


def _load_text(path: Path) -> FileContext | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read text file %s: %s", path, exc)
        return None
    return FileContext(
        path=str(path),
        kind="text",
        mime="text/plain",
        text=text,
        bytes_size=path.stat().st_size,
    )


def load_file(path: str | Path) -> FileContext | None:
    """Load ``path`` into a :class:`FileContext`.

    Returns ``None`` (with a logged warning) when the file is missing, too
    large, or unreadable. Never raises — callers can build a list of contexts
    by filtering out None.
    """
    p = Path(path)
    if not p.is_file():
        logger.warning("File not found or not a regular file: %s", p)
        return None

    try:
        size = p.stat().st_size
    except OSError as exc:
        logger.warning("Cannot stat %s: %s", p, exc)
        return None
    if size > MAX_FILE_BYTES:
        logger.warning(
            "File %s is %d bytes, over the %d-byte cap — refusing.",
            p,
            size,
            MAX_FILE_BYTES,
        )
        return None

    kind = classify(p)
    if kind == "image":
        return _load_image(p)
    if kind == "pdf":
        return _load_pdf(p)
    return _load_text(p)
