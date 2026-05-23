"""System + content builders for the multimodal mode (SPEC-005).

The multimodal mode answers a spoken question with one or more attached files
as context — image(s), PDF text, or plain text. We hand the LLM a strict
"answer factually, only from what's shown" system prompt and assemble the user
message in the OpenAI multi-modal content-array shape:

    content = [
        {"type": "text", "text": "<question + attached non-image excerpts>"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ...
    ]
"""

from __future__ import annotations

from collections.abc import Iterable

from linux_whispr.input.file_context import FileContext

# Strict-but-friendly. The model gets pictures and/or text snippets and must
# answer only on the basis of what it sees — and own up if the source does not
# carry the answer.
VISION_SYSTEM_PROMPT = """\
Du bist ein praeziser Datei-/Bild-Assistent. Du bekommst eine Frage und einen \
oder mehrere Dateikontexte (Bilder, PDF-Auszuege, Textdateien). Beantworte die \
Frage **ausschliesslich** auf Basis dieser Kontexte.

Regeln:
- Antworte kurz, sachlich, in der Sprache der Frage.
- Wenn der gegebene Kontext die Frage nicht beantwortet, sag das ehrlich \
("dazu sagt das Bild/Dokument nichts").
- Keine Halluzination, keine Quellen erfinden.
- Keine Meta-Einleitung wie "Hier ist die Antwort:" — nur das Ergebnis."""

# How much of a non-image file we embed inline. Keeps token cost predictable;
# whole PDFs can be huge.
_MAX_TEXT_CHARS_PER_FILE = 16_000


def _embed_text_snippet(file: FileContext) -> str:
    """Render a non-image file as a labelled markdown block for the text part."""
    body = file.text or ""
    truncated = ""
    if len(body) > _MAX_TEXT_CHARS_PER_FILE:
        body = body[:_MAX_TEXT_CHARS_PER_FILE]
        truncated = "\n…[abgeschnitten]"
    label = "PDF" if file.kind == "pdf-text" else "Datei"
    return f"### {label}: {file.path}\n```\n{body}{truncated}\n```"


def build_vision_content(
    transcript: str,
    files: Iterable[FileContext],
) -> list[dict]:
    """Build the OpenAI multimodal ``content`` list for the user message.

    Order is intentional: the text block (question + any embedded
    non-image excerpts) comes first so the model reads *what is asked*
    before scanning the image(s).
    """
    file_list = list(files)
    images = [f for f in file_list if f.kind == "image"]
    non_images = [f for f in file_list if f.kind != "image"]

    text_parts: list[str] = [f"Frage: {transcript.strip()}"]
    if non_images:
        text_parts.append("\nKontext-Dateien:")
        text_parts.extend(_embed_text_snippet(f) for f in non_images)

    content: list[dict] = [{"type": "text", "text": "\n".join(text_parts)}]
    for img in images:
        url = f"data:{img.mime};base64,{img.image_b64}"
        content.append({"type": "image_url", "image_url": {"url": url}})

    return content
