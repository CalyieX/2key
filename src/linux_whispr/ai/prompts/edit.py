"""System + user prompts for the edit-selection mode (SPEC-003).

The edit mode takes a piece of *already selected* text plus a spoken instruction
("translate to English", "fix the spelling and shorten this") and must return the
edited text and nothing else — the result is typed straight back over the user's
selection, so any "Here is the corrected version:" preamble would land in their
document. The system prompt is therefore strict about returning only the result.
"""

from __future__ import annotations

# Kept deliberately blunt and bilingual-friendly: the assistant is Linux/German
# first, but instructions may arrive in any language. The hard rule is the last
# line — only the edited text comes back.
EDIT_SYSTEM_PROMPT = """\
Du bist ein praeziser Text-Editor. Du bekommst einen markierten Text und eine \
Anweisung, wie er bearbeitet werden soll. Wende die Anweisung auf den Text an.

Regeln:
- Gib AUSSCHLIESSLICH den bearbeiteten Text zurueck.
- KEINE Erklaerung, keine Einleitung, kein "Hier ist...", keine Anfuehrungszeichen \
um das Ergebnis.
- Behalte Sprache, Tonfall und Formatierung bei, ausser die Anweisung verlangt \
ausdruecklich eine Aenderung.
- Wenn die Anweisung eine Uebersetzung verlangt, gib nur die Uebersetzung zurueck."""


def build_edit_prompt(selected_text: str, instruction: str) -> str:
    """Build the user message pairing the instruction with the selected text.

    Order matters: the instruction comes first so the model reads *what to do*
    before the (possibly long) text it should do it to.
    """
    return (
        f"Anweisung: {instruction.strip()}\n\n"
        f"Markierter Text:\n{selected_text}"
    )
