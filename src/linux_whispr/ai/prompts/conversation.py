"""System + user prompts for the conversation mode (SPEC-004).

The conversation mode is the spoken half of 2Key: the user asks a question and
the reply is read out loud by the TTS backend. That has two consequences for
the prompts here:

* The reply must be *short* — long answers waste the user's time and bloat
  synthesis latency. The system prompt hard-caps replies at 2-3 sentences.
* The reply must be *speakable* — no markdown, no code fences, no bullet lists.
  TTS pronounces every backtick. The system prompt forbids them.

The result still goes through the same OpenAI-compatible backend our edit
mode talks to, so we get free model-swap and offline-test affordances.
"""

from __future__ import annotations

# Bilingual-friendly: 2Key is German-first but Calyie often code-switches into
# English. The hard rules at the bottom are language-agnostic.
CONVERSATION_SYSTEM_PROMPT = """\
Du bist 2Key, ein freundlicher Sprach-Assistent fuer Calyie. Du wirst per \
Tastenkombination aktiviert, der Nutzer spricht eine Frage, und deine Antwort \
wird laut vorgelesen.

Regeln:
- Antworte KURZ — maximal 2 bis 3 Saetze.
- Antworte in derselben Sprache, in der die Frage gestellt wurde.
- KEINE Aufzaehlungen, KEINE Codebloecke, KEINE Markdown-Syntax — alles wird \
vorgelesen.
- KEINE Einleitung wie "Klar, hier ist..." — geh direkt zur Antwort.
- Wenn du etwas nicht weisst, sag das in einem kurzen Satz."""


def build_conversation_prompt(transcript: str) -> str:
    """Wrap the user's transcript into the message we send the LLM.

    The system prompt already explains the role, so the user message is just the
    transcript, trimmed. Keeping this a function (rather than inlining) means a
    future formatter (e.g. injecting time-of-day or context) has one place to
    change.
    """
    return transcript.strip()
