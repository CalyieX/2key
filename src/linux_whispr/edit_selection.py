"""Edit-selection pipeline assembly (SPEC-003).

This is the single source of truth for the "2 Key" edit mode: the user selects
text, holds Strg+Super, speaks an instruction, and the highlighted text is
replaced in place by the LLM's edited version.

    PRIMARY selection  ─┐
                        ├─►  LiteLLM (cerebras-qwen, strict "only result" prompt)  ─►  type over selection
    spoken instruction ─┘

Like ``dictation.py``, construction is GUI-free and cheap: it resolves the
selection tool and builds the LLM backend but contacts neither the X server nor
the network until a method is called. The actual keystroke replacement is a
human-test (it needs a focused window); everything up to and including the LLM
call is verified headlessly by the test-suite and ``scripts/edit_smoke.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from linux_whispr.ai.openai_llm import OpenAILLMBackend
from linux_whispr.ai.prompts.edit import EDIT_SYSTEM_PROMPT, build_edit_prompt
from linux_whispr.config import AppConfig
from linux_whispr.output.selection import SelectionCapture
from linux_whispr.platform.detect import PlatformInfo, detect_platform

logger = logging.getLogger(__name__)

# Defaults for the edit brain. Edit mode is wired to our local LiteLLM gateway
# and the free cerebras-qwen model unless config overrides them.
DEFAULT_EDIT_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_EDIT_MODEL = "cerebras-qwen"
DEFAULT_EDIT_API_KEY = "sk-litellm-local"


def resolve_edit_api_key(config: AppConfig) -> str:
    """Resolve the LiteLLM key: env first, then config, then the local default.

    Mirrors how ``app.py`` resolves the refinement key so both paths agree on
    which credential the gateway expects.
    """
    return (
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or config.ai.api_key
        or DEFAULT_EDIT_API_KEY
    )


def build_edit_llm_backend(config: AppConfig) -> OpenAILLMBackend:
    """Create the OpenAI-compatible backend the edit mode talks to.

    Uses the configured LiteLLM endpoint/model when set, otherwise the local
    cerebras-qwen defaults from SPEC-003. The backend is lazy — no network call
    happens here, only object construction.
    """
    base_url = config.ai.base_url or DEFAULT_EDIT_BASE_URL
    model = config.ai.model or DEFAULT_EDIT_MODEL
    api_key = resolve_edit_api_key(config)
    return OpenAILLMBackend(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True)
class EditResult:
    """Outcome of a single edit request.

    Attributes:
        text: The edited text (empty when ``ok`` is False).
        ok: True when the LLM returned usable, non-empty text.
        error: Human-readable reason when ``ok`` is False, else "".
    """

    text: str
    ok: bool
    error: str = ""


class EditPipeline:
    """Headless, GUI-free view of the select → edit → replace path.

    Construction resolves the selection tool and builds the LLM backend but does
    not contact the X server or the network. The heavy/networked work happens in
    :meth:`capture_selection`, :meth:`edit`, and :meth:`replace_selection`.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        platform: PlatformInfo | None = None,
        backend: OpenAILLMBackend | None = None,
        selection: SelectionCapture | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._platform = platform or detect_platform()
        self._selection = selection or SelectionCapture(self._platform)
        self._backend = backend or build_edit_llm_backend(self._config)

    @property
    def selection(self) -> SelectionCapture:
        """The PRIMARY-selection capturer used by this pipeline."""
        return self._selection

    def capture_selection(self) -> str | None:
        """Read the currently highlighted text.

        Returns the selected text, ``""`` when nothing is selected, or ``None``
        when capture is impossible (no display/tool). Callers treat both empty
        and None as "nothing to edit" but can word the message differently.
        """
        return self._selection.read()

    def edit(self, selected_text: str, instruction: str) -> EditResult:
        """Send selected text + instruction to the LLM, return the edited text.

        Edge cases handled here, never raised:
          * empty selection or empty instruction → friendly ``EditResult`` with
            ok=False and no LLM call,
          * LLM unreachable / errors / returns nothing → ok=False with the
            reason, so the caller leaves the selection untouched.
        """
        if not selected_text.strip():
            return EditResult(text="", ok=False, error="no text selected")
        if not instruction.strip():
            return EditResult(text="", ok=False, error="no instruction given")

        user_prompt = build_edit_prompt(selected_text, instruction)

        try:
            result = self._backend.generate(
                system_prompt=EDIT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.exception("Edit LLM call failed")
            return EditResult(text="", ok=False, error=f"LLM error: {exc}")

        edited = result.text.strip()
        if not edited:
            return EditResult(text="", ok=False, error="LLM returned empty text")

        logger.info(
            "Edit complete: %d → %d chars (model=%s, tokens=%d)",
            len(selected_text),
            len(edited),
            result.model,
            result.tokens_used,
        )
        return EditResult(text=edited, ok=True)

    def replace_selection(self, text: str) -> bool:
        """Type ``text`` over the highlighted selection (human-test E2E)."""
        return self._selection.replace(text)

    def run(self, instruction: str) -> EditResult:
        """Full live flow: capture selection → edit → replace it in place.

        This is what the running app calls once STT has produced ``instruction``.
        Returns the ``EditResult``; on success the edited text has already been
        typed over the selection. On any failure the selection is left untouched
        and the reason is in ``EditResult.error``.
        """
        selected = self.capture_selection()
        if selected is None:
            return EditResult(
                text="", ok=False, error=self._selection.unavailable_reason()
            )
        if not selected.strip():
            return EditResult(text="", ok=False, error="no text selected")

        result = self.edit(selected, instruction)
        if not result.ok:
            return result

        if not self.replace_selection(result.text):
            return EditResult(
                text=result.text,
                ok=False,
                error="edit succeeded but replacing the selection failed",
            )
        return result
