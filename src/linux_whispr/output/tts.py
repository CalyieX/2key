"""Text-to-speech backends for the conversation mode (SPEC-004).

The conversation mode needs a way to read the assistant's reply out loud while
staying portable across machines that do not have a TTS engine installed. The
backend protocol here keeps that swap cheap: every backend exposes the same
two methods, so :mod:`linux_whispr.conversation` can hand off the reply without
caring whether anyone is actually listening.

Two backends ship with 2Key:

* :class:`NullBackend` — default. Logs the spoken text and returns ``True``,
  produces no sound. Safe in tests, on headless machines, and as the
  out-of-the-box choice so users do not get surprise audio on first run.
* :class:`CosyVoiceBackend` — drives the ``cosy2-eu`` CLI wrapper that ships on
  wonderland. Synthesises to a temp WAV and plays it via paplay/aplay/ffplay,
  whichever is on PATH. Every external call is wrapped: if anything fails
  (binary missing, model not downloaded, timeout, player crash) we log and
  return ``False`` rather than raising — the conversation never explodes
  because TTS hiccuped.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Defaults that match the wonderland install. Callers can override every field.
DEFAULT_COSY_BINARY = "/home/calyie/bin/cosy2-eu"
# Standard German prompt-wav that ships with sunny-voice. CosyVoice copies the
# voice characteristics from this clip — it is intentionally short and clean.
DEFAULT_COSY_PROMPT = "/home/calyie/.cache/cosyvoice2-eu/prompt-de.wav"
# Players we try in order. paplay is on every PulseAudio system, aplay on ALSA,
# ffplay is the universal fallback that comes with ffmpeg.
DEFAULT_PLAYERS = ("paplay", "aplay", "ffplay")
# Synthesis can be slow on CPU; 60s is generous for the 2-3 sentence replies
# the conversation prompt asks for.
DEFAULT_SYNTH_TIMEOUT = 60
# Playback shouldn't outlive the WAV by much. 30s caps a runaway player.
DEFAULT_PLAY_TIMEOUT = 30


@runtime_checkable
class TtsBackend(Protocol):
    """Minimal contract every TTS backend implements.

    ``speak`` returns True when audio was emitted (or, for ``NullBackend``, when
    the text was logged successfully). False means "could not speak" — never an
    exception, so callers can react without try/except.
    """

    def is_available(self) -> bool: ...

    def speak(self, text: str) -> bool: ...


class NullBackend:
    """Default backend — silent, dependency-free, always available.

    Logs the requested text and returns True. The conversation pipeline uses
    this when the user has not opted into a real TTS engine yet, and the
    test-suite uses it to avoid hitting any binaries.
    """

    name = "null"

    def is_available(self) -> bool:
        return True

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False
        logger.info("NullBackend would speak: %s", text)
        return True


class CosyVoiceBackend:
    """Speak ``text`` by invoking the ``cosy2-eu`` CLI and playing the result.

    The backend is constructed cheaply (no subprocess until :meth:`speak`).
    :meth:`is_available` only inspects the filesystem and PATH — useful for
    smoke tests on machines that lack the model files.
    """

    name = "cosy"

    def __init__(
        self,
        *,
        binary_path: str = DEFAULT_COSY_BINARY,
        prompt_wav: str = DEFAULT_COSY_PROMPT,
        output_dir: str | None = None,
        players: tuple[str, ...] = DEFAULT_PLAYERS,
        synth_timeout: int = DEFAULT_SYNTH_TIMEOUT,
        play_timeout: int = DEFAULT_PLAY_TIMEOUT,
        use_final: bool = True,
    ) -> None:
        self._binary = binary_path
        self._prompt = prompt_wav
        self._output_dir = output_dir
        self._players = players
        self._synth_timeout = synth_timeout
        self._play_timeout = play_timeout
        self._use_final = use_final

    def is_available(self) -> bool:
        """Both the CLI wrapper and the prompt wav must exist."""
        if not Path(self._binary).is_file():
            return False
        if not os.access(self._binary, os.X_OK):
            return False
        if not Path(self._prompt).is_file():
            return False
        return True

    def _pick_player(self) -> str | None:
        for player in self._players:
            if shutil.which(player) is not None:
                return player
        return None

    def _build_synth_argv(self, text: str, out_path: str) -> list[str]:
        argv = [
            self._binary,
            "--text",
            text,
            "--prompt",
            self._prompt,
            "--out",
            out_path,
        ]
        if self._use_final:
            argv.append("--final")
        return argv

    def _build_player_argv(self, player: str, wav_path: str) -> list[str]:
        if player == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", wav_path]
        return [player, wav_path]

    def _synthesize(self, text: str, out_path: str) -> bool:
        try:
            result = subprocess.run(
                self._build_synth_argv(text, out_path),
                capture_output=True,
                text=True,
                timeout=self._synth_timeout,
            )
        except FileNotFoundError:
            logger.error("CosyVoice binary not found: %s", self._binary)
            return False
        except subprocess.TimeoutExpired:
            logger.error("CosyVoice synthesis timed out after %ds", self._synth_timeout)
            return False

        if result.returncode != 0:
            logger.error(
                "CosyVoice synthesis failed (rc=%d): %s",
                result.returncode,
                (result.stderr or "").strip(),
            )
            return False

        if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
            logger.error("CosyVoice produced no output file at %s", out_path)
            return False
        return True

    def _play(self, wav_path: str) -> bool:
        player = self._pick_player()
        if player is None:
            logger.error(
                "No audio player available (tried %s)", ", ".join(self._players)
            )
            return False

        try:
            result = subprocess.run(
                self._build_player_argv(player, wav_path),
                capture_output=True,
                text=True,
                timeout=self._play_timeout,
            )
        except FileNotFoundError:
            logger.error("Audio player '%s' not found", player)
            return False
        except subprocess.TimeoutExpired:
            logger.error("Audio player '%s' timed out", player)
            return False

        if result.returncode != 0:
            logger.error(
                "Audio player '%s' failed (rc=%d): %s",
                player,
                result.returncode,
                (result.stderr or "").strip(),
            )
            return False
        return True

    def speak(self, text: str) -> bool:
        """Synthesise + play. Returns True only when both steps succeeded."""
        if not text.strip():
            return False
        if not self.is_available():
            logger.warning(
                "CosyVoiceBackend unavailable (binary=%s, prompt=%s)",
                self._binary,
                self._prompt,
            )
            return False

        with tempfile.NamedTemporaryFile(
            prefix="2key-tts-",
            suffix=".wav",
            dir=self._output_dir,
            delete=False,
        ) as handle:
            out_path = handle.name

        try:
            if not self._synthesize(text, out_path):
                return False
            return self._play(out_path)
        finally:
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not clean up TTS temp file %s", out_path)
