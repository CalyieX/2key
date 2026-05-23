"""Tests for the TTS backends (SPEC-004).

Covers the auto-checkable acceptance criteria without touching audio hardware,
the cosy2-eu binary, or the network:

  * the Protocol / NullBackend contract (always available, logs silently),
  * CosyVoiceBackend availability checks (missing binary, missing prompt),
  * the synth + play paths (subprocess.run is mocked so no real CLI runs),
  * every error path returns False rather than raising.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from linux_whispr.output import tts as tts_mod
from linux_whispr.output.tts import (
    DEFAULT_COSY_BINARY,
    DEFAULT_COSY_PROMPT,
    CosyVoiceBackend,
    NullBackend,
    TtsBackend,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> object:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------------- #
# Protocol contract
# --------------------------------------------------------------------------- #
class TestTtsBackendProtocol:
    def test_null_backend_is_tts_backend(self) -> None:
        assert isinstance(NullBackend(), TtsBackend)

    def test_cosy_backend_is_tts_backend(self) -> None:
        assert isinstance(CosyVoiceBackend(), TtsBackend)


# --------------------------------------------------------------------------- #
# NullBackend
# --------------------------------------------------------------------------- #
class TestNullBackend:
    def test_always_available(self) -> None:
        assert NullBackend().is_available() is True

    def test_speak_returns_true_for_text(self) -> None:
        assert NullBackend().speak("Hallo Welt") is True

    def test_speak_returns_false_for_empty(self) -> None:
        assert NullBackend().speak("") is False

    def test_speak_returns_false_for_whitespace(self) -> None:
        assert NullBackend().speak("   \n  ") is False

    def test_speak_does_not_call_subprocess(self, monkeypatch) -> None:
        called: list[object] = []

        def _fail(*a, **k):
            called.append(a)
            return _completed()

        monkeypatch.setattr(tts_mod.subprocess, "run", _fail)
        NullBackend().speak("Hallo")
        assert called == []

    def test_has_name(self) -> None:
        assert NullBackend().name == "null"


# --------------------------------------------------------------------------- #
# CosyVoiceBackend — availability
# --------------------------------------------------------------------------- #
def _exec_binary(tmp_path: Path, name: str = "cosy2-eu") -> Path:
    """Create an executable placeholder binary inside ``tmp_path``."""
    binary = tmp_path / name
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def _prompt_wav(tmp_path: Path, name: str = "prompt-de.wav") -> Path:
    """Create a non-empty placeholder prompt WAV inside ``tmp_path``."""
    wav = tmp_path / name
    wav.write_bytes(b"RIFF....WAVE")
    return wav


class TestCosyAvailability:
    def test_default_paths_used(self) -> None:
        backend = CosyVoiceBackend()
        assert DEFAULT_COSY_BINARY.endswith("cosy2-eu")
        assert DEFAULT_COSY_PROMPT.endswith(".wav")
        # Defaults exist or don't exist on this host, but availability must
        # always return a boolean, never raise.
        assert isinstance(backend.is_available(), bool)

    def test_available_when_binary_and_prompt_exist(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(_exec_binary(tmp_path)),
            prompt_wav=str(_prompt_wav(tmp_path)),
        )
        assert backend.is_available() is True

    def test_unavailable_when_binary_missing(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(tmp_path / "nope"),
            prompt_wav=str(_prompt_wav(tmp_path)),
        )
        assert backend.is_available() is False

    def test_unavailable_when_binary_not_executable(self, tmp_path) -> None:
        binary = tmp_path / "cosy2-eu"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o644)
        backend = CosyVoiceBackend(
            binary_path=str(binary),
            prompt_wav=str(_prompt_wav(tmp_path)),
        )
        assert backend.is_available() is False

    def test_unavailable_when_prompt_missing(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(_exec_binary(tmp_path)),
            prompt_wav=str(tmp_path / "no.wav"),
        )
        assert backend.is_available() is False


# --------------------------------------------------------------------------- #
# CosyVoiceBackend — argv builders
# --------------------------------------------------------------------------- #
class TestCosyArgvBuilders:
    def test_synth_argv_includes_final_flag_by_default(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(_exec_binary(tmp_path)),
            prompt_wav=str(_prompt_wav(tmp_path)),
        )
        argv = backend._build_synth_argv("Hallo", str(tmp_path / "out.wav"))
        assert argv[0].endswith("cosy2-eu")
        assert "--text" in argv and "Hallo" in argv
        assert "--prompt" in argv
        assert "--out" in argv
        assert "--final" in argv

    def test_synth_argv_drops_final_flag_when_disabled(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(_exec_binary(tmp_path)),
            prompt_wav=str(_prompt_wav(tmp_path)),
            use_final=False,
        )
        argv = backend._build_synth_argv("Hallo", str(tmp_path / "out.wav"))
        assert "--final" not in argv

    def test_player_argv_paplay_takes_plain_args(self) -> None:
        backend = CosyVoiceBackend()
        argv = backend._build_player_argv("paplay", "/tmp/foo.wav")
        assert argv == ["paplay", "/tmp/foo.wav"]

    def test_player_argv_aplay_takes_plain_args(self) -> None:
        backend = CosyVoiceBackend()
        argv = backend._build_player_argv("aplay", "/tmp/foo.wav")
        assert argv == ["aplay", "/tmp/foo.wav"]

    def test_player_argv_ffplay_gets_loop_friendly_flags(self) -> None:
        backend = CosyVoiceBackend()
        argv = backend._build_player_argv("ffplay", "/tmp/foo.wav")
        assert "ffplay" in argv
        assert "-nodisp" in argv
        assert "-autoexit" in argv
        assert "/tmp/foo.wav" in argv


# --------------------------------------------------------------------------- #
# CosyVoiceBackend — speak path (subprocess + filesystem fully mocked)
# --------------------------------------------------------------------------- #
@pytest.fixture
def cosy(tmp_path):
    """A CosyVoiceBackend wired against placeholder binary + prompt + output dir."""
    return CosyVoiceBackend(
        binary_path=str(_exec_binary(tmp_path)),
        prompt_wav=str(_prompt_wav(tmp_path)),
        output_dir=str(tmp_path),
    )


class TestCosySpeak:
    def test_speak_refuses_empty_text(self, cosy) -> None:
        assert cosy.speak("") is False
        assert cosy.speak("   ") is False

    def test_speak_returns_false_when_unavailable(self, tmp_path) -> None:
        backend = CosyVoiceBackend(
            binary_path=str(tmp_path / "missing"),
            prompt_wav=str(tmp_path / "missing.wav"),
            output_dir=str(tmp_path),
        )
        assert backend.speak("Hallo") is False

    def test_speak_happy_path_runs_synth_then_play(self, cosy, monkeypatch) -> None:
        synth_argv: list[list[str]] = []
        play_argv: list[list[str]] = []

        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                synth_argv.append(args)
                out_idx = args.index("--out") + 1
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            play_argv.append(args)
            return _completed(0)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

        assert cosy.speak("Hallo") is True
        assert synth_argv and "--final" in synth_argv[0]
        assert play_argv and play_argv[0][0] == "paplay"

    def test_speak_returns_false_when_synth_fails(self, cosy, monkeypatch) -> None:
        def _fake_run(args, *a, **k):
            return _completed(1, stderr="bad")

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_synth_binary_missing(
        self, cosy, monkeypatch
    ) -> None:
        def _raise(*a, **k):
            raise FileNotFoundError

        monkeypatch.setattr(tts_mod.subprocess, "run", _raise)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_synth_times_out(self, cosy, monkeypatch) -> None:
        def _raise(*a, **k):
            raise subprocess.TimeoutExpired(cmd="cosy2-eu", timeout=60)

        monkeypatch.setattr(tts_mod.subprocess, "run", _raise)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_output_missing(self, cosy, monkeypatch) -> None:
        def _fake_run(args, *a, **k):
            # Pretend the CLI exited cleanly but never wrote the file.
            return _completed(0)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_no_player(self, cosy, monkeypatch) -> None:
        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                out_idx = args.index("--out") + 1
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            return _completed(0)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda _name: None)
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_player_fails(self, cosy, monkeypatch) -> None:
        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                out_idx = args.index("--out") + 1
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            return _completed(1, stderr="player blew up")

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False

    def test_speak_returns_false_when_player_times_out(
        self, cosy, monkeypatch
    ) -> None:
        synth_calls: list[int] = []

        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                synth_calls.append(1)
                out_idx = args.index("--out") + 1
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            raise subprocess.TimeoutExpired(cmd="paplay", timeout=30)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is False
        assert synth_calls == [1]

    def test_speak_picks_first_available_player(self, cosy, monkeypatch) -> None:
        seen_players: list[str] = []

        def _fake_which(name):
            # Only aplay is on PATH — paplay is missing.
            return f"/usr/bin/{name}" if name == "aplay" else None

        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                out_idx = args.index("--out") + 1
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            seen_players.append(args[0])
            return _completed(0)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", _fake_which)
        assert cosy.speak("Hallo") is True
        assert seen_players == ["aplay"]

    def test_speak_cleans_up_temp_file(self, cosy, monkeypatch) -> None:
        seen_outputs: list[str] = []

        def _fake_run(args, *a, **k):
            if args[0].endswith("cosy2-eu"):
                out_idx = args.index("--out") + 1
                seen_outputs.append(args[out_idx])
                Path(args[out_idx]).write_bytes(b"WAV-bytes")
                return _completed(0)
            return _completed(0)

        monkeypatch.setattr(tts_mod.subprocess, "run", _fake_run)
        monkeypatch.setattr(tts_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert cosy.speak("Hallo") is True
        assert seen_outputs and not Path(seen_outputs[0]).exists()

    def test_custom_backend_can_be_swapped_via_protocol(self) -> None:
        class Loud:
            name = "loud"

            def is_available(self) -> bool:
                return True

            def speak(self, text: str) -> bool:
                return bool(text)

        loud = Loud()
        assert isinstance(loud, TtsBackend)
        assert loud.speak("hi") is True


# --------------------------------------------------------------------------- #
# Construction stays cheap and tolerant of missing binaries on host.
# --------------------------------------------------------------------------- #
class TestCheapConstruction:
    def test_constructor_does_not_call_subprocess(self, monkeypatch) -> None:
        def _fail(*a, **k):
            pytest.fail("constructor should not invoke subprocess")

        monkeypatch.setattr(tts_mod.subprocess, "run", _fail)
        CosyVoiceBackend()

    def test_constructor_accepts_full_override(self) -> None:
        backend = CosyVoiceBackend(
            binary_path="/x/cosy",
            prompt_wav="/x/p.wav",
            output_dir="/x",
            players=("paplay",),
            synth_timeout=10,
            play_timeout=5,
            use_final=False,
        )
        argv = backend._build_synth_argv("hi", "/tmp/out.wav")
        assert argv[0] == "/x/cosy"
        assert "/x/p.wav" in argv
        assert "--final" not in argv


def test_magicmock_satisfies_protocol() -> None:
    backend = MagicMock(spec=TtsBackend)
    assert isinstance(backend, TtsBackend)
