"""ElevenLabs TTS — low-latency streaming + avatar sync."""
from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from config import Config

if TYPE_CHECKING:
    from avatar import AvatarPlayer


# Split on sentence boundaries for early first-audio
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


class TTSEngine:
    def __init__(self, cfg: Config, avatar: Optional["AvatarPlayer"] = None):
        self.cfg = cfg
        self.avatar = avatar
        self._client = None
        self.enabled = bool(cfg.elevenlabs_api_key) and cfg.tts_enabled

    def _get_client(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.cfg.elevenlabs_api_key)
        return self._client

    def _stream_kwargs(self, text: str) -> dict:
        """Args tuned for minimum time-to-first-audio."""
        return {
            "text": text[:2500],
            "voice_id": self.cfg.elevenlabs_voice_id,
            "model_id": self.cfg.elevenlabs_model_id,  # prefer eleven_flash_v2_5
            "output_format": self.cfg.tts_output_format,  # mp3_22050_32 is faster
            "optimize_streaming_latency": int(self.cfg.tts_optimize_latency),  # 0-4
        }

    def synthesize(self, text: str, out_path: Optional[Path] = None) -> Optional[Path]:
        """Full file synthesis (used when Unity/file backend needs a path)."""
        if not self.enabled or not text.strip():
            return None
        try:
            client = self._get_client()
            kwargs = self._stream_kwargs(text)
            # Prefer stream API then write — still lower latency start on server
            try:
                audio = client.text_to_speech.stream(**kwargs)
            except TypeError:
                # older SDK may not accept optimize_streaming_latency
                kwargs.pop("optimize_streaming_latency", None)
                try:
                    audio = client.text_to_speech.stream(**kwargs)
                except Exception:
                    audio = client.text_to_speech.convert(**kwargs)

            if out_path is None:
                fd, name = tempfile.mkstemp(suffix=".mp3", prefix="ollama-tts-")
                out_path = Path(name)
                import os
                os.close(fd)

            with open(out_path, "wb") as f:
                for chunk in audio:
                    if isinstance(chunk, bytes):
                        f.write(chunk)
            return out_path
        except Exception as e:
            print(f"[TTS error] {e}")
            return None

    def iter_audio_chunks(self, text: str) -> Iterator[bytes]:
        if not self.enabled or not text.strip():
            return
        client = self._get_client()
        kwargs = self._stream_kwargs(text)
        try:
            stream = client.text_to_speech.stream(**kwargs)
        except TypeError:
            kwargs.pop("optimize_streaming_latency", None)
            stream = client.text_to_speech.stream(**kwargs)
        for chunk in stream:
            if isinstance(chunk, bytes) and chunk:
                yield chunk

    def play_stream(self, text: str) -> bool:
        """Stream TTS and play ASAP (mpv / elevenlabs.stream / ffplay fallback)."""
        if not self.enabled or not text.strip():
            return False

        # 1) Official helper (uses mpv) — lowest glue latency
        try:
            from elevenlabs import stream as el_stream
            client = self._get_client()
            kwargs = self._stream_kwargs(text)
            try:
                audio_stream = client.text_to_speech.stream(**kwargs)
            except TypeError:
                kwargs.pop("optimize_streaming_latency", None)
                audio_stream = client.text_to_speech.stream(**kwargs)
            el_stream(audio_stream)
            return True
        except Exception:
            pass

        # 2) Pipe mp3 chunks to mpv/ffplay
        import shutil
        import subprocess

        player = shutil.which("mpv") or shutil.which("ffplay")
        if player:
            if player.endswith("mpv"):
                cmd = [
                    player, "--no-terminal", "--really-quiet",
                    "--demuxer-append-buffer=32768",
                    "-",
                ]
            else:
                cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                assert proc.stdin is not None
                for chunk in self.iter_audio_chunks(text):
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                proc.stdin.close()
                proc.wait(timeout=120)
                return True
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        # 3) File then play
        path = self.synthesize(text)
        if path:
            ok = self.play_file(path)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return ok
        return False

    def play_file(self, path: Path) -> bool:
        try:
            import soundfile as sf
            import sounddevice as sd
            data, samplerate = sf.read(str(path))
            sd.play(data, samplerate)
            sd.wait()
            return True
        except Exception:
            pass
        try:
            import subprocess
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                check=False,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    def speak_sentences_parallel(self, text: str) -> None:
        """Speak sentence-by-sentence for faster first audio on long replies."""
        parts = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
        if not parts:
            self.play_stream(text)
            return
        # Merge tiny fragments
        merged: list[str] = []
        buf = ""
        for p in parts:
            buf = (buf + " " + p).strip() if buf else p
            if len(buf) >= 40 or p[-1:] in ".!?…":
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)
        for sentence in merged:
            self.play_stream(sentence)

    def speak(self, text: str) -> None:
        from avatar import AvatarPlayer

        show_avatar = self.avatar and self.avatar.available()
        backend = self.avatar.backend if self.avatar else ""
        fast = bool(getattr(self.cfg, "tts_fast_mode", True))

        # Unity needs a file path
        if show_avatar and backend == "unity":
            audio_path = self.synthesize(text) if self.enabled else None
            try:
                if audio_path and audio_path.is_file():
                    self.avatar.speak_with_audio(audio_path, play_audio=False, text=text)
                else:
                    self.avatar.play_for(AvatarPlayer.estimate_speech_seconds(text))
            finally:
                pass
            return

        # 3D avatar: synthesize file (needed for lip envelope) but use fast model
        if show_avatar and backend == "3d" and self.enabled and self.cfg.tts_auto_play:
            audio_path = self.synthesize(text)
            if audio_path:
                try:
                    self.avatar.speak_with_audio(audio_path, play_audio=True, text=text)
                finally:
                    try:
                        audio_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                return

        # Pure TTS path — prioritize streaming / sentence split
        if self.enabled and self.cfg.tts_auto_play:
            if show_avatar:
                # start visual immediately while audio streams
                try:
                    self.avatar.start(seconds=AvatarPlayer.estimate_speech_seconds(text))
                except Exception:
                    pass
            try:
                if fast and len(text) > 120:
                    self.speak_sentences_parallel(text)
                else:
                    self.play_stream(text)
            finally:
                if show_avatar:
                    try:
                        self.avatar.stop()
                    except Exception:
                        pass
            return

        if show_avatar and self.cfg.avatar_without_tts and text.strip():
            self.avatar.play_for(AvatarPlayer.estimate_speech_seconds(text))
