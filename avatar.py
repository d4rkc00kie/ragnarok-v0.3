"""Talking-head avatar backends: 3d | video | unity."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from config import Config

_DIR = Path(__file__).resolve().parent
_WINDOW_SCRIPT = _DIR / "avatar_3d_window.py"


class AvatarPlayer:
    """Desktop talking-head companion.

    Backends:
      - ``3d``    : pygame software-3D head (no Unity required)
      - ``video`` : ffplay looping GIF/WebM/MP4
      - ``unity`` : external Unity Linux player over TCP JSON (best quality)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.avatar_enabled)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ffplay = shutil.which("ffplay")
        self._unity = None  # lazy

    @property
    def backend(self) -> str:
        return (self.cfg.avatar_backend or "3d").lower().strip()

    def _get_unity(self):
        if self._unity is None:
            from avatar_unity import UnityAvatarBridge
            self._unity = UnityAvatarBridge(self.cfg)
        self._unity.enabled = self.enabled
        return self._unity

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self.backend == "video":
            path = Path(self.cfg.avatar_path).expanduser()
            return bool(self._ffplay) and path.is_file()
        if self.backend == "unity":
            return self._get_unity().available()
        # 3d backend needs pygame + window script
        if not _WINDOW_SCRIPT.is_file():
            return False
        try:
            import pygame  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self, audio_path: Optional[Path] = None, seconds: Optional[float] = None) -> bool:
        if not self.available():
            return False
        with self._lock:
            self._stop_unlocked()
            if self.backend == "unity":
                u = self._get_unity()
                dur = seconds if seconds is not None else 3.0
                return u.speak(text="", audio_path=audio_path, duration=dur)
            if self.backend == "video":
                return self._start_video()
            return self._start_3d(audio_path=audio_path, seconds=seconds)

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self.backend == "unity" and self._unity is not None:
            try:
                self._unity.stop()
            except Exception:
                pass
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _start_video(self) -> bool:
        path = str(Path(self.cfg.avatar_path).expanduser().resolve())
        w = max(120, int(self.cfg.avatar_width))
        h = max(120, int(self.cfg.avatar_height))
        cmd = [
            self._ffplay, "-loglevel", "quiet", "-loop", "0",
            "-x", str(w), "-y", str(h),
            "-window_title", self.cfg.avatar_title, "-an", path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            self._proc = None
            return False

    def _start_3d(
        self,
        audio_path: Optional[Path] = None,
        seconds: Optional[float] = None,
        play_audio: bool = False,
    ) -> bool:
        cmd = [
            sys.executable, str(_WINDOW_SCRIPT),
            "--width", str(max(200, int(self.cfg.avatar_width))),
            "--height", str(max(220, int(self.cfg.avatar_height))),
            "--title", self.cfg.avatar_title,
            "--skin", self.cfg.avatar_skin,
        ]
        if audio_path and Path(audio_path).is_file():
            cmd.extend(["--audio", str(audio_path)])
            if play_audio:
                cmd.append("--play-audio")
        else:
            cmd.extend(["--seconds", str(seconds if seconds is not None else 3.0)])
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            self._proc = None
            return False

    def play_for(self, seconds: float) -> None:
        if self.backend == "unity":
            self._get_unity().speak(text="", duration=seconds)
            time.sleep(max(0.5, seconds))
            self._get_unity().idle()
            return
        if not self.start(seconds=seconds):
            return
        try:
            if self.backend == "video":
                time.sleep(max(0.5, seconds))
            else:
                if self._proc:
                    try:
                        self._proc.wait(timeout=max(1.0, seconds + 2.0))
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            self.stop()

    def speak_with_audio(self, audio_path: Path, play_audio: bool = False, text: str = "") -> None:
        if self.backend == "unity":
            u = self._get_unity()
            u.speak(text=text, audio_path=audio_path)
            # wait roughly for clip length
            try:
                import soundfile as sf
                info = sf.info(str(audio_path))
                time.sleep(max(0.8, float(info.duration) + 0.3))
            except Exception:
                time.sleep(3.0)
            u.idle()
            return
        if self.backend == "video":
            self.start()
            return
        self._start_3d(audio_path=audio_path, play_audio=play_audio)
        if self._proc:
            try:
                self._proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.stop()

    def unity_status(self) -> str:
        return self._get_unity().status()

    @staticmethod
    def estimate_speech_seconds(text: str, wpm: float = 150.0) -> float:
        words = max(1, len(text.split()))
        return max(1.5, (words / wpm) * 60.0)
