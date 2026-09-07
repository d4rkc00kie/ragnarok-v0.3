"""Unity avatar bridge for Fedora/Linux — WebSocket JSON protocol."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from config import Config


class UnityAvatarBridge:
    """Talk to a Unity Linux player over a simple TCP JSON line protocol.

    Protocol (one JSON object per line, UTF-8):

      {"type":"hello","client":"ollama-tui","version":1}
      {"type":"speak","id":"...","text":"...","audio_path":"/abs/path.wav","duration":3.2}
      {"type":"stop"}
      {"type":"idle"}
      {"type":"ping"}

    Unity replies (optional):

      {"type":"ready"}
      {"type":"speaking_done","id":"..."}
      {"type":"pong"}
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.avatar_enabled)
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._player_proc: Optional[subprocess.Popen] = None
        self._last_ready = 0.0

    @property
    def host(self) -> str:
        return self.cfg.unity_host

    @property
    def port(self) -> int:
        return int(self.cfg.unity_port)

    def available(self) -> bool:
        if not self.enabled:
            return False
        # Consider available if we can connect, or auto-launch is configured
        if self._can_connect(timeout=0.25):
            return True
        if self.cfg.unity_auto_launch and self.cfg.unity_player_path:
            return Path(self.cfg.unity_player_path).expanduser().is_file()
        # Soft-available: user may start Unity manually
        return True

    def _can_connect(self, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def ensure_player(self) -> bool:
        """Optionally launch the Unity Linux player binary."""
        if self._can_connect(timeout=0.3):
            return True
        path = (self.cfg.unity_player_path or "").strip()
        if not self.cfg.unity_auto_launch or not path:
            return False
        binary = Path(path).expanduser()
        if not binary.is_file():
            return False
        try:
            self._player_proc = subprocess.Popen(
                [str(binary)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # wait briefly for listen
            for _ in range(40):
                time.sleep(0.25)
                if self._can_connect(timeout=0.2):
                    return True
        except Exception:
            return False
        return False

    def connect(self) -> bool:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.getpeername()
                    return True
                except OSError:
                    self._sock = None
            self.ensure_player()
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                self._send_unlocked({"type": "hello", "client": "ollama-tui", "version": 1})
                return True
            except OSError:
                self._sock = None
                return False

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def _send_unlocked(self, payload: dict) -> bool:
        if self._sock is None:
            return False
        try:
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            self._sock.sendall(line.encode("utf-8"))
            return True
        except OSError:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            return False

    def send(self, payload: dict) -> bool:
        with self._lock:
            if self._sock is None and not self.connect():
                return False
            if not self._send_unlocked(payload):
                # one reconnect retry
                if self.connect():
                    return self._send_unlocked(payload)
                return False
            return True

    def speak(
        self,
        text: str,
        audio_path: Optional[Path] = None,
        duration: Optional[float] = None,
    ) -> bool:
        """Tell Unity to play talk animation / audio."""
        msg: dict[str, Any] = {
            "type": "speak",
            "id": str(uuid.uuid4()),
            "text": text[:4000],
        }
        if audio_path is not None:
            # copy into shared dir if configured (helps sandboxed Unity builds)
            path = Path(audio_path).resolve()
            shared = Path(self.cfg.unity_audio_dir).expanduser()
            try:
                shared.mkdir(parents=True, exist_ok=True)
                dest = shared / f"tts_{msg['id'][:8]}{path.suffix or '.mp3'}"
                shutil.copy2(path, dest)
                msg["audio_path"] = str(dest)
            except Exception:
                msg["audio_path"] = str(path)
        if duration is not None:
            msg["duration"] = float(duration)
        return self.send(msg)

    def stop(self) -> bool:
        return self.send({"type": "stop"})

    def idle(self) -> bool:
        return self.send({"type": "idle"})

    def ping(self) -> bool:
        return self.send({"type": "ping"})

    def status(self) -> str:
        online = self._can_connect(timeout=0.3)
        return (
            f"unity://{self.host}:{self.port} "
            f"{'online' if online else 'offline'} "
            f"auto_launch={'on' if self.cfg.unity_auto_launch else 'off'}"
        )
