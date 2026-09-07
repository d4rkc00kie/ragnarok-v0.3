# Unity avatar integration By Ꮆ卄ㄖ丂ㄒ  (Fedora / Linux)

Drive a Unity avatar from **ollama-tui** over a local TCP JSON bridge.

```
ollama-tui  --speak event-->  Unity Linux player (your character + lip sync)
```

## Fedora notes

1. Install **Unity Hub** (official Linux build) from [unity.com/download](https://unity.com/download).
2. Install an Editor version that supports **Linux Dedicated/Standalone** builds (6000.x / 2022.3 LTS are common).
3. Create a project, import your avatar (VRM via UniVRM, or humanoid FBX).
4. Add `OllamaTuiAvatarBridge.cs` to a GameObject with:
   - `Animator` (bool `IsTalking`, float `MouthOpen`) **or**
   - `SkinnedMeshRenderer` mouth blend shape index
   - `AudioSource` for TTS playback
5. **File → Build Settings → Linux** → build a standalone player, e.g. `~/UnityPlayers/OllamaAvatar.x86_64`.
6. Run the player **before** or with auto-launch from the TUI.

### Runtime packages often needed on Fedora

```bash
sudo dnf install mesa-libGL mesa-libGLU libX11 libXcursor libXrandr \
  libXi libXinerama pulseaudio-libs gtk3
```

NVIDIA: use proprietary drivers for best performance. AMD/Intel: Mesa is fine.

## TUI configuration

`.env`:

```bash
AVATAR_ENABLED=true
AVATAR_BACKEND=unity
UNITY_HOST=127.0.0.1
UNITY_PORT=8765
UNITY_AUTO_LAUNCH=true
UNITY_PLAYER_PATH=/home/YOU/UnityPlayers/OllamaAvatar.x86_64
UNITY_AUDIO_DIR=/tmp/ollama-tui-unity-audio
```

In the TUI:

```
/avatar unity
/avatar test
```

## Protocol

Newline-delimited JSON on `UNITY_HOST:UNITY_PORT`:

| type | fields | meaning |
|------|--------|---------|
| `hello` | client, version | TUI connected |
| `speak` | id, text, audio_path?, duration? | Start talk + optional audio file |
| `stop` | | Stop speech/anim |
| `idle` | | Return to idle |
| `ping` | | Keepalive |

`audio_path` is an absolute path on disk (copied under `UNITY_AUDIO_DIR` when possible). Unity loads it with `UnityWebRequestMultimedia` + `file://`.

## Animator setup (minimal)

- Create Animator Controller parameters: `IsTalking` (bool), `MouthOpen` (float 0–1).
- Idle state → Talk state when `IsTalking`.
- Mouth blend tree or blend shape driven by `MouthOpen`.
- Bridge fills `MouthOpen` from live audio loudness while the clip plays.

## Lip sync upgrades (optional)

- **uLipSync** (open source) — works on Linux, analyze AudioSource
- **OVR LipSync** — not ideal on Linux; prefer uLipSync / OpenLipSync
- Custom visemes from phoneme timing if you generate them in Python

## Security

The listener binds to `127.0.0.1` by default. Do not expose `UNITY_PORT` on a public interface.
