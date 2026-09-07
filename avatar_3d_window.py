#!/usr/bin/env python3
"""
Standalone 3D talking-head window for Fedora/Linux desktops.

Run as a short-lived process:
  python avatar_3d_window.py --seconds 4
  python avatar_3d_window.py --audio /tmp/speech.mp3
  python avatar_3d_window.py --audio /tmp/speech.mp3 --play-audio

Renders a stylized 3D head with audio-reactive jaw, idle motion, and blinks.
Uses pygame + numpy only (no OpenGL required).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Prefer a predictable window position on multi-monitor desktops
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "80,80")


def load_audio_envelope(path: str, fps: int = 30) -> tuple[np.ndarray, float]:
    """Return (envelope 0..1 per frame, duration_seconds)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        duration = len(mono) / float(sr)
        frame_len = max(1, int(sr / fps))
        n_frames = max(1, int(math.ceil(len(mono) / frame_len)))
        env = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            chunk = mono[i * frame_len : (i + 1) * frame_len]
            if len(chunk) == 0:
                continue
            rms = float(np.sqrt(np.mean(chunk * chunk)))
            env[i] = rms
        peak = float(env.max()) if env.max() > 1e-8 else 1.0
        env = np.clip(env / peak, 0.0, 1.0)
        # slight smoothing
        if len(env) > 3:
            kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
            env = np.convolve(env, kernel, mode="same")
            env = np.clip(env, 0.0, 1.0)
        return env, duration
    except Exception:
        return np.array([0.3], dtype=np.float32), 2.0


def synthetic_envelope(seconds: float, fps: int = 30) -> np.ndarray:
    """Procedural speech-like mouth envelope when no audio is available."""
    n = max(1, int(seconds * fps))
    t = np.linspace(0, seconds, n, dtype=np.float32)
    # bursts of speech with pauses
    carrier = (np.sin(2 * math.pi * 3.5 * t) * 0.5 + 0.5) ** 1.5
    syllables = (np.sin(2 * math.pi * 7.0 * t) * 0.5 + 0.5)
    pause = (np.sin(2 * math.pi * 0.35 * t) * 0.5 + 0.5) > 0.25
    env = carrier * syllables * pause.astype(np.float32)
    return np.clip(env * 0.85 + 0.05, 0.0, 1.0)


def icosphere(subdivisions: int = 1, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Simple icosphere vertices + triangle indices."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    def midpoint(a, b, cache, verts_list):
        key = tuple(sorted((a, b)))
        if key in cache:
            return cache[key]
        mid = (verts_list[a] + verts_list[b]) * 0.5
        mid = mid / np.linalg.norm(mid)
        verts_list.append(mid)
        cache[key] = len(verts_list) - 1
        return cache[key]

    verts_list = [v.copy() for v in verts]
    for _ in range(subdivisions):
        cache: dict = {}
        new_faces = []
        for i, j, k in faces:
            a = midpoint(i, j, cache, verts_list)
            b = midpoint(j, k, cache, verts_list)
            c = midpoint(k, i, cache, verts_list)
            new_faces.extend([[i, a, c], [j, b, a], [k, c, b], [a, b, c]])
        faces = new_faces

    v = np.array(verts_list, dtype=np.float64) * radius
    f = np.array(faces, dtype=np.int32)
    return v, f


def project(points: np.ndarray, width: int, height: int, fov: float = 2.2) -> np.ndarray:
    """Perspective project Nx3 -> Nx2 screen coords + depth."""
    z = points[:, 2] + 4.2  # camera distance
    z = np.clip(z, 0.2, None)
    scale = (height * 0.35 * fov) / z
    x = width * 0.5 + points[:, 0] * scale
    y = height * 0.48 - points[:, 1] * scale
    return np.column_stack([x, y, z])


def rotate_y(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    r = points.copy()
    r[:, 0] = points[:, 0] * c + points[:, 2] * s
    r[:, 2] = -points[:, 0] * s + points[:, 2] * c
    return r


def rotate_x(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    r = points.copy()
    r[:, 1] = points[:, 1] * c - points[:, 2] * s
    r[:, 2] = points[:, 1] * s + points[:, 2] * c
    return r


def shade_color(normal: np.ndarray, base: tuple[int, int, int], light_dir: np.ndarray) -> tuple[int, int, int]:
    intensity = float(np.clip(np.dot(normal, light_dir), 0.15, 1.0))
    # soft rim
    rim = float(np.clip(1.0 - abs(normal[2]), 0.0, 1.0)) * 0.15
    f = min(1.0, intensity + rim)
    return (
        min(255, int(base[0] * f)),
        min(255, int(base[1] * f)),
        min(255, int(base[2] * f)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="3D talking-head avatar window")
    parser.add_argument("--seconds", type=float, default=3.0, help="Display duration if no audio")
    parser.add_argument("--audio", type=str, default="", help="Path to audio for mouth sync")
    parser.add_argument("--play-audio", action="store_true", help="Also play the audio file")
    parser.add_argument("--width", type=int, default=380)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--title", type=str, default="Assistant")
    parser.add_argument("--skin", type=str, default="90,170,210", help="RGB skin base color")
    args = parser.parse_args()

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError:
        print("pygame is required: pip install pygame", file=sys.stderr)
        return 1

    fps = 30
    if args.audio and Path(args.audio).is_file():
        envelope, duration = load_audio_envelope(args.audio, fps=fps)
        duration = max(duration, 0.8)
    else:
        duration = max(0.8, float(args.seconds))
        envelope = synthetic_envelope(duration, fps=fps)

    skin = tuple(int(x) for x in args.skin.split(","))
    if len(skin) != 3:
        skin = (90, 170, 210)

    pygame.init()
    pygame.display.set_caption(args.title)
    flags = pygame.SHOWN
    try:
        # borderless looks more "popup companion"
        screen = pygame.display.set_mode((args.width, args.height), flags | pygame.NOFRAME)
    except Exception:
        screen = pygame.display.set_mode((args.width, args.height), flags)

    clock = pygame.time.Clock()
    head_v, head_f = icosphere(subdivisions=2, radius=1.0)
    # flatten slightly into a face-like ellipsoid
    head_v = head_v * np.array([0.92, 1.08, 0.85])

    light = np.array([0.35, 0.55, 0.75], dtype=np.float64)
    light /= np.linalg.norm(light)

    # optional audio playback in background
    audio_thread_started = False
    if args.play_audio and args.audio and Path(args.audio).is_file():
        def _play():
            try:
                import soundfile as sf
                import sounddevice as sd
                data, sr = sf.read(args.audio)
                sd.play(data, sr)
                sd.wait()
            except Exception:
                try:
                    import subprocess
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", args.audio],
                        check=False,
                    )
                except Exception:
                    pass

        import threading
        threading.Thread(target=_play, daemon=True).start()
        audio_thread_started = True

    start = time.perf_counter()
    running = True
    blink_until = 0.0
    next_blink = start + 2.2

    bg_top = (18, 22, 32)
    bg_bot = (28, 36, 52)

    while running:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= duration + 0.15:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # background gradient
        for y in range(args.height):
            t = y / max(1, args.height - 1)
            col = (
                int(bg_top[0] * (1 - t) + bg_bot[0] * t),
                int(bg_top[1] * (1 - t) + bg_bot[1] * t),
                int(bg_top[2] * (1 - t) + bg_bot[2] * t),
            )
            pygame.draw.line(screen, col, (0, y), (args.width, y))

        # animation params
        frame_i = min(len(envelope) - 1, int(elapsed * fps))
        mouth = float(envelope[frame_i])
        # idle motion
        yaw = math.sin(elapsed * 0.9) * 0.18
        pitch = math.sin(elapsed * 1.1) * 0.06 + 0.05
        # blink
        if now >= next_blink:
            blink_until = now + 0.12
            next_blink = now + 2.0 + (elapsed % 1.7)
        eye_open = 0.08 if now < blink_until else 1.0

        # deform head: open jaw (lower vertices move down/back)
        deformed = head_v.copy()
        jaw_mask = deformed[:, 1] < -0.15
        deformed[jaw_mask, 1] -= mouth * 0.22
        deformed[jaw_mask, 2] -= mouth * 0.08
        # slight smile curl
        corner = (np.abs(deformed[:, 0]) > 0.35) & (deformed[:, 1] < 0.05) & (deformed[:, 1] > -0.35)
        deformed[corner, 1] += mouth * 0.03

        pts = rotate_y(deformed, yaw)
        pts = rotate_x(pts, pitch)
        proj = project(pts, args.width, args.height)

        # depth sort faces
        face_depth = []
        for tri in head_f:
            z = (pts[tri[0], 2] + pts[tri[1], 2] + pts[tri[2], 2]) / 3.0
            face_depth.append(z)
        order = np.argsort(face_depth)  # far to near? actually draw far first
        # smaller z is farther in our coord (camera looks +z toward -z... points z after camera offset)
        # We used z_cam = p.z + 4.2, so larger world z is closer-ish; sort ascending world z (far first)
        order = np.argsort(pts[head_f].mean(axis=1)[:, 2])

        for idx in order:
            tri = head_f[idx]
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            normal = np.cross(b - a, c - a)
            nlen = np.linalg.norm(normal)
            if nlen < 1e-9:
                continue
            normal = normal / nlen
            # backface cull (camera looks from +z toward origin-ish)
            if normal[2] < -0.05:
                continue
            col = shade_color(normal, skin, light)
            poly = [
                (int(proj[tri[0], 0]), int(proj[tri[0], 1])),
                (int(proj[tri[1], 0]), int(proj[tri[1], 1])),
                (int(proj[tri[2], 0]), int(proj[tri[2], 1])),
            ]
            pygame.draw.polygon(screen, col, poly)

        # eyes
        def draw_eye(local_x: float, local_y: float, local_z: float):
            eye = np.array([[local_x, local_y, local_z]], dtype=np.float64)
            eye = rotate_y(eye, yaw)
            eye = rotate_x(eye, pitch)
            p = project(eye, args.width, args.height)[0]
            ex, ey = int(p[0]), int(p[1])
            rx, ry = 16, int(14 * eye_open)
            if ry < 2:
                pygame.draw.line(screen, (30, 40, 55), (ex - rx, ey), (ex + rx, ey), 2)
                return
            pygame.draw.ellipse(screen, (245, 248, 255), (ex - rx, ey - ry, rx * 2, ry * 2))
            pygame.draw.ellipse(screen, (40, 60, 90), (ex - 7, ey - int(6 * eye_open), 14, int(12 * eye_open)))
            pygame.draw.circle(screen, (20, 24, 30), (ex, ey), 4)

        draw_eye(-0.38, 0.22, 0.78)
        draw_eye(0.38, 0.22, 0.78)

        # mouth opening indicator (inner cavity)
        if mouth > 0.05:
            mouth_pts = np.array(
                [
                    [-0.22, -0.28 - mouth * 0.05, 0.82],
                    [0.22, -0.28 - mouth * 0.05, 0.82],
                    [0.16, -0.28 - mouth * 0.28, 0.75],
                    [-0.16, -0.28 - mouth * 0.28, 0.75],
                ],
                dtype=np.float64,
            )
            mouth_pts = rotate_y(mouth_pts, yaw)
            mouth_pts = rotate_x(mouth_pts, pitch)
            mp = project(mouth_pts, args.width, args.height)
            poly = [(int(p[0]), int(p[1])) for p in mp]
            pygame.draw.polygon(screen, (40, 25, 35), poly)

        # soft status bar
        bar_w = int((args.width - 40) * min(1.0, elapsed / max(0.01, duration)))
        pygame.draw.rect(screen, (40, 50, 70), (20, args.height - 18, args.width - 40, 6), border_radius=3)
        pygame.draw.rect(screen, (80, 180, 220), (20, args.height - 18, bar_w, 6), border_radius=3)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
