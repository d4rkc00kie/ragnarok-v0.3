"""Interactive shell execution with markup-friendly results."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


BLOCKED = (
    "rm -rf /",
    "mkfs",
    ":(){",
    "dd if=",
    "> /dev/sd",
    "shutdown",
    "reboot",
    "mkfs.",
    ":(){ :|:& };:",
)


def run_shell(command: str, timeout: int = 45, cwd: str | None = None) -> Tuple[int, str]:
    low = command.lower()
    for b in BLOCKED:
        if b in low:
            return 1, f"Blocked dangerous pattern: `{b}`"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(Path.cwd()),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if not out.strip():
            out = f"(exit {proc.returncode}, no output)"
        elif proc.returncode != 0:
            out = f"(exit {proc.returncode})\n{out}"
        if len(out) > 12000:
            out = out[:12000] + "\n… [truncated]"
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return 1, f"Timed out after {timeout}s"
    except Exception as e:
        return 1, str(e)


def format_shell_result(command: str, code: int, output: str) -> str:
    status = "ok" if code == 0 else f"exit {code}"
    return (
        f"**$ `{command}`** · _{status}_\n\n"
        f"```shell\n{output.rstrip()}\n```"
    )
