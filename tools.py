"""Built-in tools for Ollama function calling."""
from __future__ import annotations

import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── tool implementations ────────────────────────────────────────────

def list_directory(path: str = ".", max_entries: int = 80) -> str:
    """List files and folders in a directory.

    Args:
        path: Directory path to list (default current directory).
        max_entries: Maximum number of entries to return.

    Returns:
        A newline-separated listing of names with type markers.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: path does not exist: {p}"
    if not p.is_dir():
        return f"Error: not a directory: {p}"
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    lines = [f"{p}/"]
    for e in entries[:max_entries]:
        mark = "[dir] " if e.is_dir() else "[file]"
        size = ""
        if e.is_file():
            try:
                size = f" ({e.stat().st_size} bytes)"
            except OSError:
                pass
        lines.append(f"  {mark} {e.name}{size}")
    if len(entries) > max_entries:
        lines.append(f"  … and {len(entries) - max_entries} more")
    return "\n".join(lines)


def read_file(path: str, max_chars: int = 12000) -> str:
    """Read a text file and return its contents (truncated if very large).

    Args:
        path: Path to the file to read.
        max_chars: Maximum characters to return.

    Returns:
        File contents or an error message.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: not a file: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n… [truncated, total {len(text)} chars]"
        return text
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, append: bool = False) -> str:
    """Write text content to a file. Creates parent directories if needed.

    Args:
        path: Destination file path.
        content: Text to write.
        append: If true, append instead of overwrite.

    Returns:
        Success or error message.
    """
    p = Path(path).expanduser().resolve()
    # Safety: refuse paths outside cwd or obvious system dirs
    cwd = Path.cwd().resolve()
    try:
        p.relative_to(cwd)
    except ValueError:
        # allow absolute paths under home if user is explicit
        home = Path.home().resolve()
        try:
            p.relative_to(home)
        except ValueError:
            return f"Error: refused write outside project/home: {p}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_shell(command: str, timeout_seconds: int = 30) -> str:
    """Run a shell command and return stdout+stderr. Blocked dangerous patterns.

    Args:
        command: Shell command to execute.
        timeout_seconds: Max runtime before kill.

    Returns:
        Combined stdout/stderr or error.
    """
    blocked = ("rm -rf /", "mkfs", ":(){", "dd if=", "> /dev/sd", "shutdown", "reboot")
    low = command.lower()
    for b in blocked:
        if b in low:
            return f"Error: blocked potentially dangerous command pattern: {b}"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(Path.cwd()),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if not out.strip():
            out = f"(exit code {proc.returncode}, no output)"
        elif proc.returncode != 0:
            out = f"(exit {proc.returncode})\n{out}"
        if len(out) > 8000:
            out = out[:8000] + "\n… [truncated]"
        return out
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"Error: {e}"


def calculate(expression: str) -> str:
    """Evaluate a safe math expression (numbers, + - * / ** (), math functions).

    Args:
        expression: Math expression, e.g. "2 + 3 * sqrt(16)".

    Returns:
        Result as string or error.
    """
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "pi": math.pi, "e": math.e, "floor": math.floor, "ceil": math.ceil,
        "pow": pow,
    }
    try:
        # very small sandbox
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone_name: 'UTC' or 'local'.

    Returns:
        ISO-like datetime string.
    """
    if timezone_name.lower() == "local":
        now = datetime.now().astimezone()
    else:
        now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def search_rag(query: str, top_k: int = 4) -> str:
    """Search the local RAG knowledge base for relevant document chunks.

    Args:
        query: Natural language search query.
        top_k: Number of chunks to return.

    Returns:
        Retrieved chunks with sources, or a message if empty.
    """
    # Injected at runtime via bind_rag
    store = getattr(search_rag, "_rag_store", None)
    if store is None:
        return "RAG store is not available."
    try:
        hits = store.query(query, top_k=top_k)
        if not hits:
            return "No relevant documents found in RAG."
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(f"[{i}] source={h['filename']} (dist={h['distance']:.4f})\n{h['text']}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"RAG search error: {e}"


def bind_rag(rag_store: Any) -> None:
    """Attach the live RAG store so search_rag can use it."""
    search_rag._rag_store = rag_store  # type: ignore[attr-defined]


# ── example custom tools (walkthrough) ──────────────────────────────

def word_stats(text: str) -> str:
    """Count words, characters, and lines in a piece of text.

    Args:
        text: The text to analyze.
    """
    if not text:
        return "Empty text."
    lines = text.splitlines() or [text]
    words = text.split()
    return (
        f"lines={len(lines)} words={len(words)} "
        f"chars={len(text)} chars_no_space={len(text.replace(' ', ''))}"
    )


def dice_roll(sides: int = 6, count: int = 1) -> str:
    """Roll one or more dice and return the results.

    Args:
        sides: Number of sides on each die (default 6).
        count: How many dice to roll (default 1, max 20).
    """
    import random
    sides = max(2, min(int(sides), 1000))
    count = max(1, min(int(count), 20))
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    return f"Rolled {count}d{sides}: {rolls} (total={total})"


def http_get(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL over HTTP(S) and return the response body (truncated).

    Args:
        url: Full URL starting with http:// or https://.
        max_chars: Maximum characters of the body to return.
    """
    import urllib.request
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ollama-tui/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        if len(body) > max_chars:
            body = body[:max_chars] + "\n… [truncated]"
        return body
    except Exception as e:
        return f"Error fetching URL: {e}"


# ── registry ────────────────────────────────────────────────────────

# Name -> callable (must match function __name__ for Ollama)
TOOL_FUNCTIONS: Dict[str, Callable[..., str]] = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "calculate": calculate,
    "current_datetime": current_datetime,
    "search_rag": search_rag,
    # custom examples
    "word_stats": word_stats,
    "dice_roll": dice_roll,
    "http_get": http_get,
}

# Pass these callables to ollama.chat(tools=...)
TOOL_LIST: List[Callable] = list(TOOL_FUNCTIONS.values())


def execute_tool(name: str, arguments: dict) -> str:
    """Run a registered tool by name with given kwargs."""
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        # filter unexpected keys
        import inspect
        sig = inspect.signature(fn)
        allowed = {k: v for k, v in (arguments or {}).items() if k in sig.parameters}
        return str(fn(**allowed))
    except TypeError as e:
        return f"Error calling {name}: {e}"
    except Exception as e:
        return f"Error in {name}: {e}"


def tool_names() -> List[str]:
    return list(TOOL_FUNCTIONS.keys())
