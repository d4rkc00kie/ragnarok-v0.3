"""Natural-language command parsing for Silent Precision."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class IntentKind(str, Enum):
    CHAT = "chat"
    READ_FILE = "read_file"
    ANALYZE_FOLDER = "analyze_folder"
    LIST_DIR = "list_dir"
    SHELL = "shell"
    SEARCH = "search"
    HELP = "help"
    SLASH = "slash"  # existing /commands


@dataclass
class Intent:
    kind: IntentKind
    raw: str
    path: Optional[str] = None
    query: Optional[str] = None
    shell_cmd: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    focus: Optional[str] = None  # e.g. "code", "logs", "structure"


_PATH = r"(?:~|/|\./|\.\./)[^\s\"']+|\"([^\"]+)\"|'([^']+)'"

_READ = re.compile(
    r"\b(?:open|read|show|cat|view|inspect|look at|display)\b.{0,40}?\b(" + _PATH + r"|\S+\.\w{1,8})\b",
    re.I,
)
_FOLDER = re.compile(
    r"\b(?:open|analyze|analyse|scan|explore|review|walk|inspect)\b.{0,48}?\b(?:folder|directory|dir|codebase|project|repo)\b"
    r"|\b(?:analyze|analyse|scan|explore)\b.{0,20}?\b(" + _PATH + r")\b",
    re.I,
)
_LIST = re.compile(
    r"\b(?:list|ls|show files|what(?:'s| is) in)\b.{0,40}?\b(" + _PATH + r"|here|this|current)?",
    re.I,
)
_SHELL = re.compile(
    r"\b(?:run|execute|exec|shell|bash|sh)\b\s+[`'\"]?(.+?)[`'\"]?\s*$"
    r"|\b(?:run|execute)\b\s+(?:the\s+)?(?:command\s+)?(.+)$",
    re.I,
)
_SEARCH = re.compile(
    r"\b(?:search|find|grep|look for|locate)\b\s+(.+)$",
    re.I,
)


def _clean_path(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().strip("`'\"")
    if s.lower() in ("here", "this", "current", "cwd", "."):
        return "."
    return s


def parse_intent(text: str) -> Intent:
    raw = text.strip()
    if not raw:
        return Intent(IntentKind.CHAT, raw)

    if raw.startswith("/"):
        return Intent(IntentKind.SLASH, raw)

    low = raw.lower()

    if low in ("help", "?", "what can you do"):
        return Intent(IntentKind.HELP, raw)

    # Shell: explicit run/execute
    m = _SHELL.search(raw)
    if m:
        cmd = (m.group(1) or m.group(2) or "").strip()
        if cmd:
            return Intent(IntentKind.SHELL, raw, shell_cmd=cmd)

    # Natural phrases that imply shell
    if re.match(r"^(git|npm|pip|cargo|go|python|node|docker|make|cmake)\b", raw, re.I):
        return Intent(IntentKind.SHELL, raw, shell_cmd=raw)

    # Folder analysis
    if _FOLDER.search(raw) or re.search(
        r"\b(analyze|analyse|review|scan)\b.+\b(code|source|project|repo|logs?)\b", low
    ):
        path = "."
        pm = re.search(_PATH, raw)
        if pm:
            path = _clean_path(pm.group(0)) or "."
        else:
            # bare names: "analyze src", "open lib and review"
            mdir = re.search(
                r"\b(?:open|analyze|analyse|scan|explore|review)\s+(\.?[\w.\-]+)",
                raw,
                re.I,
            )
            if mdir:
                cand = mdir.group(1)
                if cand.lower() not in ("the", "this", "my", "code", "folder", "directory", "project"):
                    path = cand
        focus = None
        if "log" in low:
            focus = "logs"
        elif "code" in low or "source" in low or "project" in low or "repo" in low:
            focus = "code"
        return Intent(IntentKind.ANALYZE_FOLDER, raw, path=path, focus=focus)

    # Read file
    m = _READ.search(raw)
    if m:
        path = _clean_path(m.group(1) or m.group(0).split()[-1])
        if path and ("/" in path or "." in Path(path).name):
            return Intent(IntentKind.READ_FILE, raw, path=path)

    # Bare path that looks like a file
    if re.match(r"^(?:~|/|\./|\.\./).+", raw) and not raw.endswith("/"):
        p = Path(raw).expanduser()
        if p.suffix or p.is_file():
            return Intent(IntentKind.READ_FILE, raw, path=raw)

    # List directory
    if re.search(r"\b(list|ls)\b", low) or "what's in" in low or "what is in" in low:
        path = "."
        pm = re.search(_PATH, raw)
        if pm:
            path = _clean_path(pm.group(0)) or "."
        return Intent(IntentKind.LIST_DIR, raw, path=path)

    # Search
    m = _SEARCH.search(raw)
    if m:
        return Intent(IntentKind.SEARCH, raw, query=m.group(1).strip())

    return Intent(IntentKind.CHAT, raw)
