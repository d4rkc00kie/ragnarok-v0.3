"""Async-friendly file & folder analysis for Silent Precision."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh", ".ps1",
    ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".html", ".css", ".scss", ".sql", ".graphql", ".env", ".dockerfile",
    ".makefile", ".cmake", ".log", ".csv",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".idea", ".vscode", "target", "vendor", ".tox", ".mypy_cache", "chroma_db",
}


def read_text(path: Path, max_chars: int = 24000) -> Tuple[str, int]:
    data = path.read_text(encoding="utf-8", errors="replace")
    total = len(data)
    if total > max_chars:
        return data[:max_chars] + f"\n\n… [{total - max_chars} chars truncated]", total
    return data, total


def list_dir(path: Path, max_entries: int = 120) -> str:
    path = path.expanduser().resolve()
    if not path.is_dir():
        return f"Not a directory: `{path}`"
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"**{path}**", ""]
    for e in entries[:max_entries]:
        if e.is_dir():
            lines.append(f"- `📁 {e.name}/`")
        else:
            try:
                sz = e.stat().st_size
            except OSError:
                sz = 0
            lines.append(f"- `📄 {e.name}` ({sz} B)")
    if len(entries) > max_entries:
        lines.append(f"\n_{len(entries) - max_entries} more entries omitted_")
    return "\n".join(lines)


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def walk_files(root: Path, focus: Optional[str] = None, limit: int = 80) -> List[Path]:
    root = root.expanduser().resolve()
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        pdir = Path(dirpath)
        if _should_skip(pdir.relative_to(root) if root in pdir.parents or pdir == root else pdir):
            continue
        for name in filenames:
            fp = pdir / name
            ext = fp.suffix.lower()
            if focus == "logs" and ext not in {".log", ".txt", ".out"} and "log" not in name.lower():
                continue
            if focus == "code" and ext not in TEXT_EXT - {".md", ".txt", ".log", ".csv"}:
                if ext not in TEXT_EXT:
                    continue
            if ext in TEXT_EXT or name in ("Makefile", "Dockerfile", "Gemfile", "Cargo.toml"):
                found.append(fp)
            if len(found) >= limit:
                return found
    return found


def analyze_folder(
    path: str | Path,
    focus: Optional[str] = None,
    max_files: int = 40,
    max_chars_per_file: int = 4000,
    max_workers: int = 8,
) -> str:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Path not found: `{root}`"
    if root.is_file():
        body, n = read_text(root, max_chars_per_file * 4)
        return f"## File `{root.name}`\n\n```\n{body}\n```\n\n_{n} bytes_"

    files = walk_files(root, focus=focus, limit=max_files)
    if not files:
        return f"No matching files under `{root}`."

    structure = list_dir(root, max_entries=40)
    chunks: List[str] = [f"# Folder analysis: `{root}`", "", structure, "", "## File contents", ""]

    def load_one(fp: Path) -> str:
        try:
            rel = fp.relative_to(root)
        except ValueError:
            rel = fp.name
        try:
            body, n = read_text(fp, max_chars_per_file)
            lang = fp.suffix.lstrip(".") or "text"
            return f"### `{rel}`\n\n```{lang}\n{body}\n```\n"
        except Exception as e:
            return f"### `{rel}`\n\n_Error: {e}_\n"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(load_one, fp): fp for fp in files}
        for fut in as_completed(futs):
            chunks.append(fut.result())

    chunks.append(f"\n_Analyzed {len(files)} files (cap {max_files})._")
    return "\n".join(chunks)


def gather_context_for_chat(paths: Iterable[str], max_chars: int = 20000) -> str:
    parts: List[str] = []
    budget = max_chars
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        if p.is_dir():
            snippet = analyze_folder(p, max_files=12, max_chars_per_file=1500)
        else:
            body, _ = read_text(p, max_chars=min(budget, 12000))
            snippet = f"[File: {p}]\n{body}"
        if len(snippet) > budget:
            snippet = snippet[:budget] + "\n…"
        parts.append(snippet)
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n---\n\n".join(parts)
