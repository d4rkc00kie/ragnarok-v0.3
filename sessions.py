"""Session store for Silent Precision — history, titles, resume, save."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return s[:48] or "session"


@dataclass
class Session:
    id: str
    title: str = "untitled"
    messages: List[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()


class SessionManager:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or Path.home() / ".silent-precision" / "sessions")
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = Session(id=str(uuid.uuid4())[:8], title="default")

    def new(self, title: Optional[str] = None) -> Session:
        self.current = Session(
            id=str(uuid.uuid4())[:8],
            title=(title or "untitled").strip() or "untitled",
        )
        return self.current

    def set_title(self, title: str) -> None:
        self.current.title = title.strip() or self.current.title
        self.current.touch()

    def save_disk(self) -> Path:
        self.current.touch()
        path = self.root / f"{self.current.id}_{_slug(self.current.title)}.json"
        data = {
            "id": self.current.id,
            "title": self.current.title,
            "created_at": self.current.created_at,
            "updated_at": self.current.updated_at,
            "messages": self.current.messages,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def list_sessions(self) -> List[Dict[str, Any]]:
        rows = []
        for p in sorted(self.root.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                rows.append({
                    "file": str(p),
                    "id": data.get("id", p.stem),
                    "title": data.get("title", ""),
                    "updated_at": data.get("updated_at", ""),
                    "turns": len(data.get("messages") or []),
                })
            except Exception:
                continue
        return rows

    def resume(self, name_or_id: str) -> Optional[Session]:
        key = name_or_id.strip().lower()
        for p in self.root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                data.get("id", "").lower() == key
                or _slug(str(data.get("title", ""))) == _slug(key)
                or key in p.stem.lower()
            ):
                self.current = Session(
                    id=data.get("id") or str(uuid.uuid4())[:8],
                    title=data.get("title") or "restored",
                    messages=list(data.get("messages") or []),
                    created_at=data.get("created_at") or _now(),
                    updated_at=data.get("updated_at") or _now(),
                )
                return self.current
        return None

    def export(self, fmt: str, filename: Optional[str] = None, redact: bool = False) -> Path:
        fmt = fmt.lower().strip()
        messages = self.current.messages
        if redact:
            messages = [
                {**m, "content": re.sub(r"(sk-[a-zA-Z0-9]+|Bearer\s+\S+)", "[REDACTED]", str(m.get("content", "")))}
                for m in messages
            ]
        stem = filename or f"{self.current.id}_{_slug(self.current.title)}"
        if fmt == "json":
            path = Path(stem if stem.endswith(".json") else stem + ".json")
            path.write_text(
                json.dumps(
                    {
                        "id": self.current.id,
                        "title": self.current.title,
                        "messages": messages,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return path.resolve()
        if fmt in ("md", "markdown"):
            path = Path(stem if stem.endswith(".md") else stem + ".md")
            lines = [f"# {self.current.title}", "", f"_id: {self.current.id}_", ""]
            for m in messages:
                role = str(m.get("role", "?")).upper()
                lines.append(f"## {role}\n\n{m.get('content', '')}\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            return path.resolve()
        if fmt == "html":
            path = Path(stem if stem.endswith(".html") else stem + ".html")
            body = []
            for m in messages:
                role = m.get("role", "?")
                content = (
                    str(m.get("content", ""))
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                body.append(f"<h2>{role}</h2><pre>{content}</pre>")
            html = (
                f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{self.current.title}</title></head><body>"
                f"<h1>{self.current.title}</h1>"
                + "\n".join(body)
                + "</body></html>"
            )
            path.write_text(html, encoding="utf-8")
            return path.resolve()
        raise ValueError("fmt must be json|md|html")

    def compress(self, keep_last: int = 4) -> int:
        """Drop older turns; keep system-less chat tail. Returns removed count."""
        msgs = self.current.messages
        if len(msgs) <= keep_last * 2:
            return 0
        keep = msgs[-(keep_last * 2) :]
        removed = len(msgs) - len(keep)
        self.current.messages = keep
        self.current.touch()
        return removed

    def undo(self, n: int = 1) -> int:
        """Remove last n user+assistant pairs (approx). Returns removed messages."""
        n = max(1, n)
        removed = 0
        for _ in range(n):
            if not self.current.messages:
                break
            # pop assistant then user if present
            if self.current.messages and self.current.messages[-1].get("role") == "assistant":
                self.current.messages.pop()
                removed += 1
            if self.current.messages and self.current.messages[-1].get("role") == "user":
                self.current.messages.pop()
                removed += 1
        self.current.touch()
        return removed

    def status(self, model: str, num_ctx: int) -> str:
        msgs = self.current.messages
        chars = sum(len(str(m.get("content", ""))) for m in msgs)
        # rough token estimate
        toks = max(1, chars // 4)
        pct = min(100, int(100 * toks / max(1, num_ctx)))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        return (
            f"**Session** `{self.current.id}` · **{self.current.title}**\n\n"
            f"- model: `{model}`\n"
            f"- messages: **{len(msgs)}**\n"
            f"- ~tokens: **{toks}** / ctx **{num_ctx}**\n"
            f"- context: `{bar}` {pct}%\n"
            f"- updated: {self.current.updated_at}\n"
        )
