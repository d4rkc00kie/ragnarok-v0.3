#!/usr/bin/env python3
"""
Silent Precision — clean, actionable terminal AI.

Markup streaming · natural file & shell intent · focused context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

import ollama
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Header, Input, Markdown, Static

from avatar import AvatarPlayer
from config import Config, get_config
from intent import Intent, IntentKind, parse_intent
from model_spec import ModelSpec, create_model_from_spec, show_model_info
from rag import RAGStore
from shell_mode import format_shell_result, run_shell
from tools import TOOL_LIST, bind_rag, execute_tool, tool_names
from tts import TTSEngine
from workspace import analyze_folder, list_dir, read_text
from sessions import SessionManager


SYSTEM = (
    "You are Silent Precision — a terminal AI that strips noise and returns "
    "clean, actionable insights in concise Markdown.\n"
    "Rules:\n"
    "- Prefer structure: short headings, bullets, fenced code when useful.\n"
    "- Stay on the user's actual question; no filler or preamble.\n"
    "- When context from files/shell is provided, ground answers in it and cite paths.\n"
    "- If something is missing, say what is missing in one line.\n"
)

HELP = """\
# Silent Precision

| You say | What happens |
|---------|----------------|
| *open src/ and analyze the code* | Full folder analysis then focused answer |
| *read README.md* | File loaded into context |
| *list .* | Directory listing |
| *run git status* | Shell executes; output in markup |
| *search authentication* | RAG / workspace search |
| normal question | Streaming Markdown reply |

**Slash**
**Session** `/new` `/reset` `/clear` `/history` `/save` `/retry` `/undo` `/title` `/sessions` `/resume`  
**Context** `/status` `/context` `/compress` `/focus`  
**Config** `/speed` `/params` `/model` `/tools` `/rag` `/tts` `/avatar` `/verbose` `/timestamps`  
**Info** `/help` `/version` `/config` `/diff` `/copy`
"""


class You(Markdown):
    DEFAULT_CSS = """
    You {
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick $accent;
        background: $boost;
    }
    """


class Reply(Markdown):
    DEFAULT_CSS = """
    Reply {
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick $success;
    }
    """


class Note(Markdown):
    DEFAULT_CSS = """
    Note {
        color: $text-muted;
        text-style: italic;
        margin: 0 0 1 0;
    }
    """


class SilentPrecision(App):
    TITLE = "Silent Precision"
    SUB_TITLE = "clean insights · markup stream · files & shell"
    CSS = """
    Screen { layout: vertical; }
    #stream {
        height: 1fr;
        padding: 0 1;
    }
    #prompt { width: 1fr; }
    #status {
        height: 1;
        dock: bottom;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("f1", "help", "Help", show=True),
        Binding("ctrl+t", "toggle_tts", "TTS", show=False),
        Binding("ctrl+g", "toggle_tools", "Tools", show=False),
        Binding("ctrl+a", "toggle_avatar", "Avatar", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = get_config()
        self.cfg.system_prompt = SYSTEM
        self.rag = RAGStore(self.cfg)
        bind_rag(self.rag)
        self.avatar = AvatarPlayer(self.cfg)
        self.tts = TTSEngine(self.cfg, avatar=self.avatar)
        self.model_spec = ModelSpec(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=getattr(self.cfg, "top_k_gen", 40),
            num_ctx=min(int(getattr(self.cfg, "num_ctx", 4096)), 4096),
            num_predict=512,
            num_batch=512,
            repeat_penalty=self.cfg.repeat_penalty,
        )
        self._busy = False
        self._cwd = Path.cwd()
        self.sessions = SessionManager()
        self._last_user = ""
        self._focus_mode = False
        self._timestamps = False
        self.messages = self.sessions.current.messages

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="stream"):
            yield Note(
                f"**Silent Precision** · `{self.cfg.main_model}` · "
                f"tools `{'on' if self.cfg.tools_enabled else 'off'}` · "
                f"type naturally — open a folder, run a command, or ask."
            )
        yield Input(placeholder="Instruction…  (F1 help)", id="prompt")
        yield Static(self._status(), id="status")

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()

    def _status(self) -> str:
        sid = getattr(self, "sessions", None)
        sid = sid.current.id if sid else "-"
        return (
            f" {self.cfg.main_model}  ·  ctx {self.model_spec.num_ctx}  ·  "
            f"t={self.model_spec.temperature}  ·  "
            f"{'tools' if self.cfg.tools_enabled else 'no-tools'}  ·  "
            f"s:{sid}  ·  cwd {self._cwd.name}/"
        )

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status", Static).update(self._status())
        except Exception:
            pass

    def _mount(self, w: Markdown) -> None:
        stream = self.query_one("#stream", VerticalScroll)
        stream.mount(w)
        stream.scroll_end(animate=False)

    def _note(self, text: str) -> None:
        self._mount(Note(text))

    @on(Input.Submitted, "#prompt")
    async def on_submit(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.clear()
        if not raw or self._busy:
            return

        intent = parse_intent(raw)
        if intent.kind == IntentKind.SLASH:
            await self._slash(raw)
            return
        if intent.kind == IntentKind.HELP:
            self._mount(Reply(HELP))
            return

        self._busy = True
        self._last_user = raw
        self._mount(You(f"**You**\n\n{raw}"))
        reply = Reply("…")
        self._mount(reply)
        self.dispatch(intent, reply)
        self._refresh_status()

    @work(thread=True)
    def dispatch(self, intent: Intent, reply: Reply) -> None:
        try:
            if intent.kind == IntentKind.READ_FILE:
                self._do_read(intent, reply)
            elif intent.kind == IntentKind.LIST_DIR:
                self._do_list(intent, reply)
            elif intent.kind == IntentKind.ANALYZE_FOLDER:
                self._do_analyze(intent, reply)
            elif intent.kind == IntentKind.SHELL:
                self._do_shell(intent, reply)
            elif intent.kind == IntentKind.SEARCH:
                self._do_search(intent, reply)
            else:
                self._do_chat(intent.raw, reply, extra_context="")
        except Exception as e:
            self.call_from_thread(reply.update, f"**Error**\n\n```\n{e}\n```")
        finally:
            self._busy = False
            self.call_from_thread(self._refresh_status)

    def _do_read(self, intent: Intent, reply: Reply) -> None:
        path = Path(intent.path or "").expanduser()
        if not path.is_file():
            self.call_from_thread(reply.update, f"File not found: `{intent.path}`")
            return
        body, n = read_text(path)
        lang = path.suffix.lstrip(".") or "text"
        markup = f"**`{path}`** · {n} chars\n\n```{lang}\n{body}\n```"
        self.call_from_thread(reply.update, markup)
        self._do_chat(
            f"Summarize actionable points from this file only.\n\nUser request: {intent.raw}",
            reply,
            extra_context=f"[File: {path}]\n{body[:16000]}",
            replace=False,
        )

    def _do_list(self, intent: Intent, reply: Reply) -> None:
        path = Path(intent.path or ".").expanduser()
        self.call_from_thread(reply.update, list_dir(path))

    def _do_analyze(self, intent: Intent, reply: Reply) -> None:
        self.call_from_thread(reply.update, f"_Scanning `{intent.path}`…_")
        report = analyze_folder(intent.path or ".", focus=intent.focus)
        preview = report if len(report) < 8000 else report[:8000] + "\n…"
        self.call_from_thread(reply.update, preview)
        self._do_chat(
            f"From the folder analysis, answer precisely:\n{intent.raw}\n\n"
            "Return only actionable insights and structure findings in Markdown.",
            reply,
            extra_context=report[:30000],
            replace=False,
        )

    def _do_shell(self, intent: Intent, reply: Reply) -> None:
        cmd = intent.shell_cmd or intent.raw
        self.call_from_thread(reply.update, f"_$ {cmd}_")
        code, out = run_shell(cmd, cwd=str(self._cwd))
        markup = format_shell_result(cmd, code, out)
        self.call_from_thread(reply.update, markup)
        if not re.match(r"^(git|npm|pip|cargo|go|python|node|docker|make)\b", intent.raw, re.I):
            self._do_chat(
                f"Interpret this command result for the user request.\n"
                f"Request: {intent.raw}\nCommand: {cmd}\n\nKeep it short and actionable.",
                reply,
                extra_context=out[:8000],
                replace=False,
            )

    def _do_search(self, intent: Intent, reply: Reply) -> None:
        q = intent.query or intent.raw
        hits = []
        try:
            hits = self.rag.query(q)
        except Exception:
            pass
        if hits:
            block = "\n\n".join(f"**{h['filename']}**\n{h['text']}" for h in hits)
            self.call_from_thread(reply.update, f"**Search**\n\n{block}")
            self._do_chat(
                f"Answer from the search hits only: {q}",
                reply,
                extra_context=block,
                replace=False,
            )
        else:
            self._do_chat(q, reply, extra_context="")

    def _do_chat(
        self,
        user_text: str,
        reply: Reply,
        extra_context: str = "",
        replace: bool = True,
    ) -> None:
        context_parts: List[str] = []
        if extra_context:
            context_parts.append(extra_context)
        try:
            hits = self.rag.query(user_text)
            if hits:
                context_parts.append(
                    "Retrieved:\n"
                    + "\n\n".join(f"[{h['filename']}]\n{h['text']}" for h in hits)
                )
        except Exception:
            pass

        for m in re.finditer(r"@([^\s]+)", user_text):
            p = Path(m.group(1)).expanduser()
            if p.is_file():
                body, _ = read_text(p, max_chars=8000)
                context_parts.append(f"[File: {p}]\n{body}")

        content = user_text
        if context_parts:
            content = (
                "Use only relevant context. Cite paths. Be concise.\n\n"
                + "\n\n---\n\n".join(context_parts)
                + f"\n\n---\n\n{user_text}"
            )

        self.messages.append({"role": "user", "content": content})
        if len(self.messages) > self.cfg.max_history * 2:
            self.messages = self.messages[-(self.cfg.max_history * 2) :]

        system = [{"role": "system", "content": self.cfg.system_prompt}]
        prefix = "" if replace else ""

        if self.cfg.tools_enabled:
            text = self._agent(system, list(self.messages), reply, prefix)
        else:
            text = self._stream(system + self.messages, reply, prefix)

        if text is not None:
            self.messages.append({"role": "assistant", "content": text})
            if text.strip() and (self.tts.enabled or self.avatar.available()):
                self.tts.speak(text)

    def _stream(self, messages: List[dict], reply: Reply, prefix: str = "") -> str:
        stream = ollama.chat(
            model=self.cfg.main_model,
            messages=messages,
            stream=True,
            options=self.model_spec.to_options(),
        )
        content = ""
        for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            if msg is None:
                continue
            part = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if part:
                content += part
                display = f"{prefix}\n\n{content}" if prefix else content
                self.call_from_thread(reply.update, display)
        return content

    def _agent(self, system: List[dict], working: List[dict], reply: Reply, prefix: str = "") -> str:
        log: List[str] = []
        for _ in range(self.cfg.max_tool_rounds):
            response = ollama.chat(
                model=self.cfg.main_model,
                messages=system + working,
                tools=TOOL_LIST,
                stream=False,
                options=self.model_spec.to_options(),
            )
            msg = response.message if hasattr(response, "message") else response.get("message", {})
            tool_calls = self._tool_calls(msg)
            content = ""
            if isinstance(msg, dict):
                content = msg.get("content") or ""
            else:
                content = getattr(msg, "content", None) or ""

            if not tool_calls:
                display = (("\n".join(log) + "\n\n") if log else "") + (content or "")
                if prefix:
                    display = prefix + "\n\n" + display
                self.call_from_thread(reply.update, display)
                return content

            working.append(self._msg_dict(msg))
            for call in tool_calls:
                name, args = call["name"], call["arguments"]
                log.append(f"`{name}`")
                result = execute_tool(name, args)
                working.append({"role": "tool", "tool_name": name, "content": result})

        return self._stream(system + working, reply, prefix)

    def _tool_calls(self, msg: Any) -> List[dict]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return []
        out = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
            if not fn:
                continue
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name:
                out.append({"name": name, "arguments": args or {}})
        return out

    def _msg_dict(self, msg: Any) -> dict:
        if isinstance(msg, dict):
            return msg
        d: dict = {"role": getattr(msg, "role", "assistant")}
        if getattr(msg, "content", None) is not None:
            d["content"] = msg.content
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            serialized = []
            for tc in tcs:
                fn = getattr(tc, "function", tc)
                serialized.append({
                    "function": {
                        "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", {}),
                    }
                })
            d["tool_calls"] = serialized
        return d

    async def _slash(self, raw: str) -> None:
        parts = raw.split(maxsplit=2)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        if cmd in ("/help", "/h", "/?"):
            self._mount(Reply(HELP))
        elif cmd in ("/quit", "/exit"):
            self.exit()

        elif cmd in ("/new", "/reset"):
            title = (arg + " " + rest).strip() or None
            self.sessions.new(title)
            self.messages = self.sessions.current.messages
            stream = self.query_one("#stream", VerticalScroll)
            await stream.remove_children()
            self._note(f"New session `{self.sessions.current.id}`" + (f" · {title}" if title else ""))
            self._refresh_status()

        elif cmd == "/history":
            if not self.messages:
                self._note("No history.")
            else:
                lines = ["**History**", ""]
                for i, m in enumerate(self.messages, 1):
                    role = m.get("role", "?")
                    preview = str(m.get("content", "")).replace("\n", " ")[:120]
                    lines.append(f"{i}. **{role}**: {preview}")
                self._mount(Reply("\n".join(lines)))

        elif cmd == "/save":
            # /save json|md|html [filename] [redact]
            fmt = (arg or "").lower()
            if fmt not in ("json", "md", "html", "markdown"):
                self._note("Usage: `/save <json|md|html> [filename] [redact]`")
            else:
                redact = "redact" in (rest or "").lower() or rest.strip().lower() == "redact"
                fname = None
                bits = rest.split()
                bits = [b for b in bits if b.lower() != "redact"]
                if bits:
                    fname = bits[0]
                try:
                    path = self.sessions.export(fmt if fmt != "markdown" else "md", fname, redact=redact)
                    self.sessions.save_disk()
                    self._note(f"Saved `{path}`")
                except Exception as e:
                    self._note(str(e))

        elif cmd == "/retry":
            if not self._last_user:
                self._note("Nothing to retry.")
            else:
                # drop last assistant if present
                if self.messages and self.messages[-1].get("role") == "assistant":
                    self.messages.pop()
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                self._busy = True
                self._mount(You(f"**You** (retry)\n\n{self._last_user}"))
                reply = Reply("…")
                self._mount(reply)
                from intent import parse_intent
                self.dispatch(parse_intent(self._last_user), reply)

        elif cmd == "/undo":
            n = 1
            if arg.isdigit():
                n = int(arg)
            removed = self.sessions.undo(n)
            self.messages = self.sessions.current.messages
            self._note(f"Undid ~{n} turn(s) · removed {removed} messages")

        elif cmd == "/title":
            name = (arg + " " + rest).strip()
            if not name:
                self._note(f"Title: **{self.sessions.current.title}**")
            else:
                self.sessions.set_title(name)
                self._note(f"Title → **{name}**")
            self._refresh_status()

        elif cmd == "/sessions":
            rows = self.sessions.list_sessions()
            if not rows:
                self._note("No saved sessions.")
            else:
                lines = ["**Sessions**", ""]
                for r in rows[:30]:
                    lines.append(
                        f"- `{r['id']}` **{r['title']}** · {r['turns']} msgs · {r['updated_at']}"
                    )
                lines.append("\nResume: `/resume <id|title>`")
                self._mount(Reply("\n".join(lines)))

        elif cmd == "/resume":
            key = (arg + " " + rest).strip()
            if not key:
                self._note("Usage: `/resume <id|title>`")
            else:
                s = self.sessions.resume(key)
                if not s:
                    self._note("Session not found.")
                else:
                    self.messages = self.sessions.current.messages
                    stream = self.query_one("#stream", VerticalScroll)
                    await stream.remove_children()
                    self._note(
                        f"Resumed `{s.id}` · **{s.title}** · {len(s.messages)} messages"
                    )
                    self._refresh_status()

        elif cmd in ("/compress", "/compact"):
            keep = 4
            if arg == "here" and rest.isdigit():
                keep = int(rest)
            elif arg.isdigit():
                keep = int(arg)
            preview = "preview" in raw.lower() or "dry-run" in raw.lower() or "--dry-run" in raw
            if preview:
                n = max(0, len(self.messages) - keep * 2)
                self._note(f"Would remove ~{n} messages, keep last {keep} turns")
            else:
                removed = self.sessions.compress(keep)
                self.messages = self.sessions.current.messages
                self._note(f"Compressed · removed {removed} · kept last {keep} turns")

        elif cmd in ("/status", "/context", "/ctx"):
            self._mount(
                Reply(
                    self.sessions.status(
                        self.cfg.main_model,
                        int(self.model_spec.num_ctx),
                    )
                )
            )

        elif cmd == "/config":
            lines = [
                "**Config**",
                f"- model: `{self.cfg.main_model}`",
                f"- embed: `{self.cfg.embed_model}`",
                f"- tools: `{self.cfg.tools_enabled}`",
                f"- tts: `{self.tts.enabled}`",
                f"- avatar: `{self.avatar.backend}` / enabled={self.cfg.avatar_enabled}",
                f"- cwd: `{self._cwd}`",
                f"- session: `{self.sessions.current.id}` {self.sessions.current.title}",
            ]
            self._mount(Reply("\n".join(lines)))

        elif cmd == "/version":
            self._note("Silent Precision · local Ollama TUI")

        elif cmd == "/focus":
            sub = (arg or "toggle").lower()
            if sub == "on":
                self._focus_mode = True
            elif sub == "off":
                self._focus_mode = False
            elif sub == "status":
                pass
            else:
                self._focus_mode = not self._focus_mode
            self._note(f"Focus mode {'on' if self._focus_mode else 'off'}")

        elif cmd in ("/timestamps", "/ts"):
            sub = (arg or "toggle").lower()
            if sub == "on":
                self._timestamps = True
            elif sub == "off":
                self._timestamps = False
            elif sub != "status":
                self._timestamps = not self._timestamps
            self._note(f"Timestamps {'on' if self._timestamps else 'off'}")

        elif cmd == "/verbose":
            self.cfg.tools_enabled = self.cfg.tools_enabled  # no-op anchor
            self._note("Verbose: tool names shown when agent calls tools.")

        elif cmd == "/diff":
            import subprocess
            mode = arg or "all"
            try:
                if mode == "staged":
                    out = subprocess.check_output(["git", "diff", "--cached"], text=True, cwd=str(self._cwd))
                else:
                    out = subprocess.check_output(["git", "diff"], text=True, cwd=str(self._cwd))
                if not out.strip():
                    self._note("No diff.")
                else:
                    self._mount(Reply(f"```diff\n{out[:12000]}\n```"))
            except Exception as e:
                self._note(str(e))

        elif cmd == "/copy":
            # last assistant message to clipboard if possible
            text = ""
            for m in reversed(self.messages):
                if m.get("role") == "assistant":
                    text = str(m.get("content", ""))
                    break
            if not text:
                self._note("No assistant message.")
            else:
                try:
                    import subprocess
                    p = subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=False)
                    if p.returncode != 0:
                        subprocess.run(["wl-copy"], input=text.encode(), check=False)
                    self._note("Copied last reply.")
                except Exception:
                    self._note("Clipboard tool missing (xclip/wl-copy).")

        elif cmd == "/prompt" or cmd == "/compose":
            self._note("Compose: type your message in the prompt line (external $EDITOR not wired in TUI).")

        elif cmd == "/clear":
            self.messages.clear()
            await self.query_one("#stream", VerticalScroll).remove_children()
            self._note("Cleared.")
        elif cmd == "/model":
            if arg:
                self.cfg.main_model = arg
                self._note(f"Model → `{arg}`")
            else:
                self._note(f"Model: `{self.cfg.main_model}`")
            self._refresh_status()
        elif cmd == "/models":
            try:
                listing = ollama.list()
                models = listing.get("models") if isinstance(listing, dict) else getattr(listing, "models", []) or []
                lines = ["**Models**"]
                for m in models:
                    name = m.get("model") if isinstance(m, dict) else getattr(m, "model", str(m))
                    lines.append(f"- `{name}`")
                self._mount(Reply("\n".join(lines)))
            except Exception as e:
                self._note(str(e))
        elif cmd == "/tools":
            if arg.lower() in ("on", "1", "true"):
                self.cfg.tools_enabled = True
                self._note("Tools on.")
            elif arg.lower() in ("off", "0", "false"):
                self.cfg.tools_enabled = False
                self._note("Tools off.")
            else:
                self._note("Tools: " + ", ".join(f"`{n}`" for n in tool_names()))
            self._refresh_status()
        elif cmd == "/speed":
            # /speed fast|turbo|balanced|quality|longctx|gpu|cpu
            name = (arg or "fast").lower()
            self._mount(Reply(self.model_spec.apply_preset(name)))
            self._refresh_status()

        elif cmd == "/params":

            if not arg:
                self._mount(Reply(self.model_spec.summary()))
            elif arg.lower() == "reset":
                self.model_spec = ModelSpec(
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=getattr(self.cfg, "top_k_gen", 40),
                    num_ctx=self.cfg.num_ctx,
                    repeat_penalty=self.cfg.repeat_penalty,
                )
                self._note("Params reset.")
            elif arg.lower() == "set" and rest:
                k, _, v = rest.partition(" ")
                self._note(self.model_spec.set_param(k, v.strip()))
            else:
                bits = raw.split(maxsplit=2)
                if len(bits) >= 3:
                    self._note(self.model_spec.set_param(bits[1], bits[2]))
            self._refresh_status()
        elif cmd in ("/tune", "/create-model"):
            name = arg
            base = self.cfg.main_model
            if rest.lower().startswith("from "):
                base = rest[5:].strip() or base
            if not name:
                self._note("Usage: `/tune <name>` [from <base>]")
            else:
                self._note(f"Creating `{name}`…")
                msg = create_model_from_spec(name, base, self.model_spec, self.cfg.system_prompt)
                self._mount(Reply(msg))
                if "Created model" in msg:
                    self.cfg.main_model = name
                    self._refresh_status()
        elif cmd == "/show":
            self._mount(Reply(show_model_info(arg or self.cfg.main_model)))
        elif cmd == "/rag":
            sub = arg.lower()
            if sub == "add" and rest:
                path = Path(rest).expanduser()
                if path.is_file():
                    n = self.rag.add_file(path)
                    self._note(f"Indexed `{path.name}` · {n} chunks")
                elif path.is_dir():
                    f, c = self.rag.add_directory(path)
                    self._note(f"Indexed {f} files · {c} chunks")
                else:
                    self._note("Path not found.")
            elif sub == "clear":
                self.rag.clear()
                self._note("RAG cleared.")
            else:
                self._note(f"RAG chunks: {self.rag.count()}")
        elif cmd == "/tts":
            if arg.lower() in ("on", "true", "1"):
                if self.cfg.elevenlabs_api_key:
                    self.cfg.tts_enabled = True
                    self.tts.enabled = True
                    self._note("TTS on.")
                else:
                    self._note("Set ELEVENLABS_API_KEY.")
            elif arg.lower() in ("off", "false", "0"):
                self.cfg.tts_enabled = False
                self.tts.enabled = False
                self._note("TTS off.")
            else:
                self._note(f"TTS {'on' if self.tts.enabled else 'off'}")
        elif cmd == "/tts-model":
            if arg:
                self.cfg.elevenlabs_model_id = arg
                self._note(f"TTS model → `{arg}`")
            else:
                self._note(f"TTS model: `{self.cfg.elevenlabs_model_id}`")
        elif cmd == "/avatar":
            if arg.lower() in ("3d", "video", "unity"):
                self.cfg.avatar_backend = arg.lower()
                self.cfg.avatar_enabled = True
                self.avatar.enabled = True
                self._note(f"Avatar → {arg.lower()}")
            elif arg.lower() in ("on", "off"):
                self.cfg.avatar_enabled = arg.lower() == "on"
                self.avatar.enabled = self.cfg.avatar_enabled
                self._note(f"Avatar {arg.lower()}")
            elif arg.lower() == "test":
                self.avatar.play_for(3.0)
            else:
                self._note(
                    f"Avatar backend `{self.avatar.backend}` · "
                    f"{'ready' if self.avatar.available() else 'off'}"
                )
        elif cmd == "/sys":
            if arg or rest:
                self.cfg.system_prompt = (arg + " " + rest).strip()
                self._note("System updated.")
            else:
                self._mount(Reply(f"```\n{self.cfg.system_prompt}\n```"))
        elif cmd == "/export":
            out = Path("chat_export.md")
            lines = ["# Silent Precision\n"]
            for m in self.messages:
                lines.append(f"## {m.get('role', '?').upper()}\n\n{m.get('content', '')}\n")
            out.write_text("\n".join(lines), encoding="utf-8")
            self._note(f"Wrote `{out.resolve()}`")
        elif cmd == "/cd" and arg:
            p = Path(arg).expanduser().resolve()
            if p.is_dir():
                self._cwd = p
                self._note(f"cwd → `{p}`")
                self._refresh_status()
            else:
                self._note("Not a directory.")
        else:
            self._note(f"Unknown `{cmd}` — F1 for help.")

    def action_clear(self) -> None:
        self.messages.clear()
        self.query_one("#stream", VerticalScroll).remove_children()
        self._note("Cleared.")

    def action_help(self) -> None:
        self._mount(Reply(HELP))

    def action_toggle_tts(self) -> None:
        if not self.cfg.elevenlabs_api_key:
            self._note("Set ELEVENLABS_API_KEY.")
            return
        self.cfg.tts_enabled = not self.cfg.tts_enabled
        self.tts.enabled = self.cfg.tts_enabled
        self._note(f"TTS {'on' if self.tts.enabled else 'off'}")

    def action_toggle_tools(self) -> None:
        self.cfg.tools_enabled = not self.cfg.tools_enabled
        self._note(f"Tools {'on' if self.cfg.tools_enabled else 'off'}")
        self._refresh_status()

    def action_toggle_avatar(self) -> None:
        self.cfg.avatar_enabled = not self.cfg.avatar_enabled
        self.avatar.enabled = self.cfg.avatar_enabled
        self._note(f"Avatar {'on' if self.avatar.enabled else 'off'}")


def main() -> None:
    SilentPrecision().run()


if __name__ == "__main__":
    main()
