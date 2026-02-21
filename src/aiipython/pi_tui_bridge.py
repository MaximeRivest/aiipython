"""Pi-TUI bridge for aiipython.

This module provides a full backend/frontend bridge so aiipython can run
with a Node.js frontend built on ``@mariozechner/pi-tui`` while keeping the
Python/IPython execution core unchanged.

Architecture
============

- Python process (this package) stays the source of truth for:
  - Session state
  - IPython namespace and execution
  - Agent loop
  - Auth/model/context/checkpoints
- Node process renders the terminal UI (pi-tui)
- The two sides communicate over a local JSON-RPC-over-TCP channel
  (localhost + ephemeral port).
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import sys
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from aiipython.agent import PromptAborted, ReactionStep
from aiipython.auth import ENV_KEY_MAP, get_auth_manager
from aiipython.context import summarize_items
from aiipython.model_catalog import discover_provider_catalog
from aiipython.parser import strip_executable_fenced_code
from aiipython.session import Session
from aiipython.settings import get_settings
from aiipython.tabminion import discover_services, is_running as tabminion_running
from aiipython.wire import WireEntry


UI_MAX_LINE_CHARS = 140
UI_MAX_BLOCK_LINES = 120
SNAPSHOT_TEXT_CHAR_LIMIT = 20_000
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


_TTY_EXIT_CLEANUP_REGISTERED = False


def _fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    return f"{n / 1_000_000:.1f}M"


def _abbrev_line(line: str, max_len: int = UI_MAX_LINE_CHARS) -> str:
    if len(line) <= max_len:
        return line
    keep = max(20, max_len - 22)
    return f"{line[:keep]}… [{len(line)} chars]"


def _abbrev_multiline(
    text: str,
    *,
    max_len: int = UI_MAX_LINE_CHARS,
    max_lines: int = UI_MAX_BLOCK_LINES,
) -> str:
    lines = text.splitlines() or [""]
    out = [_abbrev_line(line, max_len=max_len) for line in lines[:max_lines]]
    if len(lines) > max_lines:
        out.append(f"… [{len(lines) - max_lines} more lines]")
    return "\n".join(out)


def _without_fenced_code(markdown: str) -> str:
    return strip_executable_fenced_code(markdown)


@dataclass
class _PromptWaiter:
    token: str
    q: queue.Queue[str]


class PiTuiBackend:
    """Backend command + agent orchestration for the pi-tui frontend."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._emit_lock = threading.Lock()
        self._emit_raw: Callable[[dict[str, Any]], None] | None = None

        self._queue_lock = threading.Lock()
        self._pending_inputs: list[str] = []
        self._worker: threading.Thread | None = None
        self._busy = False

        self._prompt_waiters: dict[str, _PromptWaiter] = {}
        self._prompt_lock = threading.Lock()

        self._total_prompt = 0
        self._total_completion = 0
        self._total_cost = 0.0
        self._last_duration = 0.0

        self._model_menu_cache_title = ""
        self._model_menu_cache_options: list[tuple[str, str]] = []
        self._model_menu_cache_until = 0.0

        # Synchronous prompt call path (used by pi-native host provider bridge)
        self._prompt_once_lock = threading.Lock()
        self._prompt_cancel = threading.Event()

        # Preserve existing wire callbacks so we can restore on close.
        self._wire_on_request_prev = self.session.wire_log.on_request
        self._wire_on_chunk_prev = self.session.wire_log.on_chunk
        self._wire_on_done_prev = self.session.wire_log.on_done

        self.session.wire_log.on_request = self._wire_request
        self.session.wire_log.on_chunk = self._wire_chunk
        self.session.wire_log.on_done = self._wire_done

    @property
    def kernel(self):
        return self.session.kernel

    @property
    def agent(self):
        return self.session.agent

    def close(self) -> None:
        self.session.wire_log.on_request = self._wire_on_request_prev
        self.session.wire_log.on_chunk = self._wire_on_chunk_prev
        self.session.wire_log.on_done = self._wire_on_done_prev

    # ── transport ──────────────────────────────────────────────────

    def attach_transport(self, emit_raw: Callable[[dict[str, Any]], None]) -> None:
        with self._emit_lock:
            self._emit_raw = emit_raw

    def detach_transport(self) -> None:
        with self._emit_lock:
            self._emit_raw = None

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        with self._emit_lock:
            fn = self._emit_raw
        if not fn:
            return
        try:
            fn({"type": "event", "event": event, "data": data})
        except Exception:
            pass

    def _chat(self, role: str, content: str, *, fmt: str = "markdown") -> None:
        self._emit(
            "chat_message",
            {
                "role": role,
                "content": content,
                "format": fmt,
                "ts": time.time(),
            },
        )

    # ── hello / status ────────────────────────────────────────────

    def hello(self) -> dict[str, Any]:
        self._chat("system", self._welcome_markdown(), fmt="markdown")
        self._emit_status()
        self._emit_queue_depth()
        return {
            "model": self.session.model,
            "cwd": str(Path.cwd()),
            "version": "1",
        }

    def _welcome_markdown(self) -> str:
        from aiipython import __version__

        auth_info = ""
        if self.session.auth_source:
            src = self.session.auth_source
            if src.is_subscription:
                auth_info = (
                    f"\n🔑 **{src.provider}**: {src.source} "
                    "*(subscription — no per-call cost)*"
                )
            else:
                auth_info = (
                    f"\n🔑 **{src.provider}**: {src.source} "
                    "*(billed per API call)*"
                )
        else:
            auth_info = "\n⚠️ No auth configured — use `/login` or set an API key env var"

        tm_info = ""
        if tabminion_running():
            svcs = discover_services()
            if svcs:
                names = ", ".join(f"{s['emoji']}{s['name']}" for s in svcs)
                tm_info = f"\n🌐 **TabMinion**: {names} — `/tabminion` for details"
            else:
                tm_info = "\n🌐 **TabMinion** running (no AI tabs detected)"

        return (
            f"**aiipython** v{__version__} (pi-tui frontend){auth_info}{tm_info}\n\n"
            "`ctrl+c` quit · `ctrl+i` inspector\n"
            "Use `@file.py` (live) · `@clip` or `@ `+paste (snapshot)\n"
            "`!cmd` IPython input · `!!cmd` same as `!cmd`\n"
            "`/model` `/context` `/vars [filter]` `/image` `/login` `/logout` `/auth` `/tabminion`\n"
            "`/tree` `/undo` `/restore <id>` `/fork [label]`"
        )

    def _emit_status(self) -> None:
        auth_label = ""
        if self.session.auth_source:
            auth_label = f"🔑 {self.session.auth_source.source}"

        self._emit(
            "status",
            {
                "model": self.session.model,
                "prompt_tokens": self._total_prompt,
                "completion_tokens": self._total_completion,
                "total_cost": self._total_cost,
                "duration_ms": self._last_duration,
                "auth_label": auth_label,
                "stats_compact": self._compact_stats_line(),
                "cwd": str(Path.cwd()),
                "busy": self._busy,
            },
        )

    def _compact_stats_line(self) -> str:
        parts: list[str] = []
        if self._total_prompt or self._total_completion:
            parts.append(f"↑{_fmt_tokens(self._total_prompt)} ↓{_fmt_tokens(self._total_completion)}")
        if self._total_cost > 0:
            parts.append(f"${self._total_cost:.3f}")
        if self._last_duration > 0:
            parts.append(f"{self._last_duration / 1000:.1f}s")
        return " ".join(parts)

    # ── queue + work loop ─────────────────────────────────────────

    def submit_input(self, raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").rstrip("\n")
        if not text.strip():
            return {"accepted": False, "reason": "empty"}

        with self._queue_lock:
            self._pending_inputs.append(text)
            self._emit_queue_depth_locked()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work_loop, daemon=True)
                self._worker.start()

        return {"accepted": True, "queued": True}

    def _emit_queue_depth_locked(self) -> None:
        self._emit("queue", {"depth": len(self._pending_inputs)})

    def _emit_queue_depth(self) -> None:
        with self._queue_lock:
            depth = len(self._pending_inputs)
        self._emit("queue", {"depth": depth})

    def _work_loop(self) -> None:
        while True:
            with self._queue_lock:
                if not self._pending_inputs:
                    self._busy = False
                    self._emit_queue_depth_locked()
                    self._emit_status()
                    return
                raw = self._pending_inputs.pop(0)
                self._busy = True
                self._emit_queue_depth_locked()
                self._emit_status()

            try:
                self._process_input(raw)
            except Exception as exc:
                self._chat("error", f"Backend error: {exc}", fmt="text")

    # ── wire callbacks ────────────────────────────────────────────

    def _wire_request(self, entry: WireEntry) -> None:
        self._emit(
            "wire_request",
            {
                "id": entry.entry_id,
                "model": entry.model,
                "ts": entry.ts,
                "messages": entry.messages,
                "kwargs": entry.kwargs,
            },
        )

    def _wire_chunk(self, entry: WireEntry, chunk: str) -> None:
        self._emit(
            "wire_chunk",
            {
                "id": entry.entry_id,
                "chunk": chunk,
            },
        )

    def _wire_done(self, entry: WireEntry) -> None:
        self._emit(
            "wire_done",
            {
                "id": entry.entry_id,
                "duration_ms": entry.duration_ms,
                "usage": entry.usage,
                "error": entry.error,
                "full_response": entry.full_response,
            },
        )

        self._total_prompt += int(entry.usage.get("prompt_tokens", 0) or 0)
        self._total_completion += int(entry.usage.get("completion_tokens", 0) or 0)
        self._total_cost += float(entry.usage.get("cost", 0) or 0)
        self._last_duration = float(entry.duration_ms or 0.0)
        self._emit_status()

    # ── processing ────────────────────────────────────────────────

    def transform_at_refs(self, raw_text: str) -> dict[str, Any]:
        """Apply aiipython @-reference behavior to one input string.

        Returns:
            {
              "handled": bool,     # true => no model call should be made
              "text": str,         # transformed user text
              "messages": [str],   # informational/warning lines for UI
            }
        """
        raw = raw_text or ""
        text = raw.strip()
        if not text:
            return {"handled": False, "text": "", "messages": []}

        # Match previous behavior: slash/shell commands are handled first and
        # do not pass through @-pinning rewrite.
        if text.startswith("/") or text.startswith("!"):
            return {"handled": False, "text": text, "messages": []}

        raw_lstrip = raw.lstrip()
        if raw_lstrip.startswith("@ "):
            payload = raw_lstrip[2:]
            msg = self._context_add_snapshot_payload(payload)
            return {"handled": True, "text": "", "messages": [msg]}

        transformed, messages = self._consume_at_context_refs(text)
        handled = not bool(transformed)
        return {
            "handled": handled,
            "text": transformed,
            "messages": messages,
        }

    def _process_input(self, raw_text: str) -> None:
        text = raw_text.strip()
        if not text:
            return

        # Shell shortcuts
        # In aiipython, route both !cmd and !!cmd through IPython so they
        # affect the same Python process/session state.
        if text.startswith("!!") and len(text) > 2:
            self._handle_ipython_shell(text[2:].strip())
            return
        if text.startswith("!") and len(text) > 1:
            self._handle_ipython_shell(text[1:].strip())
            return

        # Slash commands
        if text.startswith("/model"):
            self._handle_model(text[6:].strip())
            return
        if text.startswith("/image "):
            self._handle_image(text[7:].strip())
            return
        if text.startswith("/login"):
            self._handle_login(text[6:].strip())
            return
        if text.startswith("/logout"):
            self._handle_logout(text[7:].strip())
            return
        if text.startswith("/context"):
            self._handle_context(text[8:].strip())
            return
        if text.startswith("/vars"):
            self._handle_vars(text[5:].strip())
            return
        if text in {"/state", "/variables"}:
            self._handle_vars("")
            return
        if text == "/auth":
            self._handle_auth()
            return
        if text == "/tabminion":
            self._handle_tabminion()
            return
        if text == "/tree":
            self._handle_tree()
            return
        if text == "/undo":
            self._handle_undo()
            return
        if text.startswith("/restore"):
            self._handle_restore(text[8:].strip())
            return
        if text.startswith("/fork"):
            self._handle_fork(text[5:].strip())
            return

        transformed = self.transform_at_refs(text)
        for msg in transformed.get("messages", []):
            self._chat("system", str(msg), fmt="text")

        if transformed.get("handled"):
            return

        text = str(transformed.get("text", "") or "").strip()
        if not text:
            return

        self._chat("user", text, fmt="text")
        self.kernel.push_user_input(text, source="chat", is_clipboard=False)

        self._emit("thinking", {"active": True})
        try:
            self.agent.react(on_step=self._display_step)
        except Exception as exc:
            self._chat("error", f"Agent error: {exc}", fmt="text")
        finally:
            self._emit("thinking", {"active": False})

    def _handle_ipython_shell(self, cmd: str) -> None:
        """Run `!cmd` as IPython input (bound aiipython session state)."""
        if not cmd:
            self._chat("system", "Usage: !<python_or_ipython_input>", fmt="text")
            return

        self._chat("user", f"!{cmd}", fmt="text")
        entry = self.kernel.execute(cmd, tag="user")
        self._chat(
            "system",
            self._format_exec_output(entry),
            fmt="text",
        )

    def run_ipython_shell(self, command: str, *, exclude_from_context: bool = False) -> dict[str, Any]:
        """Run a shell command via IPython and return a bash-result payload.

        Used by pi-native's `user_bash` interception so `!`/`!!` operate
        against aiipython's bound IPython process.
        """
        cmd = (command or "").strip()
        if not cmd:
            return {
                "output": "",
                "exit_code": 0,
                "cancelled": False,
                "truncated": False,
                "exclude_from_context": bool(exclude_from_context),
            }

        with self._prompt_once_lock:
            entry = self.kernel.execute(cmd, tag="user")
            out = self._format_exec_output(entry)
            exit_code = 0 if bool(entry.get("success", True)) else 1

        return {
            "output": out,
            "exit_code": exit_code,
            "cancelled": False,
            "truncated": False,
            "exclude_from_context": bool(exclude_from_context),
        }

    def _handle_local_shell(self, cmd: str) -> None:
        """Backward-compatible alias: `!!cmd` now routes through IPython too."""
        self._handle_ipython_shell(cmd)

    def _display_step(self, step: ReactionStep) -> None:
        shown_markdown = step.markdown
        if step.blocks:
            prose = _without_fenced_code(step.markdown)
            shown_markdown = prose or "Running requested python blocks…"

        blocks_payload: list[dict[str, Any]] = []
        if step.blocks:
            for idx, block in enumerate(step.blocks, start=1):
                entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                success = bool(entry.get("success", True))
                blocks_payload.append(
                    {
                        "index": idx,
                        "lang": block.lang,
                        "code": _abbrev_multiline(block.code),
                        "success": success,
                        "output": self._format_exec_output(entry),
                    }
                )

        self._emit(
            "reaction_step",
            {
                "iteration": step.iteration,
                "assistant_markdown": shown_markdown,
                "raw_markdown": step.markdown,
                "blocks": blocks_payload,
                "is_final": step.is_final,
            },
        )

    def _format_exec_output(self, entry: dict[str, Any]) -> str:
        parts: list[str] = []

        stdout = (entry.get("stdout") or "").rstrip()
        stderr = (entry.get("stderr") or "").rstrip()
        result = entry.get("result") or ""
        error = entry.get("error") or ""

        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"stderr: {stderr}")
        if result:
            parts.append(f"→ {result}")
        if error:
            parts.append(f"✗ {error}")
        if not parts:
            parts.append("✓ (no output)")

        return _abbrev_multiline("\n".join(parts))

    # ── @context refs ─────────────────────────────────────────────

    def _consume_at_context_refs(self, text: str) -> tuple[str, list[str]]:
        messages: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            token = raw.rstrip(",.;:!?)")
            suffix = raw[len(token):]
            if not token:
                return match.group(0)

            lower = token.lower()
            if lower in {"clip", "clipboard"}:
                msg = self._context_add_from_clipboard(snapshot=True)
                messages.append(msg)
                return suffix

            resolved = self._resolve_at_file_token(token)
            if isinstance(resolved, list):
                shown = ", ".join(str(p.relative_to(Path.cwd())) for p in resolved[:5])
                messages.append(f"⚠ `@{token}` matched multiple files. Be specific: {shown}")
                return match.group(0)

            if resolved is None:
                return match.group(0)

            try:
                item = self.agent.context_add_file(str(resolved), source=f"@{token}")
                messages.append(f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**")
                return suffix
            except Exception as exc:
                messages.append(f"⚠ failed to pin `@{token}`: {exc}")
                return match.group(0)

        updated = re.sub(r"(?<!\S)@([^\s]+)", _replace, text)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        return updated, messages

    def _resolve_at_file_token(self, token: str) -> Path | list[Path] | None:
        raw = token.strip().strip('"').strip("'")
        if not raw:
            return None

        if raw.startswith("file://"):
            parsed = urlparse(raw)
            raw = unquote(parsed.path)

        p = Path(raw).expanduser()
        if p.is_file():
            return p.resolve()

        rel = (Path.cwd() / raw).expanduser()
        if rel.is_file():
            return rel.resolve()

        if "/" not in raw and "\\" not in raw:
            matches = self._find_cwd_file_candidates(raw, limit=8)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return matches

        return None

    def _find_cwd_file_candidates(self, needle: str, *, limit: int = 8) -> list[Path]:
        out: list[Path] = []
        root = Path.cwd()
        skip_dirs = {
            ".git", "node_modules", ".venv", "dist", "build", ".cache",
            "__pycache__", ".aiipython_checkpoints",
        }
        needle_l = needle.lower()

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            depth = len(current.relative_to(root).parts)
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            if depth >= 6:
                dirnames[:] = []

            for fname in filenames:
                if needle_l not in fname.lower():
                    continue
                out.append((current / fname).resolve())
                if len(out) >= limit:
                    return out

        return out

    # ── context snapshot helpers ──────────────────────────────────

    def _snapshot_context_dir(self) -> Path:
        d = Path.cwd() / ".aiipython_context"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _snapshot_file_to_context(self, path: Path, *, source: str) -> dict[str, Any]:
        p = path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))

        if p.suffix.lower() in IMAGE_EXTS:
            snap = self._snapshot_context_dir() / f"snapshot_{int(time.time())}_{p.name}"
            shutil.copy2(p, snap)
            return self.agent.context_add_file(str(snap), source=source)

        raw = p.read_bytes()
        if b"\x00" in raw[:2048]:
            text = f"<binary snapshot: {p} ({len(raw)} bytes)>"
        else:
            text = raw.decode("utf-8", errors="replace")
            if len(text) > SNAPSHOT_TEXT_CHAR_LIMIT:
                text = (
                    text[:SNAPSHOT_TEXT_CHAR_LIMIT]
                    + f"\n… [{len(text) - SNAPSHOT_TEXT_CHAR_LIMIT} more chars]"
                )

        return self.agent.context_add_text(
            text,
            label=f"{p.name} snapshot",
            source=source,
        )

    def _context_add_snapshot_payload(self, payload: str) -> str:
        raw = payload.rstrip("\n")
        if not raw.strip():
            return self._context_add_from_clipboard(snapshot=True)

        token = raw.strip().strip('"').strip("'")
        if token.startswith("file://"):
            parsed = urlparse(token)
            token = unquote(parsed.path)

        if "\n" not in token:
            p = Path(token).expanduser()
            if p.is_file():
                item = self._snapshot_file_to_context(p, source="@paste:file")
                return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}** (snapshot)"

        item = self.agent.context_add_text(raw, label="pasted", source="@paste:text")
        return f"📌 pinned `{item['id']}` [text] **{item['label']}** (snapshot)"

    def _read_clipboard_text(self) -> str | None:
        commands = [
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["pbpaste"],
        ]
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
            except Exception:
                continue
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.strip()
        return None

    def _context_add_from_clipboard(self, *, snapshot: bool = False) -> str:
        try:
            from PIL import Image, ImageGrab

            grabbed = ImageGrab.grabclipboard()
            if isinstance(grabbed, Image.Image):
                img_path = self._snapshot_context_dir() / f"clipboard_{int(time.time())}.png"
                grabbed.save(img_path)

                item = self.agent.context_add_file(str(img_path), source="@clip:image")
                name = img_path.stem
                self.kernel.push_image(name, grabbed)
                return (
                    f"📌 pinned `{item['id']}` [image] **{item['label']}** "
                    f"and loaded `images['{name}']`"
                )

            if isinstance(grabbed, list) and grabbed:
                added: list[str] = []
                for path_like in grabbed[:3]:
                    p = Path(path_like).expanduser()
                    if not p.is_file():
                        continue

                    if snapshot:
                        item = self._snapshot_file_to_context(p, source="@clip:path-snapshot")
                    else:
                        item = self.agent.context_add_file(str(p), source="@clip:path")
                    added.append(f"`{item['id']}` {item['label']}")
                if added:
                    suffix = " (snapshot)" if snapshot else ""
                    return "📌 pinned clipboard paths" + suffix + ": " + ", ".join(added)
        except Exception:
            pass

        clip = self._read_clipboard_text()
        if not clip:
            return "⚠ clipboard is empty or unreadable"

        clip = clip.strip()
        resolved = self._resolve_at_file_token(clip)
        if isinstance(resolved, Path):
            if snapshot:
                item = self._snapshot_file_to_context(resolved, source="@clip:path-snapshot")
                return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}** (snapshot)"
            item = self.agent.context_add_file(str(resolved), source="@clip:path")
            return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**"

        item = self.agent.context_add_text(clip, label="clipboard", source="@clip:text")
        return f"📌 pinned `{item['id']}` [text] **{item['label']}**"

    # ── commands ──────────────────────────────────────────────────

    def _handle_context(self, arg: str) -> None:
        cmd = arg.strip()
        parts = cmd.split(maxsplit=2) if cmd else []
        verb = (parts[0].lower() if parts else "list")

        try:
            if verb in {"", "open", "list"}:
                items = self.agent.context_list()
                body = summarize_items(items)
                self._chat(
                    "system",
                    "**Pinned context**\n\n"
                    f"{body}\n\n"
                    "Use: `/context add <path>`, `/context add-text <text>`, "
                    "`/context add-clip` (snapshot), `/context rm <id>`, `/context clear`\n"
                    "Shortcuts: `@path/to/file` (live ref), `@clip` (snapshot), or `@ ` + paste.",
                    fmt="markdown",
                )
            elif verb == "add" and len(parts) >= 2:
                path_str = cmd[len(parts[0]):].strip()
                item = self.agent.context_add_file(path_str, source="/context add")
                self._chat("system", f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**")
            elif verb in {"add-text", "text"} and len(parts) >= 2:
                text_value = cmd[len(parts[0]):].strip()
                item = self.agent.context_add_text(
                    text_value,
                    label="manual",
                    source="/context add-text",
                )
                self._chat("system", f"📌 pinned `{item['id']}` [text] **{item['label']}**")
            elif verb in {"add-clip", "clip", "paste"}:
                self._chat("system", self._context_add_from_clipboard(snapshot=True), fmt="text")
            elif verb in {"rm", "remove", "del", "delete"} and len(parts) >= 2:
                item_id = parts[1].strip()
                ok = self.agent.context_remove(item_id)
                if ok:
                    self._chat("system", f"Removed `{item_id}` from pinned context.")
                else:
                    self._chat("error", f"No pinned context item `{item_id}`.", fmt="text")
            elif verb == "clear":
                n = self.agent.context_clear()
                self._chat("system", f"Cleared {n} pinned context item(s).")
            else:
                self._chat(
                    "system",
                    "Usage: `/context [open|list|add <path>|add-text <text>|add-clip|rm <id>|clear]`\n"
                    "Tip: type `@ ` then paste to pin as snapshot text/file.",
                    fmt="text",
                )
        except Exception as exc:
            self._chat("error", f"/context error: {exc}", fmt="text")

    def _handle_vars(self, arg: str) -> None:
        query = (arg or "").strip().lower()
        snapshot = self.kernel.snapshot()
        items = sorted(snapshot.items(), key=lambda kv: kv[0].lower())

        if query:
            items = [
                (k, v)
                for (k, v) in items
                if query in k.lower() or query in str(v).lower()
            ]

        if not items:
            suffix = f" matching `{query}`" if query else ""
            self._chat("system", f"No IPython variables{suffix}.", fmt="text")
            return

        max_items = 80
        shown = items[:max_items]
        lines = [
            "**IPython variables**"
            + (f" _(filter: `{query}`)_" if query else "")
            + f" — {len(items)} item(s)",
            "",
        ]
        lines.extend(f"- `{name}`: {summary}" for name, summary in shown)
        if len(items) > max_items:
            lines.extend(
                [
                    "",
                    f"… {len(items) - max_items} more variables. Narrow with `/vars <filter>`.",
                ]
            )

        self._chat("system", "\n".join(lines), fmt="markdown")

    def _model_menu_options(self, force_refresh: bool = False) -> tuple[str, list[tuple[str, str]]]:
        now = time.monotonic()
        if (
            not force_refresh
            and self._model_menu_cache_options
            and now < self._model_menu_cache_until
        ):
            return self._model_menu_cache_title, self._model_menu_cache_options

        options: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(value: str, label: str) -> None:
            if value not in seen:
                seen.add(value)
                options.append((value, label))

        add(self.session.model, "current")

        catalog = discover_provider_catalog(timeout=2.0, per_provider_limit=8)
        provider_name = {
            "anthropic": "Anthropic",
            "openai": "OpenAI",
            "gemini": "Gemini",
        }

        status_parts: list[str] = []
        for provider in ("anthropic", "openai", "gemini"):
            info = catalog.get(provider)
            if not info:
                continue

            if info.source and info.models:
                src_kind = "oauth" if "oauth" in info.source.lower() else "key"
                for model_id in info.models:
                    value = model_id if "/" in model_id else f"{provider}/{model_id}"
                    add(value, f"{provider_name[provider]} live · {src_kind}")
                status_parts.append(f"{provider}:{len(info.models)}")
            elif info.source and not info.models:
                status_parts.append(f"{provider}:auth")
            else:
                status_parts.append(f"{provider}:—")

        openai_info = catalog.get("openai")
        block_openai_recents = bool(
            openai_info
            and openai_info.source
            and "codex" in openai_info.source.lower()
            and not openai_info.models
        )
        for recent in get_settings().get_recent_models():
            value = recent.strip()
            lower = value.lower()
            if lower in {"openai/codex", "codex", "refresh", "--refresh", "rescan", "verify"}:
                continue
            if not re.match(r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9._:-]+$", value):
                continue
            if block_openai_recents and value.startswith("openai/"):
                continue
            add(value, "recent")

        fallbacks = {
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "openai": "openai/gpt-4o-mini",
            "gemini": "gemini/gemini-2.0-flash",
        }
        for provider, fallback_model in fallbacks.items():
            info = catalog.get(provider)
            if info and info.models:
                continue

            if info and info.source and provider == "openai" and "codex" in info.source.lower():
                continue

            if info and info.source:
                add(fallback_model, f"{provider_name[provider]} fallback · unverified")
            else:
                add(fallback_model, f"{provider_name[provider]} fallback · requires auth")

        codex_source = get_auth_manager().resolve_api_key("openai-codex")
        if codex_source:
            codex_models = [
                "openai-codex/gpt-5.3-codex",
                "openai-codex/gpt-5.2-codex",
                "openai-codex/gpt-5.1",
                "openai-codex/gpt-5.1-codex-max",
                "openai-codex/gpt-5.1-codex-mini",
            ]
            for m in codex_models:
                add(m, "Codex subscription · oauth")
            status_parts.append(f"codex:{len(codex_models)}")
        else:
            status_parts.append("codex:—")

        tab_count = 0
        if tabminion_running():
            services = discover_services()
            tab_count = len(services)
            for svc in services:
                add(f"tabminion/{svc['id']}", f"TabMinion live {svc['emoji']} {svc['name']}")
        status_parts.append(f"tab:{tab_count}" if tab_count else "tab:—")

        title = f"Select model (current: {self.session.model}) · " + " · ".join(status_parts)

        self._model_menu_cache_title = title
        self._model_menu_cache_options = options
        self._model_menu_cache_until = now + 30.0

        return title, options

    def _normalize_model_alias(self, model_str: str) -> str:
        raw = model_str.strip()
        lower = raw.lower()
        if lower in {"codex", "openai/codex", "openai-codex"}:
            _, options = self._model_menu_options(force_refresh=False)
            for value, _ in options:
                if value.startswith("openai-codex/") and "codex" in value.lower():
                    return value
            for value, _ in options:
                if value.startswith("openai-codex/"):
                    return value
            for value, _ in options:
                if value.startswith("openai/") and "codex" in value.lower():
                    return value
            return raw
        return raw

    def _handle_model(self, model_str: str) -> None:
        cmd = model_str.strip()
        if not cmd or cmd.lower() in {"refresh", "--refresh", "rescan", "verify", "check"}:
            force = cmd.lower() in {"refresh", "--refresh", "rescan", "verify", "check"}
            title, options = self._model_menu_options(force_refresh=force)
            lines = [f"**{title}**", "", "Available:"]
            for value, label in options:
                lines.append(f"- `{value}` — {label}")
            lines.append("")
            lines.append("Switch with: `/model <provider/model>`")
            self._chat("system", "\n".join(lines), fmt="markdown")
            return

        model_str = self._normalize_model_alias(cmd)
        if not re.match(r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9._:-]+$", model_str):
            self._chat(
                "error",
                "Invalid model format. Use `provider/model` (e.g. `openai/gpt-4o-mini`) or `/model`.",
                fmt="text",
            )
            return

        try:
            self.session.switch_model(model_str)
            self._model_menu_cache_until = 0.0
            self._chat("system", f"Switched to **{model_str}**", fmt="markdown")
            self._emit_status()
        except Exception as exc:
            self._chat("error", f"Model switch failed: {exc}", fmt="text")

    def _handle_image(self, path_str: str) -> None:
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            self._chat("error", f"File not found: {path}", fmt="text")
            return
        try:
            from PIL import Image

            img = Image.open(path)
            name = path.stem
            self.kernel.push_image(name, img)
            self._chat(
                "system",
                f"Loaded **{name}** ({img.size[0]}×{img.size[1]}, {img.mode}) → `images['{name}']`",
                fmt="markdown",
            )
        except Exception as exc:
            self._chat("error", f"Image load error: {exc}", fmt="text")

    def _handle_login(self, provider_arg: str) -> None:
        if not provider_arg:
            self._chat(
                "system",
                "Usage:\n"
                "- `/login anthropic` (OAuth)\n"
                "- `/login openai` (OAuth/Codex)\n"
                "- `/login openai sk-...` (API key)\n"
                "- `/login gemini AIza...` (API key)",
                fmt="markdown",
            )
            return

        parts = provider_arg.split(maxsplit=1)
        provider = parts[0].strip().lower()
        secret = parts[1].strip() if len(parts) > 1 else ""

        if provider == "anthropic":
            self._chat("system", "Starting Anthropic OAuth login…", fmt="text")
            self._run_login("anthropic")
            return

        if provider in {"openai", "openai-codex"}:
            if not secret:
                self._chat("system", "Starting OpenAI Codex OAuth login…", fmt="text")
                self._run_login("openai-codex")
                return

            auth = get_auth_manager()
            auth.set_api_key("openai", secret)
            self._model_menu_cache_until = 0.0

            active_provider = self.session.model.split("/", 1)[0] == "openai"
            if active_provider:
                self.session.auth_source = self.session._resolve_auth()
                self._emit_status()

            self._chat(
                "system",
                "✅ Stored **openai** API key in `auth.json` for reuse across sessions.",
                fmt="markdown",
            )
            return

        if provider in {"gemini", "google"}:
            normalized = "gemini" if provider == "google" else provider
            if not secret:
                self._chat(
                    "system",
                    "Usage: `/login <provider> <api_key>`\n"
                    "Examples:\n"
                    "- `/login openai` (OAuth)\n"
                    "- `/login openai sk-...` (API key)\n"
                    "- `/login gemini AIza...`",
                    fmt="markdown",
                )
                return

            auth = get_auth_manager()
            auth.set_api_key(normalized, secret)
            self._model_menu_cache_until = 0.0

            active_provider = self.session.model.split("/", 1)[0] == normalized
            if active_provider:
                self.session.auth_source = self.session._resolve_auth()
                self._emit_status()

            self._chat(
                "system",
                f"✅ Stored **{normalized}** API key in `auth.json` for reuse across sessions.",
                fmt="markdown",
            )
            return

        self._chat("error", "Unknown provider. Supported: `anthropic`, `openai`, `gemini`.", fmt="text")

    def _request_prompt(self, message: str, timeout_s: float = 300.0) -> str:
        token = secrets.token_hex(8)
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        waiter = _PromptWaiter(token=token, q=q)

        with self._prompt_lock:
            self._prompt_waiters[token] = waiter

        self._emit("auth_prompt", {"token": token, "message": message})

        try:
            return q.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("Login timed out or was cancelled") from exc
        finally:
            with self._prompt_lock:
                self._prompt_waiters.pop(token, None)

    def provide_prompt(self, token: str, value: str) -> bool:
        with self._prompt_lock:
            waiter = self._prompt_waiters.get(token)
        if not waiter:
            return False
        try:
            waiter.q.put_nowait(value)
            return True
        except queue.Full:
            return False

    def _run_login(self, provider: str) -> None:
        import webbrowser

        auth = get_auth_manager()

        def on_url(url: str) -> None:
            self._chat(
                "system",
                f"🔗 **Open this URL to authorize:**\n\n{url}\n\n_(attempting to open in browser…)_",
                fmt="markdown",
            )
            try:
                webbrowser.open(url)
            except Exception:
                pass

        def on_prompt(message: str) -> str:
            return self._request_prompt(message)

        def on_status(msg: str) -> None:
            self._chat("system", msg, fmt="text")

        try:
            if provider == "anthropic":
                auth.login_anthropic(on_url, on_prompt, on_status)
            elif provider == "openai-codex":
                auth.login_openai(on_url, on_prompt, on_status)
            else:
                raise RuntimeError(f"Unsupported OAuth provider: {provider}")

            self.session.auth_source = self.session._resolve_auth()
            self._model_menu_cache_until = 0.0
            self._emit_status()

            src = self.session.auth_source
            provider_label = "openai" if provider == "openai-codex" else provider
            if src:
                self._chat(
                    "system",
                    f"✅ **Logged in to {provider_label}**: {src.source}\n\n"
                    "Use `/auth` to verify and `/logout <provider>` to remove credentials.",
                    fmt="markdown",
                )
            else:
                self._chat("system", f"✅ Login completed for {provider}", fmt="text")
        except Exception as exc:
            self._chat("error", f"❌ Login failed: {exc}", fmt="text")

    def _handle_logout(self, provider: str) -> None:
        provider = provider.strip().lower()
        if provider == "google":
            provider = "gemini"
        if provider == "openai-codex":
            provider = "openai"

        if not provider:
            self._chat(
                "system",
                "Usage: `/logout <provider>`\nExamples: `/logout anthropic`, `/logout openai`, `/logout gemini`",
                fmt="text",
            )
            return

        auth = get_auth_manager()
        auth.resolve_api_key(provider)
        if provider == "openai":
            auth.resolve_api_key("openai-codex")

        provider_keys = [provider]
        if provider == "openai":
            provider_keys = ["openai-codex", "openai"]

        existing = [(k, auth.get(k) or {}) for k in provider_keys if auth.get(k)]
        if not existing:
            self._chat("system", f"No stored credentials for `{provider}`.", fmt="text")
            return

        env_var = ENV_KEY_MAP.get(provider)
        if env_var:
            for _, cred in existing:
                removed_key = ""
                if cred.get("type") == "oauth":
                    removed_key = cred.get("access", "")
                elif cred.get("type") == "api_key":
                    removed_key = cred.get("key", "")
                if removed_key and os.environ.get(env_var) == removed_key:
                    os.environ.pop(env_var, None)

        auth.logout(provider)
        self.session.auth_source = self.session._resolve_auth()
        src = self.session.auth_source

        if src:
            self._chat(
                "system",
                f"✅ Logged out of **{provider}**. Now using: **{src.source}**",
                fmt="markdown",
            )
        else:
            if provider == "openai":
                hint = "`/login openai` (OAuth) or `/login openai <api_key>`"
            else:
                hint = f"`/login {provider} <api_key>`"
            self._chat(
                "system",
                f"✅ Logged out of **{provider}**. No other credentials found — {hint}.",
                fmt="markdown",
            )

        self._model_menu_cache_until = 0.0
        self._emit_status()

    def _handle_auth(self) -> None:
        auth = get_auth_manager()
        summary = auth.auth_summary()

        if not summary:
            self._chat(
                "system",
                "**No auth configured.**\n\n"
                "Use `/login anthropic` or `/login openai` for OAuth, "
                "`/login <provider> <api_key>` to persist keys, "
                "or environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).",
                fmt="markdown",
            )
            return

        lines = ["**Auth sources** (highest priority first):\n"]
        lines.append("| Provider | Source | Subscription | Key Preview |")
        lines.append("|----------|--------|:------------:|-------------|")
        for entry in summary:
            sub = "✅ free" if entry["subscription"] == "yes" else "💰 billed"
            lines.append(
                f"| {entry['provider']} | {entry['source']} | {sub} | `{entry['key_preview']}` |"
            )

        src = self.session.auth_source
        if src:
            lines.append("")
            lines.append(f"**Active for {self.session.model}**: {src.source}")

        self._chat("system", "\n".join(lines), fmt="markdown")

    def _handle_tabminion(self) -> None:
        from aiipython.tabminion import status_summary

        self._chat("system", status_summary(), fmt="markdown")

    def _handle_tree(self) -> None:
        ckpt = self.kernel.checkpoints
        if not ckpt:
            self._chat("system", "(checkpoints not available)", fmt="text")
            return
        self._chat("system", f"```text\n{ckpt.show_tree()}\n```", fmt="markdown")

    def _handle_undo(self) -> None:
        ckpt = self.kernel.checkpoints
        if not ckpt:
            self._chat("system", "(checkpoints not available)", fmt="text")
            return
        nid = ckpt.undo(self.kernel, agent=self.agent)
        if nid:
            node = ckpt.nodes[nid]
            self._chat("system", f"✓ Restored to [{nid}]: {node.label}", fmt="text")
        else:
            self._chat("system", "Nothing to undo — no checkpoints yet.", fmt="text")

    def _handle_restore(self, arg: str) -> None:
        ckpt = self.kernel.checkpoints
        if not ckpt:
            self._chat("system", "(checkpoints not available)", fmt="text")
            return
        nid = arg.strip()
        if not nid:
            self._chat("system", "Usage: /restore <checkpoint_id>", fmt="text")
            return
        try:
            ckpt.restore(self.kernel, nid, agent=self.agent)
            node = ckpt.nodes[nid]
            self._chat("system", f"✓ Restored to [{nid}]: {node.label}", fmt="text")
        except KeyError as exc:
            self._chat("error", f"Error: {exc}", fmt="text")

    def _handle_fork(self, arg: str) -> None:
        ckpt = self.kernel.checkpoints
        if not ckpt:
            self._chat("system", "(checkpoints not available)", fmt="text")
            return
        label = arg.strip() or None
        nid = ckpt.fork(self.kernel, agent=self.agent, label=label)
        node = ckpt.nodes[nid]
        self._chat("system", f"✓ Forked: [{nid}] {node.label}", fmt="text")

    def get_inspector(self) -> dict[str, Any]:
        return {
            "snapshot": self.kernel.snapshot(),
            "history": self.kernel.history[-80:],
        }

    def abort_prompt(self) -> dict[str, Any]:
        """Request cancellation of an in-flight prompt_once turn."""
        self._prompt_cancel.set()
        return {"requested": True}

    def prompt_once(self, text: str, model: str | None = None) -> dict[str, Any]:
        """Run one aiipython reactive turn synchronously and return final assistant text.

        Used by the pi-native host custom provider bridge.
        """
        clean = (text or "").strip()
        if not clean:
            return {
                "assistant_markdown": "",
                "display_markdown": "",
                "react_cells": [],
                "aborted": False,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "model": self.session.model,
                "iterations": 0,
            }

        with self._prompt_once_lock:
            import dspy

            target_model = model.strip() if isinstance(model, str) else ""
            needs_lm = (not hasattr(dspy.settings, "lm")) or (dspy.settings.lm is None)

            if target_model and (needs_lm or target_model != self.session.model):
                self.session.switch_model(target_model)
            elif needs_lm:
                self.session.switch_model(self.session.model)

            self._prompt_cancel.clear()
            start_wire_idx = len(self.session.wire_log.entries)
            steps: list[ReactionStep] = []
            aborted = False

            self.kernel.push_user_input(clean)
            try:
                self.agent.react(
                    on_step=lambda s: steps.append(s),
                    should_abort=self._prompt_cancel.is_set,
                )
            except PromptAborted:
                aborted = True

            new_entries = self.session.wire_log.entries[start_wire_idx:]
            prompt_tokens = sum(int((e.usage or {}).get("prompt_tokens", 0) or 0) for e in new_entries)
            completion_tokens = sum(int((e.usage or {}).get("completion_tokens", 0) or 0) for e in new_entries)
            cost = 0.0
            for e in new_entries:
                u = e.usage or {}
                c = u.get("cost", 0)
                try:
                    cost += float(c or 0.0)
                except Exception:
                    pass

            assistant_markdown = ""
            if steps:
                assistant_markdown = steps[-1].markdown or ""

            # Build a display transcript so pi-native can show the full reactive loop,
            # not just the final no-code assistant turn.
            display_parts: list[str] = []
            react_cells: list[dict[str, str]] = []
            for step in steps:
                shown_markdown = step.markdown
                if step.blocks:
                    prose = _without_fenced_code(step.markdown)
                    shown_markdown = prose or "Running requested python blocks…"
                shown_markdown = (shown_markdown or "").strip()
                if shown_markdown:
                    display_parts.append(shown_markdown)

                for idx, block in enumerate(step.blocks, start=1):
                    entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                    out = self._format_exec_output(entry)
                    code = _abbrev_multiline(block.code)
                    react_cells.append({"code": code, "output": out})
                    display_parts.append(
                        f"▶ python #{idx}\n"
                        f"```python\n{code}\n```\n"
                        f"```text\n{out}\n```"
                    )

            display_markdown = "\n\n".join(p for p in display_parts if p).strip()
            if not display_markdown:
                display_markdown = assistant_markdown

            return {
                "assistant_markdown": assistant_markdown,
                "display_markdown": display_markdown,
                "react_cells": react_cells,
                "aborted": aborted,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": cost,
                },
                "model": self.session.model,
                "iterations": len(steps),
            }

    def prompt_stream(
        self,
        text: str,
        model: str | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        """Like prompt_once(), but emits incremental LM chunks as backend events."""
        clean = (text or "").strip()
        if not clean:
            return {
                "assistant_markdown": "",
                "display_markdown": "",
                "react_cells": [],
                "aborted": False,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "model": self.session.model,
                "iterations": 0,
                "stream_id": (stream_id or "").strip() or "",
            }

        sid = (stream_id or "").strip() or secrets.token_hex(8)

        with self._prompt_once_lock:
            import dspy

            target_model = model.strip() if isinstance(model, str) else ""
            needs_lm = (not hasattr(dspy.settings, "lm")) or (dspy.settings.lm is None)

            if target_model and (needs_lm or target_model != self.session.model):
                self.session.switch_model(target_model)
            elif needs_lm:
                self.session.switch_model(self.session.model)

            self._prompt_cancel.clear()
            start_wire_idx = len(self.session.wire_log.entries)
            steps: list[ReactionStep] = []
            aborted = False

            self.kernel.push_user_input(clean)
            self._emit("prompt_stream_start", {"stream_id": sid})

            def _on_stream_chunk(chunk: str, is_last: bool) -> None:
                self._emit(
                    "prompt_stream_delta",
                    {
                        "stream_id": sid,
                        "chunk": chunk,
                        "is_last": bool(is_last),
                    },
                )

            def _on_step(step: ReactionStep) -> None:
                steps.append(step)
                blocks_payload: list[dict[str, Any]] = []
                if step.blocks:
                    for idx, block in enumerate(step.blocks, start=1):
                        entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                        blocks_payload.append(
                            {
                                "index": idx,
                                "code": _abbrev_multiline(block.code),
                                "output": self._format_exec_output(entry),
                                "success": bool(entry.get("success", True)),
                            }
                        )
                self._emit(
                    "prompt_stream_step",
                    {
                        "stream_id": sid,
                        "iteration": int(step.iteration),
                        "is_final": bool(step.is_final),
                        "blocks": blocks_payload,
                    },
                )

            try:
                self.agent.react(
                    on_step=_on_step,
                    on_stream_chunk=_on_stream_chunk,
                    should_abort=self._prompt_cancel.is_set,
                )
            except PromptAborted:
                aborted = True
            finally:
                self._emit("prompt_stream_end", {"stream_id": sid})

            new_entries = self.session.wire_log.entries[start_wire_idx:]
            prompt_tokens = sum(int((e.usage or {}).get("prompt_tokens", 0) or 0) for e in new_entries)
            completion_tokens = sum(int((e.usage or {}).get("completion_tokens", 0) or 0) for e in new_entries)
            cost = 0.0
            for e in new_entries:
                u = e.usage or {}
                c = u.get("cost", 0)
                try:
                    cost += float(c or 0.0)
                except Exception:
                    pass

            assistant_markdown = ""
            if steps:
                assistant_markdown = steps[-1].markdown or ""

            display_parts: list[str] = []
            react_cells: list[dict[str, str]] = []
            for step in steps:
                shown_markdown = step.markdown
                if step.blocks:
                    prose = _without_fenced_code(step.markdown)
                    shown_markdown = prose or "Running requested python blocks…"
                shown_markdown = (shown_markdown or "").strip()
                if shown_markdown:
                    display_parts.append(shown_markdown)

                for idx, block in enumerate(step.blocks, start=1):
                    entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                    out = self._format_exec_output(entry)
                    code = _abbrev_multiline(block.code)
                    react_cells.append({"code": code, "output": out})
                    display_parts.append(
                        f"▶ python #{idx}\n"
                        f"```python\n{code}\n```\n"
                        f"```text\n{out}\n```"
                    )

            display_markdown = "\n\n".join(p for p in display_parts if p).strip()
            if not display_markdown:
                display_markdown = assistant_markdown

            return {
                "assistant_markdown": assistant_markdown,
                "display_markdown": display_markdown,
                "react_cells": react_cells,
                "aborted": aborted,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": cost,
                },
                "model": self.session.model,
                "iterations": len(steps),
                "stream_id": sid,
            }


class PiRpcServer:
    """Single-client JSON-RPC-over-TCP server for the pi-tui frontend."""

    def __init__(self, backend: PiTuiBackend, host: str = "127.0.0.1", port: int = 0) -> None:
        self.backend = backend
        self.host = host
        self.port = port

        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._conn_lock = threading.Lock()
        self._conn_file_w: Any = None

    def start(self) -> tuple[str, int]:
        if self._sock is not None:
            raise RuntimeError("server already started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.5)

        self._sock = sock
        self.host, self.port = sock.getsockname()[0], int(sock.getsockname()[1])

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        return self.host, self.port

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        with self._conn_lock:
            fw = self._conn_file_w
            self._conn_file_w = None
        if fw is not None:
            try:
                fw.close()
            except Exception:
                pass

        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)

        self.backend.detach_transport()

    # ── internals ──────────────────────────────────────────────────

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                self._serve_connection(conn)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                self.backend.detach_transport()

    def _serve_connection(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        file_r = conn.makefile("r", encoding="utf-8", newline="\n")
        file_w = conn.makefile("w", encoding="utf-8", newline="\n")

        def send_raw(obj: dict[str, Any]) -> None:
            text = json.dumps(obj, ensure_ascii=False, default=str)
            with self._conn_lock:
                if self._conn_file_w is None:
                    return
                self._conn_file_w.write(text + "\n")
                self._conn_file_w.flush()

        with self._conn_lock:
            self._conn_file_w = file_w
        self.backend.attach_transport(send_raw)

        def _handle_request(req_id: Any, method: str, params: dict[str, Any]) -> None:
            try:
                result = self._dispatch(method, params)
                send_raw({"type": "response", "id": req_id, "ok": True, "result": result})
            except Exception as exc:
                send_raw(
                    {
                        "type": "response",
                        "id": req_id,
                        "ok": False,
                        "error": {"message": str(exc), "type": type(exc).__name__},
                    }
                )

        while not self._stop.is_set():
            line = file_r.readline()
            if not line:
                break

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") != "request":
                continue

            req_id = msg.get("id")
            method = str(msg.get("method", ""))
            params = msg.get("params") or {}

            # Long-running prompt methods run in a worker thread so this
            # connection can still process abort/ping/other requests.
            if method in {"prompt_once", "prompt_stream"}:
                threading.Thread(
                    target=_handle_request,
                    args=(req_id, method, params),
                    daemon=True,
                ).start()
            else:
                _handle_request(req_id, method, params)

        with self._conn_lock:
            self._conn_file_w = None

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "hello":
            return self.backend.hello()
        if method == "submit_input":
            return self.backend.submit_input(str(params.get("text", "")))
        if method == "prompt_once":
            model = params.get("model")
            return self.backend.prompt_once(
                text=str(params.get("text", "")),
                model=str(model) if model is not None else None,
            )
        if method == "prompt_stream":
            model = params.get("model")
            return self.backend.prompt_stream(
                text=str(params.get("text", "")),
                model=str(model) if model is not None else None,
                stream_id=str(params.get("stream_id", "") or ""),
            )
        if method == "abort_prompt":
            return self.backend.abort_prompt()
        if method == "transform_at_refs":
            return self.backend.transform_at_refs(str(params.get("text", "")))
        if method == "run_ipython_shell":
            return self.backend.run_ipython_shell(
                str(params.get("command", "")),
                exclude_from_context=bool(params.get("excludeFromContext", False)),
            )
        if method == "provide_prompt":
            ok = self.backend.provide_prompt(
                str(params.get("token", "")),
                str(params.get("value", "")),
            )
            return {"accepted": ok}
        if method == "get_inspector":
            return self.backend.get_inspector()
        if method == "ping":
            return {"pong": True}
        raise RuntimeError(f"Unknown method: {method}")


def _snapshot_terminal_state() -> tuple[int, Any] | None:
    """Capture terminal mode so we can restore it if Node exits uncleanly."""
    if not sys.stdin.isatty():
        return None
    try:
        import termios

        fd = sys.stdin.fileno()
        return (fd, termios.tcgetattr(fd))
    except Exception:
        return None


def _restore_terminal_state(snapshot: tuple[int, Any] | None) -> None:
    """Best-effort terminal cleanup after pi native/tui subprocess exits."""
    # Reset common TUI modes in case child process exited before full cleanup.
    try:
        if sys.stdout.isatty():
            sys.stdout.write(
                "\x1b[?2026l"  # synchronized output off
                "\x1b[?2004l"  # bracketed paste off
                "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1004l\x1b[?1006l"  # mouse/focus off
                "\x1b[?47l\x1b[?1047l\x1b[?1049l"  # leave alternate screen buffers
                "\x1b[?25h"  # show cursor
            )
            sys.stdout.flush()
    except Exception:
        pass

    if snapshot is not None:
        fd, attrs = snapshot
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception:
            pass

    # Always enforce a sane tty baseline afterwards.
    stty = shutil.which("stty")
    if stty and sys.stdin.isatty():
        try:
            subprocess.run([stty, "sane"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def _ensure_exit_tty_cleanup() -> None:
    """Register one-shot process-exit tty cleanup (helps when leaving IPython)."""
    global _TTY_EXIT_CLEANUP_REGISTERED
    if _TTY_EXIT_CLEANUP_REGISTERED:
        return
    _TTY_EXIT_CLEANUP_REGISTERED = True

    def _cleanup() -> None:
        _restore_terminal_state(None)

    atexit.register(_cleanup)


def _run_node_ui(node: str, entry: Path, env: dict[str, str]) -> None:
    """Run Node UI and always restore terminal state on exit."""
    _ensure_exit_tty_cleanup()
    tty_snapshot = _snapshot_terminal_state()
    proc = subprocess.Popen([node, str(entry)], env=env)
    try:
        proc.wait()
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _restore_terminal_state(tty_snapshot)


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent / "pi_tui_frontend"


def _ensure_frontend_dependencies(frontend_dir: Path) -> None:
    pkg = frontend_dir / "package.json"
    deps_marker = frontend_dir / "node_modules" / "@mariozechner" / "pi-tui"
    if deps_marker.exists():
        return
    if not pkg.exists():
        raise RuntimeError(f"pi-tui frontend package.json missing: {pkg}")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError(
            "npm is required for the pi-tui frontend. Install Node.js/npm, "
            "or run with AIIPYTHON_UI=textual."
        )

    proc = subprocess.run(
        [npm, "install", "--no-audit", "--no-fund"],
        cwd=str(frontend_dir),
        check=False,
    )
    if proc.returncode != 0 or not deps_marker.exists():
        raise RuntimeError(
            "Failed to install pi-tui frontend dependencies. "
            f"Try manually: cd {frontend_dir} && npm install"
        )


def run_pi_tui(session: Session) -> None:
    """Run the pi-tui frontend for the given aiipython session."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "node is required for AIIPYTHON_UI=pi-tui. "
            "Install Node.js or use AIIPYTHON_UI=textual."
        )

    frontend_dir = _frontend_dir()
    _ensure_frontend_dependencies(frontend_dir)

    entry = frontend_dir / "pi_tui_app.mjs"
    if not entry.exists():
        raise RuntimeError(f"pi-tui frontend entry not found: {entry}")

    backend = PiTuiBackend(session)
    server = PiRpcServer(backend)
    host, port = server.start()

    env = os.environ.copy()
    env["AIIPYTHON_RPC_HOST"] = host
    env["AIIPYTHON_RPC_PORT"] = str(port)

    try:
        _run_node_ui(node, entry, env)
    finally:
        server.stop()
        backend.close()


def _native_host_dir() -> Path:
    return Path(__file__).resolve().parent / "pi_native_host"


def _ensure_native_host_dependencies(native_dir: Path) -> None:
    pkg = native_dir / "package.json"
    deps_marker = native_dir / "node_modules" / "@mariozechner" / "pi-coding-agent"
    if deps_marker.exists():
        return
    if not pkg.exists():
        raise RuntimeError(f"pi-native host package.json missing: {pkg}")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError(
            "npm is required for AIIPYTHON_UI=pi-native. Install Node.js/npm, "
            "or run with AIIPYTHON_UI=pi-tui/textual."
        )

    proc = subprocess.run(
        [npm, "install", "--no-audit", "--no-fund"],
        cwd=str(native_dir),
        check=False,
    )
    if proc.returncode != 0 or not deps_marker.exists():
        raise RuntimeError(
            "Failed to install pi-native host dependencies. "
            f"Try manually: cd {native_dir} && npm install"
        )


def run_pi_native(session: Session) -> None:
    """Run real Pi InteractiveMode frontend against aiipython backend bridge."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "node is required for AIIPYTHON_UI=pi-native. "
            "Install Node.js or use AIIPYTHON_UI=pi-tui/textual."
        )

    native_dir = _native_host_dir()
    _ensure_native_host_dependencies(native_dir)

    entry = native_dir / "pi_native_host.mjs"
    if not entry.exists():
        raise RuntimeError(f"pi-native host entry not found: {entry}")

    backend = PiTuiBackend(session)
    server = PiRpcServer(backend)
    host, port = server.start()

    env = os.environ.copy()
    env["AIIPYTHON_RPC_HOST"] = host
    env["AIIPYTHON_RPC_PORT"] = str(port)

    try:
        _run_node_ui(node, entry, env)
    finally:
        server.stop()
        backend.close()
