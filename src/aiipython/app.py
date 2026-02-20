"""Textual TUI — pi-inspired chat interface."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Collapsible, Input, Static

from aiipython import __version__
from aiipython.agent import ReactionStep
from aiipython.auth import ENV_KEY_MAP, get_auth_manager
from aiipython.context import summarize_items
from aiipython.session import Session
from aiipython.wire import WireEntry, format_wire_messages


# ── Helpers ─────────────────────────────────────────────────────────

UI_MAX_LINE_CHARS = 140
UI_MAX_BLOCK_LINES = 80
SNAPSHOT_TEXT_CHAR_LIMIT = 20_000
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


def _fmt_tokens(n: int) -> str:
    """Format token count like pi: 1234 → 1.2k, 12345 → 12k."""
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    return f"{n / 1_000_000:.1f}M"


def _abbrev_line(line: str, max_len: int = UI_MAX_LINE_CHARS) -> str:
    """Abbreviate a single long line for compact terminal UI display."""
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
    """Abbreviate long lines and very long blocks for chat widgets."""
    lines = text.splitlines() or [""]
    out = [_abbrev_line(line, max_len=max_len) for line in lines[:max_lines]]
    if len(lines) > max_lines:
        out.append(f"… [{len(lines) - max_lines} more lines]")
    return "\n".join(out)


def _without_fenced_code(markdown: str) -> str:
    """Return markdown with fenced code blocks removed."""
    cleaned = re.sub(r"```[\w-]*\s*\n.*?```", "", markdown, flags=re.DOTALL)
    return cleaned.strip()


def _build_aiipython_theme(background: str) -> Theme:
    """Build app theme with configurable background.

    background can be:
      - "ansi_default" (terminal-native)
      - "#RRGGBB" (forced solid color)
    """
    return Theme(
        name="aiipython",
        primary="#8abeb7",
        secondary="#5f87ff",
        accent="#8abeb7",
        foreground="#d4d4d4",
        background=background,
        panel=background,
        surface="#1e1e24",
        warning="#ffff00",
        error="#cc6666",
        success="#b5bd68",
        dark=True,
    )


# ── Widgets ─────────────────────────────────────────────────────────

class ChatMessage(Static):
    """A single message in the chat."""

    def __init__(self, content: str, role: str = "user", **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.role = role
        self.add_class("msg", f"msg-{role}")

    def on_mount(self) -> None:
        if self.role in ("assistant", "system"):
            self.update(Markdown(self.content))
        else:
            self.update(self.content)


class CodeChunk(Static):
    """One executable python block extracted from the assistant markdown."""

    def __init__(self, title: str, code: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._code = code
        self.add_class("code-chunk")

    def on_mount(self) -> None:
        body = f"{self._title}\n{self._code}" if self._title else self._code
        self.update(body)


class ExecOutput(Static):
    """Code execution output — pi-style colored box."""

    def __init__(self, text: str, success: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text
        self.add_class("exec-output", "exec-success" if success else "exec-error")

    def on_mount(self) -> None:
        self.update(self._text)


class VarsPanel(Static):
    def refresh_vars(self, snapshot: dict[str, str]) -> None:
        if not snapshot:
            self.update(Text("(empty)", style="dim"))
            return
        lines = Text()
        lines.append("── Namespace ──\n", style="bold #8abeb7")
        for name, summary in snapshot.items():
            lines.append(f"  {name}", style="#f0c674")
            lines.append(f"  {summary}\n", style="#808080")
        self.update(lines)


class ReplPanel(Static):
    def refresh_repl(self, history: list[dict]) -> None:
        meaningful = [h for h in history if h.get("tag") in ("user", "agent")]
        if not meaningful:
            self.update(Text("(no activity yet)", style="dim"))
            return
        lines = Text()
        lines.append("── REPL ──\n", style="bold #8abeb7")
        for h in meaningful[-30:]:
            tag = h.get("tag", "?")
            tag_style = "#f0c674" if tag == "user" else "#b5bd68"
            success = h.get("success", True)
            lines.append(f"  [{tag}] ", style=tag_style)
            code = h["code"][:120] + ("…" if len(h["code"]) > 120 else "")
            lines.append(f"{code}\n", style="#d4d4d4" if success else "#cc6666")
            for key, style, prefix in [
                ("stdout", "#808080", "→"),
                ("stderr", "#cc6666", "⚠"),
                ("result", "#808080", "="),
            ]:
                val = h.get(key, "").strip()
                if val:
                    val = val if len(val) <= 200 else f"({len(val)} chars)"
                    lines.append(f"    {prefix} {val}\n", style=style)
            if not success:
                error_msg = h.get("error", "FAILED")
                lines.append(f"    ✗ {error_msg}\n", style="#cc6666")
        self.update(lines)


class PiFooter(Widget):
    """Pi-style 2-line footer: path on line 1, stats + model on line 2."""

    DEFAULT_CSS = """
    PiFooter {
        height: 2;
        color: #666666;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd: str = ""
        self._stats: str = ""
        self._model: str = ""
        self._auth_label: str = ""
        self._refresh_cwd()

    def _refresh_cwd(self) -> None:
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home):]
        self._cwd = cwd

    def update_stats(
        self,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_cost: float = 0.0,
        duration_ms: float = 0.0,
        auth_label: str = "",
    ) -> None:
        self._model = model
        self._auth_label = auth_label

        parts: list[str] = []
        if prompt_tokens or completion_tokens:
            parts.append(f"↑{_fmt_tokens(prompt_tokens)} ↓{_fmt_tokens(completion_tokens)}")
        if total_cost > 0:
            parts.append(f"${total_cost:.3f}")
        if duration_ms > 0:
            parts.append(f"{duration_ms / 1000:.1f}s")
        self._stats = " ".join(parts)

        self._refresh_cwd()
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width or 80

        text = Text()

        # Line 1: cwd
        cwd = self._cwd
        if len(cwd) > width:
            half = width // 2 - 2
            cwd = cwd[:half] + "…" + cwd[-(half - 1):]
        text.append(cwd, style="#666666")
        text.append("\n")

        # Line 2: stats (left) ... auth + model (right)
        left = self._stats

        # Build right side: auth label + model
        right_parts: list[str] = []
        if self._auth_label:
            right_parts.append(self._auth_label)
        if self._model:
            right_parts.append(self._model)
        right = " · ".join(right_parts)

        left_len = len(left)
        right_len = len(right)

        if left_len + right_len + 2 <= width:
            gap = width - left_len - right_len
            text.append(left, style="#666666")
            text.append(" " * gap)
            # Auth label gets a different color if subscription
            if self._auth_label and self._model:
                auth_style = "#b5bd68" if "oauth" in self._auth_label.lower() else "#f0c674"
                text.append(self._auth_label, style=auth_style)
                text.append(" · ", style="#666666")
                text.append(self._model, style="#666666")
            else:
                text.append(right, style="#666666")
        elif left_len + 2 + right_len <= width + 10:
            text.append(left, style="#666666")
            text.append("  ")
            text.append(right, style="#666666")
        else:
            text.append(left, style="#666666")

        return text


# ── App ─────────────────────────────────────────────────────────────

class AiiPythonApp(App):
    TITLE = "aiipython"
    CSS_PATH = "app.tcss"
    theme = "aiipython"
    ansi_color = True

    def _watch_theme(self, theme_name: str) -> None:
        # Textual sets ansi_color=True only for the literal "textual-ansi"
        # theme. We use our own theme name, so force ANSI mode back on.
        super()._watch_theme(theme_name)
        self.ansi_color = True

    BINDINGS = [
        Binding("ctrl+i", "toggle_inspector", "Inspector", show=False),
        Binding("ctrl+b", "toggle_debug", "Debug", show=False),
        Binding("ctrl+w", "toggle_wire", "Wire", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, session: Session, **kwargs) -> None:
        super().__init__(**kwargs)

        # Background strategy:
        # - AIIPYTHON_BG=ansi_default (default): use terminal native background
        # - AIIPYTHON_BG=#RRGGBB: force a solid background color
        #   (PYCODE_BG is supported as a legacy alias)
        bg = (
            os.environ.get("AIIPYTHON_BG")
            or os.environ.get("PYCODE_BG")
            or "ansi_default"
        ).strip() or "ansi_default"
        self.register_theme(_build_aiipython_theme(bg))
        self.theme = "aiipython"

        self.session = session
        self._inspector_visible = False
        self._debug_visible = False
        self._wire_visible = False
        self._total_prompt = 0
        self._total_completion = 0
        self._total_cost = 0.0
        self._last_duration = 0.0
        # Wire panel: track streaming widgets by entry_id
        self._wire_stream_widgets: dict[int, Static] = {}
        self._wire_stream_text: dict[int, str] = {}
        # Login flow state
        self._login_waiting_for_code = False
        self._login_prompt_event: Any = None
        self._login_prompt_result: list[str] = []

        # Generic interactive menu state (numbered selections via input)
        self._menu_active = False
        self._menu_title = ""
        self._menu_options: list[tuple[str, str]] = []  # (value, label)
        self._menu_on_select: Callable[[str], None] | None = None

        # Model menu discovery cache
        self._model_menu_cache_title = ""
        self._model_menu_cache_options: list[tuple[str, str]] = []
        self._model_menu_cache_until = 0.0

    @property
    def kernel(self):
        return self.session.kernel

    @property
    def agent(self):
        return self.session.agent

    def _call_from_thread_safe(self, callback, *args) -> None:
        """Call into UI thread, ignoring late callbacks after app exits."""
        try:
            if self.is_running:
                self.call_from_thread(callback, *args)
        except RuntimeError:
            # Happens if a worker finishes while the app is closing.
            pass

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield VerticalScroll(id="chat-log")
            with VerticalScroll(id="inspector"):
                yield VarsPanel(id="vars-panel")
                yield ReplPanel(id="repl-panel")
            yield VerticalScroll(id="debug-log")
            yield VerticalScroll(id="wire-log")
        with Vertical(id="bottom-bar"):
            yield Input(
                placeholder="Type a message… (@file, @clip, @ + paste, /model, /context)",
                id="user-input",
            )
            yield PiFooter(id="pi-footer")

    def on_mount(self) -> None:
        self.query_one("#user-input", Input).focus()
        self.query_one("#inspector").display = False
        self.query_one("#debug-log").display = False
        self.query_one("#wire-log").display = False

        # Wire callbacks
        self.session.wire_log.on_request = lambda e: self._call_from_thread_safe(
            self._wire_request, e)
        self.session.wire_log.on_chunk = lambda e, c: self._call_from_thread_safe(
            self._wire_chunk, e, c)
        self.session.wire_log.on_done = lambda e: self._call_from_thread_safe(
            self._wire_done, e)

        self._update_status()

        # Welcome message (pi-style: name + version + keybinding hints)
        chat = self.query_one("#chat-log")

        # Auth info line
        auth_info = ""
        if self.session.auth_source:
            src = self.session.auth_source
            if src.is_subscription:
                auth_info = f"\n🔑 **{src.provider}**: {src.source} *(subscription — no per-call cost)*"
            else:
                auth_info = f"\n🔑 **{src.provider}**: {src.source} *(billed per API call)*"
        else:
            auth_info = "\n⚠️ No auth configured — use `/login` or set an API key env var"

        # Check TabMinion
        from aiipython.tabminion import is_running as _tm_check
        tm_info = ""
        if _tm_check():
            from aiipython.tabminion import discover_services
            svcs = discover_services()
            if svcs:
                names = ", ".join(f"{s['emoji']}{s['name']}" for s in svcs)
                tm_info = f"\n🌐 **TabMinion**: {names} — `/tabminion` for details"
            else:
                tm_info = "\n🌐 **TabMinion** running (no AI tabs detected)"

        chat.mount(ChatMessage(
            f"**aiipython** v{__version__}{auth_info}{tm_info}\n\n"
            "`ctrl+c` quit · "
            "`ctrl+i` inspector · "
            "`ctrl+b` debug · "
            "`ctrl+w` wire\n"
            "Use `@file.py` (live) · `@clip` or `@ `+paste (snapshot)\n"
            "`/model` `/context` `/image` `/login` `/logout` `/auth` `/tabminion`",
            role="system",
        ))

    # ── status ──────────────────────────────────────────────────────
    def _update_status(self) -> None:
        auth_label = ""
        if self.session.auth_source:
            src = self.session.auth_source
            if src.is_subscription:
                auth_label = f"🔑 {src.source}"
            else:
                auth_label = f"🔑 {src.source}"

        self.query_one("#pi-footer", PiFooter).update_stats(
            model=self.session.model,
            prompt_tokens=self._total_prompt,
            completion_tokens=self._total_completion,
            total_cost=self._total_cost,
            duration_ms=self._last_duration,
            auth_label=auth_label,
        )

    # ── panel toggles ───────────────────────────────────────────────
    def action_toggle_inspector(self) -> None:
        self._inspector_visible = not self._inspector_visible
        self.query_one("#inspector").display = self._inspector_visible
        if self._inspector_visible:
            self._refresh_inspector()

    def action_toggle_debug(self) -> None:
        self._debug_visible = not self._debug_visible
        self.query_one("#debug-log").display = self._debug_visible

    def action_toggle_wire(self) -> None:
        self._wire_visible = not self._wire_visible
        self.query_one("#wire-log").display = self._wire_visible

    def _refresh_inspector(self) -> None:
        self.query_one("#vars-panel", VarsPanel).refresh_vars(self.kernel.snapshot())
        self.query_one("#repl-panel", ReplPanel).refresh_repl(self.kernel.history)
        self.query_one("#inspector").scroll_end(animate=False)

    # ── wire callbacks ──────────────────────────────────────────────

    def _wire_request(self, entry: WireEntry) -> None:
        """Called when an LM request starts — show request details."""
        wire_log = self.query_one("#wire-log")
        eid = entry.entry_id

        # Header
        ts_short = entry.ts[11:19] if len(entry.ts) > 19 else entry.ts
        header = Text()
        header.append(f"━━ #{eid} ", style="bold #505050")
        header.append(entry.model, style="bold #5f87ff")
        header.append(f"  {ts_short} ", style="#666666")
        header.append("━" * 30, style="#505050")
        wire_log.mount(Static(header, classes="wire-header"))

        # Params
        if entry.kwargs:
            params_str = " · ".join(
                f"{k}={json.dumps(v, default=str)}" for k, v in entry.kwargs.items()
                if k != "model"
            )
            if params_str:
                wire_log.mount(Static(
                    Text(f"  {params_str}", style="#666666"),
                    classes="wire-params",
                ))

        # Request messages (collapsible)
        msg_display = format_wire_messages(entry.messages)
        total_chars = len(msg_display)
        wire_log.mount(Collapsible(
            Static(msg_display, classes="wire-msg-content"),
            title=f"📤 Request · {len(entry.messages)} msgs · {total_chars:,} chars",
            collapsed=True,
        ))

        # Streaming area
        stream_widget = Static(
            Text("⏳ waiting for response…", style="italic #666666"),
            classes="wire-stream",
        )
        wire_log.mount(stream_widget)

        self._wire_stream_widgets[eid] = stream_widget
        self._wire_stream_text[eid] = ""

        wire_log.scroll_end(animate=False)

    def _wire_chunk(self, entry: WireEntry, chunk: str) -> None:
        """Called for each streaming chunk — update the live display."""
        eid = entry.entry_id
        if eid not in self._wire_stream_text:
            return

        self._wire_stream_text[eid] += chunk
        stream_widget = self._wire_stream_widgets.get(eid)
        if stream_widget:
            stream_widget.update(self._wire_stream_text[eid])

        wire_log = self.query_one("#wire-log")
        wire_log.scroll_end(animate=False)

    def _wire_done(self, entry: WireEntry) -> None:
        """Called when an LM response completes — show footer and response."""
        wire_log = self.query_one("#wire-log")
        eid = entry.entry_id

        # Update streaming area with final text
        stream_widget = self._wire_stream_widgets.get(eid)
        final_text = self._wire_stream_text.get(eid, "") or entry.full_response
        if stream_widget:
            if final_text:
                stream_widget.update(final_text)
            else:
                stream_widget.update(
                    Text("(no response text)", style="italic #666666")
                )

        # Footer
        footer = Text()
        footer.append("◀ ", style="bold #5f87ff")
        footer.append(f"{entry.duration_ms:.0f}ms", style="#808080")
        if entry.usage:
            p = entry.usage.get("prompt_tokens", "?")
            c = entry.usage.get("completion_tokens", "?")
            t = entry.usage.get("total_tokens", "?")
            footer.append(f"  ↑{p} ↓{c} ({t} total)", style="#808080")
        if entry.error:
            footer.append(f"\n  ERROR: {entry.error}", style="bold #cc6666")
        wire_log.mount(Static(footer, classes="wire-footer"))

        # Full response (collapsible)
        if entry.full_response:
            wire_log.mount(Collapsible(
                Static(entry.full_response, classes="wire-resp-content"),
                title=f"📥 Full Response · {len(entry.full_response):,} chars",
                collapsed=True,
            ))

        # Separator
        wire_log.mount(Static(
            Text("─" * 60, style="#505050"),
            classes="wire-separator",
        ))

        wire_log.scroll_end(animate=False)

        # Update totals
        self._total_prompt += entry.usage.get("prompt_tokens", 0)
        self._total_completion += entry.usage.get("completion_tokens", 0)
        self._total_cost += entry.usage.get("cost", 0) or 0
        self._last_duration = entry.duration_ms
        self._update_status()

        # Cleanup tracking
        self._wire_stream_widgets.pop(eid, None)
        self._wire_stream_text.pop(eid, None)

    # ── interactive menus ───────────────────────────────────────────
    def _open_menu(
        self,
        title: str,
        options: list[tuple[str, str]],
        on_select: Callable[[str], None],
    ) -> None:
        """Show a numbered interactive menu in chat.

        The next input line is interpreted as selection:
          - number (1..N)
          - exact value (e.g. `tabminion/claude`)
          - `q` / `cancel` to dismiss
        """
        if not options:
            chat = self.query_one("#chat-log")
            chat.mount(ChatMessage("No options available.", role="system"))
            chat.scroll_end(animate=False)
            return

        self._menu_active = True
        self._menu_title = title
        self._menu_options = options
        self._menu_on_select = on_select

        lines = [f"**{title}**", ""]
        for idx, (value, label) in enumerate(options, start=1):
            lines.append(f"{idx}. `{value}` — {label}")
        lines.append("")
        lines.append("Type a number, paste a value, or `q` to cancel.")

        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage("\n".join(lines), role="system"))
        chat.scroll_end(animate=False)

        inp = self.query_one("#user-input", Input)
        inp.placeholder = "Choose option number (or q to cancel)…"

    def _close_menu(self) -> None:
        self._menu_active = False
        self._menu_title = ""
        self._menu_options = []
        self._menu_on_select = None
        self.query_one("#user-input", Input).placeholder = (
            "Type a message… (@file, @clip, @ + paste, /model, /context)"
        )

    def _handle_menu_input(self, text: str) -> None:
        chat = self.query_one("#chat-log")

        if text.lower() in {"q", "quit", "cancel", "/cancel", "esc"}:
            chat.mount(ChatMessage("Menu cancelled.", role="system"))
            chat.scroll_end(animate=False)
            self._close_menu()
            return

        selected_value: str | None = None

        # Numeric selection
        if text.isdigit():
            idx = int(text)
            if 1 <= idx <= len(self._menu_options):
                selected_value = self._menu_options[idx - 1][0]

        # Direct value selection
        if selected_value is None:
            values = [v for v, _ in self._menu_options]
            if text in values:
                selected_value = text

        if selected_value is None:
            chat.mount(ChatMessage(
                f"Invalid selection: `{text}`. Choose 1..{len(self._menu_options)} or `q`.",
                role="error",
            ))
            chat.scroll_end(animate=False)
            return

        handler = self._menu_on_select
        self._close_menu()
        if handler:
            handler(selected_value)

    # ── input handling ──────────────────────────────────────────────
    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_text = event.value
        text = raw_text.strip()
        if not text:
            return
        event.input.value = ""

        # If we're waiting for an OAuth code, route input there
        if self._login_waiting_for_code and self._login_prompt_event:
            self._login_prompt_result.append(text)
            self._login_prompt_event.set()
            self._login_waiting_for_code = False
            return

        # Interactive menu selection mode
        if self._menu_active:
            self._handle_menu_input(text)
            return

        # Snapshot paste shortcut:
        #   type "@ " then Ctrl+Shift+V (or any paste), press Enter.
        raw_lstrip = raw_text.lstrip()
        if raw_lstrip.startswith("@ "):
            chat = self.query_one("#chat-log")
            payload = raw_lstrip[2:]
            chat.mount(ChatMessage(self._context_add_snapshot_payload(payload), role="system"))
            chat.scroll_end(animate=False)
            self._refresh_inspector()
            return

        # Commands
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
        if text == "/auth":
            self._handle_auth()
            return
        if text == "/tabminion":
            self._handle_tabminion()
            return

        text, pinned_messages = self._consume_at_context_refs(text)

        chat = self.query_one("#chat-log")
        if pinned_messages:
            chat.mount(ChatMessage("\n".join(pinned_messages), role="system"))

        if not text:
            chat.scroll_end(animate=False)
            self._refresh_inspector()
            return

        chat.mount(ChatMessage(text, role="user"))
        chat.scroll_end(animate=False)

        # Debug
        debug = self.query_one("#debug-log")
        debug.mount(ChatMessage(f">>> USER:\n{text}", role="system"))
        debug.scroll_end(animate=False)

        self.kernel.push_user_input(text)
        self._refresh_inspector()

        # Thinking indicator
        thinking = ChatMessage("⏳ thinking…", role="system")
        thinking.id = "thinking-indicator"
        chat.mount(thinking)
        chat.scroll_end(animate=False)

        self._run_agent()

    def _consume_at_context_refs(self, text: str) -> tuple[str, list[str]]:
        """Handle @mentions that pin live context references."""
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
                messages.append(
                    f"⚠ `@{token}` matched multiple files. Be specific: {shown}"
                )
                return match.group(0)

            if resolved is None:
                return match.group(0)

            try:
                item = self.agent.context_add_file(str(resolved), source=f"@{token}")
                messages.append(
                    f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**"
                )
                return suffix
            except Exception as exc:
                messages.append(f"⚠ failed to pin `@{token}`: {exc}")
                return match.group(0)

        # Only treat @tokens that start at whitespace/start boundaries.
        updated = re.sub(r"(?<!\S)@([^\s]+)", _replace, text)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        return updated, messages

    def _resolve_at_file_token(self, token: str) -> Path | list[Path] | None:
        """Resolve @token to a file path in/under cwd when possible."""
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

        # Basename find in cwd tree (bounded/excluded for speed).
        if "/" not in raw and "\\" not in raw:
            matches = self._find_cwd_file_candidates(raw, limit=8)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return matches

        return None

    def _find_cwd_file_candidates(self, needle: str, *, limit: int = 8) -> list[Path]:
        """Small bounded search for filename matches under cwd."""
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

    def _snapshot_context_dir(self) -> Path:
        d = Path.cwd() / ".aiipython_context"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _snapshot_file_to_context(self, path: Path, *, source: str) -> dict[str, Any]:
        """Store a non-live snapshot of a file into pinned context."""
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
        """Pin pasted payload as static content (not live file reference)."""
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
        """Best-effort clipboard text retrieval across Linux/macOS."""
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
        """Pin clipboard payload as context (image, path, or text)."""
        # 1) Try image/files via Pillow clipboard bridge.
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

        # 2) Fallback to text clipboard.
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

    def _handle_context(self, arg: str) -> None:
        """Manage pinned human context entries."""
        chat = self.query_one("#chat-log")
        cmd = arg.strip()
        parts = cmd.split(maxsplit=2) if cmd else []
        verb = (parts[0].lower() if parts else "list")

        try:
            if verb in {"", "open", "list"}:
                items = self.agent.context_list()
                body = summarize_items(items)
                chat.mount(ChatMessage(
                    "**Pinned context**\n\n"
                    f"{body}\n\n"
                    "Use: `/context add <path>`, `/context add-text <text>`, "
                    "`/context add-clip` (snapshot), `/context rm <id>`, `/context clear`\n"
                    "Shortcuts: `@path/to/file` (live ref), `@clip` (snapshot), or `@ ` + paste.",
                    role="system",
                ))
            elif verb == "add" and len(parts) >= 2:
                path_str = cmd[len(parts[0]):].strip()
                item = self.agent.context_add_file(path_str, source="/context add")
                chat.mount(ChatMessage(
                    f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**",
                    role="system",
                ))
            elif verb in {"add-text", "text"} and len(parts) >= 2:
                text_value = cmd[len(parts[0]):].strip()
                item = self.agent.context_add_text(
                    text_value,
                    label="manual",
                    source="/context add-text",
                )
                chat.mount(ChatMessage(
                    f"📌 pinned `{item['id']}` [text] **{item['label']}**",
                    role="system",
                ))
            elif verb in {"add-clip", "clip", "paste"}:
                chat.mount(ChatMessage(self._context_add_from_clipboard(snapshot=True), role="system"))
            elif verb in {"rm", "remove", "del", "delete"} and len(parts) >= 2:
                item_id = parts[1].strip()
                ok = self.agent.context_remove(item_id)
                if ok:
                    chat.mount(ChatMessage(f"Removed `{item_id}` from pinned context.", role="system"))
                else:
                    chat.mount(ChatMessage(f"No pinned context item `{item_id}`.", role="error"))
            elif verb == "clear":
                n = self.agent.context_clear()
                chat.mount(ChatMessage(f"Cleared {n} pinned context item(s).", role="system"))
            else:
                chat.mount(ChatMessage(
                    "Usage: `/context [open|list|add <path>|add-text <text>|add-clip|rm <id>|clear]`\n"
                    "Tip: type `@ ` then paste to pin as snapshot text/file.",
                    role="system",
                ))
        except Exception as exc:
            chat.mount(ChatMessage(f"/context error: {exc}", role="error"))

        chat.scroll_end(animate=False)
        self._refresh_inspector()

    @work(thread=True)
    def _run_agent(self) -> None:
        try:
            self.agent.react(
                on_step=lambda step: self._call_from_thread_safe(
                    self._display_step, step),
            )
        except Exception as exc:
            self._call_from_thread_safe(self._display_error, f"Agent error: {exc}")
        finally:
            self._call_from_thread_safe(self._remove_thinking)

    def _remove_thinking(self) -> None:
        try:
            self.query_one("#thinking-indicator").remove()
        except Exception:
            pass

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

    def _display_step(self, step: ReactionStep) -> None:
        self._remove_thinking()
        chat = self.query_one("#chat-log")

        # 1) Show assistant message first
        shown_markdown = step.markdown
        if step.blocks:
            prose = _without_fenced_code(step.markdown)
            shown_markdown = prose or "Running requested python blocks…"
        chat.mount(ChatMessage(shown_markdown, role="assistant"))

        # 2) Show executable blocks and results in order
        if step.blocks:
            for idx, block in enumerate(step.blocks, start=1):
                title = f"▶ python #{idx}"
                chat.mount(CodeChunk(title, _abbrev_multiline(block.code)))

                # There is one execution entry per runnable block.
                entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                success = bool(entry.get("success", True))
                chat.mount(ExecOutput(self._format_exec_output(entry), success=success))

        debug = self.query_one("#debug-log")
        debug.mount(ChatMessage(
            f">>> ASSISTANT (raw, step {step.iteration + 1}):\n{step.markdown}",
            role="system",
        ))

        chat.scroll_end(animate=False)
        debug.scroll_end(animate=False)
        self._refresh_inspector()

    def _display_error(self, msg: str) -> None:
        self._remove_thinking()
        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage(msg, role="error"))
        chat.scroll_end(animate=False)
        self._refresh_inspector()

    # ── /model handler ──────────────────────────────────────────────
    def _model_menu_options(self, force_refresh: bool = False) -> tuple[str, list[tuple[str, str]]]:
        """Build model choices for the interactive /model menu.

        Includes live provider discovery (auth-aware) + TabMinion services,
        cached briefly so repeated menu opens stay snappy.
        """
        now = time.monotonic()
        if (
            not force_refresh
            and self._model_menu_cache_options
            and now < self._model_menu_cache_until
        ):
            return self._model_menu_cache_title, self._model_menu_cache_options

        from aiipython.model_catalog import discover_provider_catalog
        from aiipython.settings import get_settings
        from aiipython.tabminion import is_running as _tm_running, discover_services

        options: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(value: str, label: str) -> None:
            if value not in seen:
                seen.add(value)
                options.append((value, label))

        # Current model first
        add(self.session.model, "current")

        # Live provider discovery based on currently available auth.
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

        # Recent models from persisted settings (with light filtering).
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
            # Keep only provider/model-ish recent values.
            if not re.match(r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9._:-]+$", value):
                continue
            if block_openai_recents and value.startswith("openai/"):
                continue
            add(value, "recent")

        # Curated fallback defaults (when live discovery is unavailable).
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
                # ChatGPT/Codex OAuth may not expose standard OpenAI model listings.
                # Avoid suggesting a likely-broken fallback model here.
                continue

            if info and info.source:
                add(fallback_model, f"{provider_name[provider]} fallback · unverified")
            else:
                add(fallback_model, f"{provider_name[provider]} fallback · requires auth")

        # Pi/Codex OAuth models (subscription transport).
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

        # TabMinion browser-backed models
        tab_count = 0
        if _tm_running():
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
        """Map shorthand aliases to discovered concrete models when possible."""
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
        chat = self.query_one("#chat-log")
        cmd = model_str.strip()

        if not cmd or cmd.lower() in {"refresh", "--refresh", "rescan", "verify", "check"}:
            force = cmd.lower() in {"refresh", "--refresh", "rescan", "verify", "check"}
            title, options = self._model_menu_options(force_refresh=force)
            if force:
                chat.mount(ChatMessage(f"Refreshed live model catalog.\n\n{title}", role="system"))
            self._open_menu(
                title=title,
                options=options,
                on_select=lambda value: self._handle_model(value),
            )
            return

        model_str = self._normalize_model_alias(cmd)

        # Guardrail against accidental non-model text becoming current model.
        if not re.match(r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9._:-]+$", model_str):
            chat.mount(ChatMessage(
                "Invalid model format. Use `provider/model` (e.g. `openai/gpt-4o-mini`) "
                "or run `/model` to pick from discovered options.",
                role="error",
            ))
            chat.scroll_end(animate=False)
            return

        try:
            self.session.switch_model(model_str)
            self._model_menu_cache_until = 0.0
            chat.mount(ChatMessage(f"Switched to **{model_str}**", role="system"))
            self._update_status()
        except Exception as exc:
            chat.mount(ChatMessage(f"Model switch failed: {exc}", role="error"))
        chat.scroll_end(animate=False)

    # ── /image handler ──────────────────────────────────────────────
    def _handle_image(self, path_str: str) -> None:
        chat = self.query_one("#chat-log")
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            chat.mount(ChatMessage(f"File not found: {path}", role="error"))
            return
        try:
            from PIL import Image
            img = Image.open(path)
            name = path.stem
            self.kernel.push_image(name, img)
            chat.mount(ChatMessage(
                f"Loaded **{name}** ({img.size[0]}×{img.size[1]}, {img.mode}) "
                f"→ `images['{name}']`",
                role="system",
            ))
        except Exception as exc:
            chat.mount(ChatMessage(f"Image load error: {exc}", role="error"))
        chat.scroll_end(animate=False)
        self._refresh_inspector()

    # ── /login handler ──────────────────────────────────────────────
    def _handle_login(self, provider_arg: str) -> None:
        chat = self.query_one("#chat-log")
        if not provider_arg:
            self._open_menu(
                title="Select login provider",
                options=[
                    ("anthropic", "Claude Pro/Max OAuth subscription"),
                    ("openai", "ChatGPT Plus/Pro OAuth (Codex)"),
                    ("gemini", "API key saved to auth.json"),
                ],
                on_select=lambda value: self._handle_login(value),
            )
            return

        parts = provider_arg.split(maxsplit=1)
        provider = parts[0].strip().lower()
        secret = parts[1].strip() if len(parts) > 1 else ""

        if provider == "anthropic":
            chat.mount(ChatMessage("Starting Anthropic OAuth login…", role="system"))
            chat.scroll_end(animate=False)
            self._run_login("anthropic")
            return

        if provider in {"openai", "openai-codex"}:
            # `/login openai` => OAuth; `/login openai <key>` => API key
            if not secret:
                chat.mount(ChatMessage("Starting OpenAI Codex OAuth login…", role="system"))
                chat.scroll_end(animate=False)
                self._run_login("openai-codex")
                return

            auth = get_auth_manager()
            auth.set_api_key("openai", secret)
            self._model_menu_cache_until = 0.0

            active_provider = self.session.model.split("/", 1)[0] == "openai"
            if active_provider:
                self.session.auth_source = self.session._resolve_auth()
                self._update_status()

            chat.mount(ChatMessage(
                "✅ Stored **openai** API key in `auth.json` for reuse across sessions.",
                role="system",
            ))
            chat.scroll_end(animate=False)
            return

        if provider in {"gemini", "google"}:
            normalized = "gemini" if provider == "google" else provider
            if not secret:
                chat.mount(ChatMessage(
                    "Usage: `/login <provider> <api_key>`\n"
                    "Examples:\n"
                    "- `/login openai` (OAuth)\n"
                    "- `/login openai sk-...` (API key)\n"
                    "- `/login gemini AIza...`\n\n"
                    "The key is stored in `~/.aiipython/auth.json` (0600 permissions) "
                    "and reused across sessions.",
                    role="system",
                ))
                chat.scroll_end(animate=False)
                return

            auth = get_auth_manager()
            auth.set_api_key(normalized, secret)
            self._model_menu_cache_until = 0.0

            active_provider = self.session.model.split("/", 1)[0] == normalized
            if active_provider:
                self.session.auth_source = self.session._resolve_auth()
                self._update_status()

            chat.mount(ChatMessage(
                f"✅ Stored **{normalized}** API key in `auth.json` for reuse across sessions.",
                role="system",
            ))
            chat.scroll_end(animate=False)
            return

        chat.mount(ChatMessage(
            "Unknown provider. Supported: `anthropic`, `openai`, `gemini`.",
            role="error",
        ))
        chat.scroll_end(animate=False)

    @work(thread=True)
    def _run_login(self, provider: str) -> None:
        """Run the OAuth login flow in a worker thread."""
        import webbrowser

        auth = get_auth_manager()

        def on_url(url: str) -> None:
            self._call_from_thread_safe(self._login_show_url, url)
            webbrowser.open(url)

        def on_prompt(message: str) -> str:
            # We need to get input from the user through the TUI.
            # Signal the main thread to show a prompt, then block.
            import threading
            event = threading.Event()
            result: list[str] = []

            def _ask() -> None:
                self._login_prompt_event = event
                self._login_prompt_result = result
                chat = self.query_one("#chat-log")
                chat.mount(ChatMessage(
                    f"**{message}**\n\n"
                    "Paste the code below and press Enter:",
                    role="system",
                ))
                chat.scroll_end(animate=False)
                inp = self.query_one("#user-input", Input)
                inp.placeholder = "Paste authorization code here…"
                self._login_waiting_for_code = True

            self._call_from_thread_safe(_ask)
            event.wait(timeout=300)  # 5 min timeout
            if not result:
                raise RuntimeError("Login timed out or was cancelled")
            return result[0]

        def on_status(msg: str) -> None:
            self._call_from_thread_safe(
                lambda: self.query_one("#chat-log").mount(
                    ChatMessage(msg, role="system")
                )
            )

        try:
            if provider == "anthropic":
                auth.login_anthropic(on_url, on_prompt, on_status)
            elif provider == "openai-codex":
                auth.login_openai(on_url, on_prompt, on_status)
            else:
                raise RuntimeError(f"Unsupported OAuth provider: {provider}")

            # Re-resolve auth for current model
            self.session.auth_source = self.session._resolve_auth()
            self._call_from_thread_safe(self._login_success, provider)
        except Exception as exc:
            self._call_from_thread_safe(self._login_error, str(exc))

    def _login_show_url(self, url: str) -> None:
        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage(
            f"🔗 **Open this URL to authorize:**\n\n{url}\n\n"
            "_(attempting to open in browser…)_",
            role="system",
        ))
        chat.scroll_end(animate=False)

    def _login_success(self, provider: str) -> None:
        self._login_waiting_for_code = False
        inp = self.query_one("#user-input", Input)
        inp.placeholder = "Type a message… (@file, @clip, @ + paste, /model, /context)"

        chat = self.query_one("#chat-log")
        src = self.session.auth_source
        provider_label = "openai" if provider == "openai-codex" else provider
        if src:
            chat.mount(ChatMessage(
                f"✅ **Logged in to {provider_label}**: {src.source}\n\n"
                f"This subscription token will be used instead of any API key "
                f"in your environment. Use `/auth` to verify, `/logout {provider_label}` "
                f"to revert.",
                role="system",
            ))
        else:
            chat.mount(ChatMessage(
                f"✅ Login completed for {provider}",
                role="system",
            ))
        chat.scroll_end(animate=False)
        self._model_menu_cache_until = 0.0
        self._update_status()

    def _login_error(self, msg: str) -> None:
        self._login_waiting_for_code = False
        inp = self.query_one("#user-input", Input)
        inp.placeholder = "Type a message… (@file, @clip, @ + paste, /model, /context)"

        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage(f"❌ Login failed: {msg}", role="error"))
        chat.scroll_end(animate=False)

    # ── /logout handler ─────────────────────────────────────────────
    def _handle_logout(self, provider: str) -> None:
        chat = self.query_one("#chat-log")
        provider = provider.strip().lower()
        if provider == "google":
            provider = "gemini"
        if provider == "openai-codex":
            provider = "openai"

        if not provider:
            chat.mount(ChatMessage(
                "Usage: `/logout <provider>`\n"
                "Examples: `/logout anthropic`, `/logout openai`, `/logout gemini`",
                role="system",
            ))
            chat.scroll_end(animate=False)
            return

        auth = get_auth_manager()
        # Trigger lazy import from Codex/Pi auth files before logout checks.
        auth.resolve_api_key(provider)
        if provider == "openai":
            auth.resolve_api_key("openai-codex")

        provider_keys = [provider]
        if provider == "openai":
            provider_keys = ["openai-codex", "openai"]

        existing = [(k, auth.get(k) or {}) for k in provider_keys if auth.get(k)]
        if not existing:
            chat.mount(ChatMessage(
                f"No stored credentials for `{provider}`.", role="system",
            ))
            chat.scroll_end(animate=False)
            return

        # If this process currently has the same key injected from auth.json,
        # remove it before resolving fallback so logout actually takes effect.
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

        # Re-resolve — will fall back to env var if available
        self.session.auth_source = self.session._resolve_auth()
        src = self.session.auth_source

        if src:
            chat.mount(ChatMessage(
                f"✅ Logged out of **{provider}**. "
                f"Now using: **{src.source}** "
                f"({'subscription' if src.is_subscription else '⚠️ billed per call'})",
                role="system",
            ))
        else:
            if provider == "openai":
                hint = "`/login openai` (OAuth) or `/login openai <api_key>`"
            else:
                hint = f"`/login {provider} <api_key>`"
            chat.mount(ChatMessage(
                f"✅ Logged out of **{provider}**. No other credentials found — "
                f"set `{provider.upper()}_API_KEY` or {hint}.",
                role="system",
            ))
        chat.scroll_end(animate=False)
        self._model_menu_cache_until = 0.0
        self._update_status()

    # ── /auth handler ───────────────────────────────────────────────
    def _handle_auth(self) -> None:
        chat = self.query_one("#chat-log")
        auth = get_auth_manager()
        summary = auth.auth_summary()

        if not summary:
            chat.mount(ChatMessage(
                "**No auth configured.**\n\n"
                "Use `/login anthropic` or `/login openai` for subscription OAuth, "
                "store keys with `/login <provider> <api_key>`, "
                "or set environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).",
                role="system",
            ))
            chat.scroll_end(animate=False)
            return

        lines = ["**Auth sources** (highest priority first):\n"]
        lines.append("| Provider | Source | Subscription | Key Preview |")
        lines.append("|----------|--------|:------------:|-------------|")
        for entry in summary:
            sub = "✅ free" if entry["subscription"] == "yes" else "💰 billed"
            lines.append(
                f"| {entry['provider']} | {entry['source']} "
                f"| {sub} | `{entry['key_preview']}` |"
            )

        # Show which is active
        src = self.session.auth_source
        if src:
            lines.append("")
            if src.is_subscription:
                lines.append(
                    f"**Active for {self.session.model}**: "
                    f"{src.source} — *subscription, no per-call cost*"
                )
            else:
                lines.append(
                    f"**⚠️ Active for {self.session.model}**: "
                    f"{src.source} — *billed per API call*"
                )

        lines.append("")
        lines.append(
            "*Priority: OAuth subscription > auth.json key > env var. "
            "Use `/login anthropic` or `/login openai` for OAuth, "
            "or `/login <provider> <api_key>` to persist keys.*"
        )

        chat.mount(ChatMessage("\n".join(lines), role="system"))
        chat.scroll_end(animate=False)

    # ── /tabminion handler ──────────────────────────────────────────
    def _handle_tabminion(self) -> None:
        chat = self.query_one("#chat-log")
        from aiipython.tabminion import status_summary
        chat.mount(ChatMessage(status_summary(), role="system"))
        chat.scroll_end(animate=False)


# Backward compatibility
PyCodeApp = AiiPythonApp
