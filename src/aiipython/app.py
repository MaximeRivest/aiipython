"""Textual TUI — pi-inspired chat interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

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
from aiipython.session import Session
from aiipython.wire import WireEntry, format_wire_messages


# ── Helpers ─────────────────────────────────────────────────────────

def _fmt_tokens(n: int) -> str:
    """Format token count like pi: 1234 → 1.2k, 12345 → 12k."""
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    return f"{n / 1_000_000:.1f}M"


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
                placeholder="Type a message… (/model, /image, /login, /auth)",
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
            "`/model` `/image` `/login` `/logout` `/auth` `/tabminion`",
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
            "Type a message… (/model, /image, /login, /auth)"
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
        text = event.value.strip()
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
        if text == "/auth":
            self._handle_auth()
            return
        if text == "/tabminion":
            self._handle_tabminion()
            return

        chat = self.query_one("#chat-log")
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

    @work(thread=True)
    def _run_agent(self) -> None:
        try:
            self.agent.react(
                on_response=lambda md: self._call_from_thread_safe(
                    self._display_response, md),
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

    def _display_response(self, markdown: str) -> None:
        self._remove_thinking()
        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage(markdown, role="assistant"))

        debug = self.query_one("#debug-log")
        debug.mount(ChatMessage(f">>> ASSISTANT (raw):\n{markdown}", role="system"))

        # Show execution output in pi-style colored boxes
        for entry in self.kernel.history:
            output_parts: list[str] = []
            if entry.get("stdout"):
                output_parts.append(entry["stdout"].rstrip())
            if entry.get("stderr"):
                output_parts.append(f"stderr: {entry['stderr'].rstrip()}")
            if entry.get("result"):
                output_parts.append(f"→ {entry['result']}")
            if entry.get("error"):
                output_parts.append(f"✗ {entry['error']}")
            if output_parts and entry.get("_shown") is not True:
                success = entry.get("success", True)
                chat.mount(ExecOutput(
                    "\n".join(output_parts),
                    success=success,
                ))
                entry["_shown"] = True

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
    def _model_menu_options(self) -> list[tuple[str, str]]:
        """Build model choices for the interactive /model menu."""
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

        # Recent models from persisted settings
        for m in get_settings().get_recent_models():
            add(m, "recent")

        # Curated defaults
        add("anthropic/claude-sonnet-4-20250514", "Anthropic")
        add("openai/gpt-4o-mini", "OpenAI")
        add("gemini/gemini-2.0-flash", "Gemini")

        # TabMinion browser-backed models
        if _tm_running():
            for svc in discover_services():
                add(f"tabminion/{svc['id']}", f"TabMinion {svc['emoji']} {svc['name']}")

        return options

    def _handle_model(self, model_str: str) -> None:
        chat = self.query_one("#chat-log")
        if not model_str:
            self._open_menu(
                title=f"Select model (current: {self.session.model})",
                options=self._model_menu_options(),
                on_select=lambda value: self._handle_model(value),
            )
            return

        try:
            self.session.switch_model(model_str)
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
                    ("openai", "API key saved to auth.json"),
                    ("gemini", "API key saved to auth.json"),
                ],
                on_select=lambda value: self._handle_login(value),
            )
            return

        parts = provider_arg.split(maxsplit=1)
        provider = parts[0].strip().lower()
        secret = parts[1].strip() if len(parts) > 1 else ""

        if provider == "anthropic":
            # Start the OAuth login flow in the TUI
            chat.mount(ChatMessage("Starting Anthropic OAuth login…", role="system"))
            chat.scroll_end(animate=False)
            self._run_login(provider)
            return

        if provider in {"openai", "gemini", "google"}:
            normalized = "gemini" if provider == "google" else provider
            if not secret:
                chat.mount(ChatMessage(
                    "Usage: `/login <provider> <api_key>`\n"
                    "Examples:\n"
                    "- `/login openai sk-...`\n"
                    "- `/login gemini AIza...`\n\n"
                    "The key is stored in `~/.aiipython/auth.json` (0600 permissions) "
                    "and reused across sessions.",
                    role="system",
                ))
                chat.scroll_end(animate=False)
                return

            auth = get_auth_manager()
            auth.set_api_key(normalized, secret)

            # If this provider is currently active, refresh session auth immediately.
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
        auth_url_shown = False
        prompt_future: list = []  # poor man's channel

        def on_url(url: str) -> None:
            nonlocal auth_url_shown
            auth_url_shown = True
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
            auth.login_anthropic(on_url, on_prompt, on_status)
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
        inp.placeholder = "Type a message… (/model, /image, /login, /auth)"

        chat = self.query_one("#chat-log")
        src = self.session.auth_source
        if src:
            chat.mount(ChatMessage(
                f"✅ **Logged in to {provider}**: {src.source}\n\n"
                f"This subscription token will be used instead of any API key "
                f"in your environment. Use `/auth` to verify, `/logout {provider}` "
                f"to revert.",
                role="system",
            ))
        else:
            chat.mount(ChatMessage(
                f"✅ Login completed for {provider}",
                role="system",
            ))
        chat.scroll_end(animate=False)
        self._update_status()

    def _login_error(self, msg: str) -> None:
        self._login_waiting_for_code = False
        inp = self.query_one("#user-input", Input)
        inp.placeholder = "Type a message… (/model, /image, /login, /auth)"

        chat = self.query_one("#chat-log")
        chat.mount(ChatMessage(f"❌ Login failed: {msg}", role="error"))
        chat.scroll_end(animate=False)

    # ── /logout handler ─────────────────────────────────────────────
    def _handle_logout(self, provider: str) -> None:
        chat = self.query_one("#chat-log")
        provider = provider.strip().lower()
        if provider == "google":
            provider = "gemini"

        if not provider:
            chat.mount(ChatMessage(
                "Usage: `/logout <provider>`\n"
                "Examples: `/logout anthropic`, `/logout openai`, `/logout gemini`",
                role="system",
            ))
            chat.scroll_end(animate=False)
            return

        auth = get_auth_manager()
        if not auth.has(provider):
            chat.mount(ChatMessage(
                f"No stored credentials for `{provider}`.", role="system",
            ))
            chat.scroll_end(animate=False)
            return

        # If this process currently has the same key injected from auth.json,
        # remove it before resolving fallback so logout actually takes effect.
        old_cred = auth.get(provider) or {}
        removed_key = ""
        if old_cred.get("type") == "oauth":
            removed_key = old_cred.get("access", "")
        elif old_cred.get("type") == "api_key":
            removed_key = old_cred.get("key", "")

        env_var = ENV_KEY_MAP.get(provider)
        if env_var and removed_key and os.environ.get(env_var) == removed_key:
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
            chat.mount(ChatMessage(
                f"✅ Logged out of **{provider}**. No other credentials found — "
                f"set `{provider.upper()}_API_KEY` or `/login {provider} <api_key>`.",
                role="system",
            ))
        chat.scroll_end(animate=False)
        self._update_status()

    # ── /auth handler ───────────────────────────────────────────────
    def _handle_auth(self) -> None:
        chat = self.query_one("#chat-log")
        auth = get_auth_manager()
        summary = auth.auth_summary()

        if not summary:
            chat.mount(ChatMessage(
                "**No auth configured.**\n\n"
                "Use `/login anthropic` for subscription access, or store keys with "
                "`/login openai <api_key>` / `/login gemini <api_key>`, "
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
            "Use `/login anthropic` for OAuth or `/login <provider> <api_key>` to persist keys.*"
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
