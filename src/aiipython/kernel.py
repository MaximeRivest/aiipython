"""IPython kernel wrapper — all state lives here."""

from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from typing import Any

from IPython.core.interactiveshell import InteractiveShell

from aiipython.checkpoint import CheckpointTree


def _preview(text: str, limit: int = 80) -> str:
    """Short preview of a string for history entries."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class Kernel:
    """Wraps an IPython InteractiveShell.

    Can attach to an *existing* shell (e.g. when called from a running
    IPython session via ``chat()``).  Namespace init is idempotent — safe
    to call multiple times on the same shell.
    """

    def __init__(
        self,
        shell: InteractiveShell | None = None,
        *,
        enable_checkpoints: bool = True,
    ) -> None:
        self.shell = shell or InteractiveShell.instance()
        self.history: list[dict[str, Any]] = []
        self.checkpoints: CheckpointTree | None = (
            CheckpointTree() if enable_checkpoints else None
        )
        self._init_namespace()

    def _init_namespace(self) -> None:
        """Ensure our variables exist — idempotent, never overwrites."""
        from aiipython.magics import register as register_magics
        register_magics(self.shell)

        ns = self.shell.user_ns
        if "user_inputs" not in ns:
            ns["user_inputs"] = []
        if "user_input" not in ns:
            ns["user_input"] = None
        if "images" not in ns:
            ns["images"] = {}
        if "ai_responses" not in ns:
            ns["ai_responses"] = []
        if "agent_context" not in ns:
            ns["agent_context"] = ""

        from aiipython.context import ensure_context_namespace
        ensure_context_namespace(ns)

        ns["_kernel"] = self

    # ── execute code ────────────────────────────────────────────────
    def execute(self, code: str, tag: str | None = None) -> dict[str, Any]:
        """Run *code* in IPython and capture stdout/stderr."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = self.shell.run_cell(code, silent=False)

        entry: dict[str, Any] = {
            "code": code,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "success": result.success,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if tag:
            entry["tag"] = tag
        if result.result is not None:
            entry["result"] = repr(result.result)

        # Capture the error type and message so the agent always sees
        # *why* code failed, even when the full traceback is too long.
        if result.error_in_exec is not None:
            exc = result.error_in_exec
            entry["error"] = f"{type(exc).__name__}: {exc}"
        elif result.error_before_exec is not None:
            exc = result.error_before_exec
            entry["error"] = f"{type(exc).__name__}: {exc}"

        self.history.append(entry)
        self.shell.user_ns["terminal_history"] = self.history
        return entry

    # ── convenience helpers ─────────────────────────────────────────
    def push(self, **kwargs: Any) -> None:
        """Inject variables into the IPython namespace (no history trace)."""
        self.shell.push(kwargs)

    def push_user_input(self, text: str) -> None:
        """Store a user message — metadata only in history."""
        self.shell.user_ns["user_input"] = text
        self.shell.user_ns["user_inputs"].append(text)
        self._record(
            code=f"user_input = <message, {len(text)} chars>",
            tag="user",
            summary=_preview(text),
        )

    def push_image(self, name: str, image: Any) -> None:
        """Store an image in the namespace."""
        self.shell.user_ns["images"][name] = image
        size = getattr(image, "size", ("?", "?"))
        mode = getattr(image, "mode", "?")
        self._record(
            code=f'images["{name}"] = <Image, {size[0]}×{size[1]}, {mode}>',
            tag="user",
        )

    def push_ai_response(self, text: str) -> None:
        """Store an AI response."""
        self.shell.user_ns["ai_responses"].append(text)
        self._record(
            code=f"ai_responses += <response, {len(text)} chars>",
            tag="agent",
        )

    def _record(self, code: str, tag: str, summary: str | None = None,
                success: bool = True) -> None:
        """Append a metadata-only entry to the history log."""
        entry: dict[str, Any] = {
            "code": code, "stdout": "", "stderr": "",
            "success": success,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tag": tag,
        }
        if summary:
            entry["summary"] = summary
        self.history.append(entry)
        self.shell.user_ns["terminal_history"] = self.history

    # ── smart snapshot ──────────────────────────────────────────────
    def snapshot(self) -> dict[str, str]:
        """Return a {name: human-readable summary} dict."""
        ns = self.shell.user_ns
        skip = {
            "In", "Out", "get_ipython", "exit", "quit", "open",
            "_kernel", "_oh", "_dh", "_ih", "_ii", "_iii", "_i",
            "terminal_history", "chat",
            "agent", "spawn_agent", "look_at",
            "context_add", "context_add_file", "context_add_text",
            "context_remove", "context_clear", "context_list",
        }
        out: dict[str, str] = {}
        for k, v in sorted(ns.items()):
            if k.startswith("_") or k in skip:
                continue
            out[k] = _summarise(k, v)
        return out


# ── variable summariser ─────────────────────────────────────────────

def _summarise(name: str, v: Any) -> str:
    t = type(v).__name__

    if isinstance(v, (int, float, bool, complex)):
        return f"{t} = {v}"
    if isinstance(v, str):
        if len(v) <= 80:
            return f'str = "{v}"'
        return f'str, {len(v)} chars, starts: "{v[:60]}…"'
    if v is None:
        return "None"
    if isinstance(v, list):
        if not v:
            return "list, empty"
        return f"list, {len(v)} items, latest: {_short_repr(v[-1])}"
    if isinstance(v, dict):
        if not v:
            return "dict, empty"
        keys = list(v.keys())
        shown = ", ".join(repr(k) for k in keys[:6])
        return f"dict, {len(v)} keys: [{shown}{'…' if len(keys) > 6 else ''}]"
    if isinstance(v, (set, frozenset)):
        return f"{t}, {len(v)} items"
    if isinstance(v, tuple):
        return f"tuple = {v!r}" if len(v) <= 5 else f"tuple, {len(v)} items"

    try:
        import pandas as pd
        if isinstance(v, pd.DataFrame):
            cols = ", ".join(v.columns[:8])
            return f"DataFrame, {len(v)} rows × {len(v.columns)} cols: [{cols}{'…' if len(v.columns) > 8 else ''}]"
        if isinstance(v, pd.Series):
            return f"Series, {len(v)} rows, dtype={v.dtype}"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return f"ndarray, shape={v.shape}, dtype={v.dtype}"
    except ImportError:
        pass
    try:
        from PIL import Image
        if isinstance(v, Image.Image):
            return f"Image, {v.size[0]}×{v.size[1]}, mode={v.mode}"
    except ImportError:
        pass

    if callable(v):
        return f"{t} (callable)"
    r = repr(v)
    return f"{t} = {r}" if len(r) <= 80 else f"{t}, {_short_repr(v)}"


def _short_repr(v: Any, limit: int = 60) -> str:
    r = repr(v)
    return r if len(r) <= limit else r[:limit - 1] + "…"
