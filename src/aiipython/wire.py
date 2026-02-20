"""Wire logger — captures raw LM API requests and streaming responses.

Registers as a LiteLLM CustomLogger so it sees every call, including
the exact messages POST'd and each streaming chunk as it arrives.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import litellm
from litellm.integrations.custom_logger import CustomLogger


@dataclass
class WireEntry:
    """One LM API call — request + response."""
    ts: str
    model: str
    messages: list[dict]
    kwargs: dict
    entry_id: int = 0
    chunks: list[str] = field(default_factory=list)
    full_response: str = ""
    usage: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


class WireLog(CustomLogger):
    """LiteLLM callback that captures every API call.

    Attributes:
        entries:    completed request/response pairs
        on_chunk:   callback fired for each streaming chunk
        on_request: callback fired when a new request starts
        on_done:    callback fired when a response completes
    """

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[WireEntry] = []
        self._pending: dict[str, WireEntry] = {}  # keyed by litellm call id
        self._next_id: int = 0

        # Callbacks the TUI can hook into
        self.on_chunk: Callable[[WireEntry, str], None] | None = None
        self.on_request: Callable[[WireEntry], None] | None = None
        self.on_done: Callable[[WireEntry], None] | None = None

    def _call_id(self, kwargs: dict) -> str:
        """Consistent call-id extraction across all callbacks."""
        return str(kwargs.get("litellm_call_id", ""))

    # ── request ─────────────────────────────────────────────────────
    def log_pre_api_call(self, model, messages, kwargs):
        call_id = self._call_id(kwargs)
        self._next_id += 1
        entry = WireEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            model=model or "?",
            messages=_sanitize_messages(messages),
            kwargs=_safe_kwargs(kwargs),
            entry_id=self._next_id,
        )
        self._pending[call_id] = entry

        if self.on_request:
            self.on_request(entry)

    # ── streaming chunks ────────────────────────────────────────────
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        call_id = self._call_id(kwargs)
        entry = self._pending.get(call_id)
        if not entry:
            return

        # Extract the chunk text
        chunk_text = ""
        try:
            choices = getattr(response_obj, "choices", [])
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    chunk_text = getattr(delta, "content", "") or ""
        except Exception:
            chunk_text = str(response_obj)

        if chunk_text:
            entry.chunks.append(chunk_text)
            if self.on_chunk:
                self.on_chunk(entry, chunk_text)

    # ── success ─────────────────────────────────────────────────────
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        call_id = self._call_id(kwargs)
        entry = self._pending.pop(call_id, None)
        if not entry:
            # Wasn't tracked (e.g. cache hit) — create a synthetic entry
            self._next_id += 1
            entry = WireEntry(
                ts=datetime.now(timezone.utc).isoformat(),
                model=kwargs.get("model", "?"),
                messages=_sanitize_messages(kwargs.get("messages", [])),
                kwargs=_safe_kwargs(kwargs),
                entry_id=self._next_id,
            )

        # Extract full response
        try:
            choices = getattr(response_obj, "choices", [])
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    entry.full_response = getattr(msg, "content", "") or ""
            usage = getattr(response_obj, "usage", None)
            if usage:
                entry.usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
        except Exception:
            entry.full_response = str(response_obj)

        if start_time and end_time:
            entry.duration_ms = (end_time - start_time).total_seconds() * 1000

        self.entries.append(entry)
        if self.on_done:
            self.on_done(entry)

    # ── failure ─────────────────────────────────────────────────────
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        call_id = self._call_id(kwargs)
        entry = self._pending.pop(call_id, None)
        if not entry:
            self._next_id += 1
            entry = WireEntry(
                ts=datetime.now(timezone.utc).isoformat(),
                model=kwargs.get("model", "?"),
                messages=_sanitize_messages(kwargs.get("messages", [])),
                kwargs=_safe_kwargs(kwargs),
                entry_id=self._next_id,
            )
        entry.error = str(response_obj)
        if start_time and end_time:
            entry.duration_ms = (end_time - start_time).total_seconds() * 1000

        self.entries.append(entry)
        if self.on_done:
            self.on_done(entry)


# ── helpers ─────────────────────────────────────────────────────────

def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Make messages JSON-safe — full text preserved, base64 images summarised."""
    out = []
    for msg in (messages or []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        parts.append(f"[base64 image, {len(url):,} chars]")
                    else:
                        parts.append(f"[image: {url}]")
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": content})
    return out


def _safe_kwargs(kwargs: dict) -> dict:
    """Extract display-worthy kwargs, skipping internal litellm keys."""
    skip = {
        "messages", "api_key", "api_base", "base_url",
        "litellm_call_id", "litellm_logging_obj", "litellm_params",
        "headers", "acompletion", "complete_input_dict",
        "extra_headers", "custom_llm_provider", "metadata",
        "caching", "mock_response", "proxy_server_request",
        "preset_cache_key", "no-log",
    }
    return {
        k: v for k, v in kwargs.items()
        if k not in skip
        and not k.startswith("_")
        and not k.startswith("litellm")
        and v is not None
    }


def format_wire_messages(messages: list[dict]) -> str:
    """Render sanitised messages for the wire panel display."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_str = "\n".join(str(p) for p in content)
        else:
            content_str = str(content)
        parts.append(f"──── [{role}] ({len(content_str):,} chars) ────")
        parts.append(content_str)
        parts.append("")
    return "\n".join(parts)


# ── registration ────────────────────────────────────────────────────

_wire_log: WireLog | None = None


def get_wire_log() -> WireLog:
    """Get or create the global WireLog and register it with LiteLLM."""
    global _wire_log
    if _wire_log is None:
        _wire_log = WireLog()
        litellm.callbacks.append(_wire_log)
    return _wire_log
