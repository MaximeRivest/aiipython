"""Bridge between WireLog and the standalone traffic inspector server.

Pushes every WireLog event directly to :mod:`traffic_server` (in-process,
no external dependencies).  The inspector server starts automatically.

Usage — call once at startup::

    from aiipython.wire_bridge import attach_traffic_bridge
    attach_traffic_bridge(session.wire_log)

The viewer is then available at ``http://127.0.0.1:<port>``.
Call :func:`get_inspector_url` to retrieve the URL.
"""

from __future__ import annotations

import logging
from typing import Any

from aiipython.wire import WireEntry, WireLog

log = logging.getLogger(__name__)

_BRIDGE_ATTACHED = False
_inspector_port: int = 0


def get_inspector_url() -> str | None:
    """Return the inspector URL if it's running, else None."""
    if _inspector_port:
        return f"http://127.0.0.1:{_inspector_port}"
    return None


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Make messages JSON-safe for transmission (trim huge base64 images)."""
    out = []
    for msg in messages or []:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    parts.append({"type": "text", "text": "[image]"})
                elif isinstance(block, str) and len(block) > 50_000:
                    parts.append({"type": "text", "text": block[:50_000] + "...[truncated]"})
                else:
                    parts.append(block)
            out.append({"role": role, "content": parts})
        elif isinstance(content, str) and len(content) > 100_000:
            out.append({"role": role, "content": content[:100_000] + "...[truncated]"})
        else:
            out.append({"role": role, "content": content})
    return out


def attach_traffic_bridge(wire_log: WireLog) -> int:
    """Hook into *wire_log* and start the traffic inspector server.

    Returns the server port.  Safe to call multiple times — only attaches once.
    """
    global _BRIDGE_ATTACHED, _inspector_port
    if _BRIDGE_ATTACHED:
        return _inspector_port

    from aiipython import traffic_server

    port = traffic_server.start()
    _inspector_port = port
    _BRIDGE_ATTACHED = True

    # Save any pre-existing callbacks so we can chain
    prev_on_request = wire_log.on_request
    prev_on_chunk = wire_log.on_chunk
    prev_on_done = wire_log.on_done

    def on_request(entry: WireEntry) -> None:
        if prev_on_request:
            prev_on_request(entry)
        try:
            traffic_server.ingest_request(
                entry_id=entry.entry_id,
                model=entry.model,
                messages=_sanitize_messages(entry.messages),
                kwargs=entry.kwargs,
            )
        except Exception:
            log.debug("traffic bridge: ingest_request failed", exc_info=True)

    def on_chunk(entry: WireEntry, chunk: str) -> None:
        if prev_on_chunk:
            prev_on_chunk(entry, chunk)
        try:
            traffic_server.ingest_chunk(entry.entry_id, chunk)
        except Exception:
            pass

    def on_done(entry: WireEntry) -> None:
        if prev_on_done:
            prev_on_done(entry)
        try:
            traffic_server.ingest_done(
                entry_id=entry.entry_id,
                full_response=entry.full_response,
                duration_ms=entry.duration_ms,
                usage=entry.usage,
                error=entry.error,
            )
        except Exception:
            log.debug("traffic bridge: ingest_done failed", exc_info=True)

    wire_log.on_request = on_request
    wire_log.on_chunk = on_chunk
    wire_log.on_done = on_done

    log.debug("Traffic inspector bridge attached (port %d)", port)
    return port
