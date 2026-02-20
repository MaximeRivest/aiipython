"""TabMinion integration — use browser AI subscriptions as LM backends.

TabMinion (https://github.com/…/tabminion) exposes an OpenAI-compatible
API at ``http://localhost:8080/v1``.  Any ChatGPT, Claude, Grok, or
Gemini tab open in Firefox/Zen becomes a free LM backend — no API keys,
no per-call cost.

This module provides:
  - Auto-detection of a running TabMinion instance
  - Model discovery (which browser tabs are available)
  - LM configuration so litellm routes through the browser

Usage in aiipython::

    /tabminion           # show status + available tabs
    /model tabminion/claude   # switch to Claude via browser
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"
OPENAI_BASE = f"{BASE_URL}/v1"

# ── health / discovery ───────────────────────────────────────────────

def is_running(timeout: float = 2.0) -> bool:
    """Check if TabMinion is reachable."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/health", timeout=timeout)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def get_status(timeout: float = 3.0) -> dict | None:
    """Return TabMinion health status, or None if unreachable."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/health", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return None


def list_tabs(timeout: float = 5.0) -> list[dict]:
    """List all browser tabs known to TabMinion."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/tabs", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        return []
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return []


def list_models(timeout: float = 5.0) -> list[dict]:
    """List available AI models via the OpenAI-compatible endpoint."""
    try:
        resp = httpx.get(f"{OPENAI_BASE}/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        return []
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return []


# Service metadata for display
SERVICE_INFO = {
    "chatgpt": {"name": "ChatGPT", "emoji": "🤖", "urls": ["chatgpt.com"]},
    "claude":  {"name": "Claude",  "emoji": "🟣", "urls": ["claude.ai"]},
    "grok":    {"name": "Grok",    "emoji": "⚡", "urls": ["grok.com"]},
    "gemini":  {"name": "Gemini",  "emoji": "💎", "urls": ["gemini.google.com"]},
}


def discover_services(timeout: float = 5.0) -> list[dict[str, str]]:
    """Discover which AI services have open browser tabs.

    Returns a list of dicts with keys: id, name, emoji, tab_url.
    """
    tabs = list_tabs(timeout)
    found = []
    seen = set()

    for svc_id, info in SERVICE_INFO.items():
        for tab in tabs:
            url = tab.get("url", "")
            if any(pattern in url for pattern in info["urls"]):
                if svc_id not in seen:
                    seen.add(svc_id)
                    found.append({
                        "id": svc_id,
                        "name": info["name"],
                        "emoji": info["emoji"],
                        "tab_url": url,
                        "tab_id": tab.get("id"),
                    })
                break

    return found


# ── model string mapping ─────────────────────────────────────────────

def is_tabminion_model(model_str: str) -> bool:
    """Check if a model string is a tabminion model."""
    return model_str.startswith("tabminion/")


def parse_tabminion_model(model_str: str) -> str:
    """Extract the service name from ``tabminion/claude`` → ``claude``."""
    return model_str.split("/", 1)[1] if "/" in model_str else model_str


def litellm_model_kwargs(model_str: str) -> dict[str, Any]:
    """Return kwargs for litellm/dspy to route through TabMinion.

    ``tabminion/claude`` becomes ``openai/claude`` routed to localhost:8080.
    """
    service = parse_tabminion_model(model_str)
    return {
        "model": f"openai/{service}",
        "api_base": OPENAI_BASE,
        "api_key": "not-needed",
    }


# ── summary for display ─────────────────────────────────────────────

def status_summary() -> str:
    """Markdown summary for the ``/tabminion`` command."""
    status = get_status()
    if not status:
        return (
            "**TabMinion** is not running.\n\n"
            "Start TabMinion and load the browser extension to use "
            "browser AI subscriptions as free LM backends.\n\n"
            "See: `~/Projects/tabminion/README.md`"
        )

    services = discover_services()
    lines = [
        f"**TabMinion** is running ✅\n",
        f"API: `{BASE_URL}`\n",
    ]

    if services:
        lines.append("**Available models:**\n")
        lines.append("| Model | Service | Tab |")
        lines.append("|-------|---------|-----|")
        for svc in services:
            model_str = f"`tabminion/{svc['id']}`"
            lines.append(
                f"| {model_str} | {svc['emoji']} {svc['name']} "
                f"| {svc['tab_url'][:50]}… |"
            )
        lines.append("")
        lines.append("Use `/model tabminion/claude` (or chatgpt, grok, gemini) to switch.")
    else:
        lines.append(
            "\n⚠️ No AI tabs detected. Open ChatGPT, Claude, Grok, "
            "or Gemini in Firefox/Zen."
        )

    lines.append("\n*Browser subscriptions — no API cost.*")
    return "\n".join(lines)
