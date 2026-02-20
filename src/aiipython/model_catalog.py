"""Model catalog discovery for /model menu.

Builds provider-aware model options based on configured auth and live services.
Designed to be fast, best-effort, and safe to call frequently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import httpx

from aiipython.auth import get_auth_manager


@dataclass
class ProviderCatalog:
    provider: str
    source: str | None
    models: list[str]
    error: str | None = None


def _sort_by_priority(values: list[str], keys: list[str]) -> list[str]:
    def score(v: str) -> tuple[int, str]:
        lower = v.lower()
        for i, k in enumerate(keys):
            if k in lower:
                return i, lower
        return len(keys), lower

    return sorted(values, key=score)


def _discover_openai_models(api_key: str, timeout: float, limit: int) -> list[str]:
    resp = httpx.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json().get("data", [])
    values: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            continue
        lower = model_id.lower()
        # Keep chat/codex/reasoning families; drop obvious non-chat endpoints.
        if any(x in lower for x in ("gpt", "o1", "o3", "o4", "codex")) and not any(
            x in lower for x in ("whisper", "tts", "embedding", "moderation", "transcribe")
        ):
            values.append(model_id)

    values = list(dict.fromkeys(values))
    values = _sort_by_priority(values, ["codex", "gpt-5", "gpt-4.1", "o3", "o1", "mini"])
    return values[:limit]


def _discover_anthropic_models(api_key: str, timeout: float, limit: int) -> list[str]:
    resp = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json().get("data", [])
    values: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str) and "claude" in model_id.lower():
            values.append(model_id)

    values = list(dict.fromkeys(values))
    values = _sort_by_priority(values, ["opus", "sonnet", "haiku"])
    return values[:limit]


def _discover_gemini_models(api_key: str, timeout: float, limit: int) -> list[str]:
    resp = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
        timeout=timeout,
    )
    resp.raise_for_status()

    items = resp.json().get("models", [])
    values: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if isinstance(methods, list) and "generateContent" not in methods:
            continue

        name = item.get("name")
        if not isinstance(name, str) or not name.startswith("models/"):
            continue
        model_id = name.split("/", 1)[1]
        if "gemini" in model_id.lower():
            values.append(model_id)

    values = list(dict.fromkeys(values))
    values = _sort_by_priority(values, ["2.5-pro", "2.5-flash", "2.0-flash", "1.5-pro", "1.5-flash"])
    return values[:limit]


def discover_provider_catalog(timeout: float = 2.0, per_provider_limit: int = 10) -> dict[str, ProviderCatalog]:
    """Discover available models for configured providers.

    Best effort: network/auth errors are captured in ``error`` and do not raise.
    """
    auth = get_auth_manager()
    results: dict[str, ProviderCatalog] = {}

    discoverers: dict[str, Callable[[str, float, int], list[str]]] = {
        "anthropic": _discover_anthropic_models,
        "openai": _discover_openai_models,
        "gemini": _discover_gemini_models,
    }

    for provider, discover in discoverers.items():
        source = auth.resolve_api_key(provider)
        if not source:
            results[provider] = ProviderCatalog(
                provider=provider,
                source=None,
                models=[],
                error="no-auth",
            )
            continue

        try:
            models = discover(source.key, timeout=timeout, limit=per_provider_limit)
            results[provider] = ProviderCatalog(
                provider=provider,
                source=source.source,
                models=models,
                error=None,
            )
        except Exception as exc:  # network/auth/shape errors
            results[provider] = ProviderCatalog(
                provider=provider,
                source=source.source,
                models=[],
                error=str(exc),
            )

    return results
