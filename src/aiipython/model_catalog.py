"""Model catalog discovery for /model menu (pi-native only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderCatalog:
    provider: str
    source: str | None
    models: list[str]
    error: str | None = None


def discover_provider_catalog(timeout: float = 2.0, per_provider_limit: int = 10) -> dict[str, ProviderCatalog]:
    """Discover available models via pi-native gateway (best effort)."""
    try:
        from aiipython.pi_native_client import PiNativeClient
        client = PiNativeClient()
        models = client.list_models()
        auth = client.auth_status()
    except Exception as exc:
        return {
            "anthropic": ProviderCatalog("anthropic", None, [], str(exc)),
            "openai": ProviderCatalog("openai", None, [], str(exc)),
            "gemini": ProviderCatalog("gemini", None, [], str(exc)),
        }

    source_by_provider: dict[str, str | None] = {}
    for row in auth.get("providers", []):
        p = row.get("provider")
        s = row.get("source")
        if isinstance(p, str):
            source_by_provider[p] = s if isinstance(s, str) else None

    grouped: dict[str, list[str]] = {"anthropic": [], "openai": [], "gemini": []}
    for m in models:
        provider = m.get("provider")
        model_id = m.get("id")
        available = bool(m.get("available", False))
        if provider in grouped and isinstance(model_id, str) and available:
            grouped[provider].append(model_id)

    out: dict[str, ProviderCatalog] = {}
    for provider in ("anthropic", "openai", "gemini"):
        uniq = list(dict.fromkeys(grouped.get(provider, [])))[:per_provider_limit]
        src = source_by_provider.get(provider)
        out[provider] = ProviderCatalog(
            provider=provider,
            source=src,
            models=uniq,
            error=None if uniq else ("no-auth" if not src else "no-models"),
        )
    return out
