"""OAuth authentication and API key management.

Supports:
  - Anthropic OAuth (Claude Pro/Max subscription)
  - API keys from ``~/.aiipython/auth.json``
  - Environment variable fallback

Priority (highest → lowest):
  1. OAuth token from ``auth.json``  (your subscription — free at point of use)
  2. API key from ``auth.json``       (explicit file-based key)
  3. Environment variable             (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

This ordering ensures subscription tokens are preferred over billed API
keys so users don't get surprised costs.

Usage::

    from aiipython.auth import get_auth_manager

    auth = get_auth_manager()
    key, source = auth.resolve_api_key("anthropic")
    # key    = "sk-ant-..."
    # source = "oauth (Claude Pro/Max)"
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

log = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────

AUTH_DIR = Path.home() / ".aiipython"
AUTH_FILE = AUTH_DIR / "auth.json"

# Provider → environment variable mapping
ENV_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


# ── data types ───────────────────────────────────────────────────────

@dataclass
class AuthSource:
    """Describes where an API key came from."""

    key: str
    source: str          # human-readable label for UI
    provider: str        # e.g. "anthropic"
    is_subscription: bool = False  # True for OAuth (free at point of use)

    @property
    def display(self) -> str:
        """Short string for the footer/status bar."""
        if self.is_subscription:
            return f"🔑 {self.source}"
        return f"🔑 {self.source}"


# ── PKCE ─────────────────────────────────────────────────────────────

def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge) for PKCE S256."""
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url(verifier_bytes)
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# ── Anthropic OAuth ──────────────────────────────────────────────────

_ANTHROPIC_CLIENT_ID = base64.b64decode(
    "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl"
).decode()
_ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_ANTHROPIC_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_ANTHROPIC_SCOPES = "org:create_api_key user:profile user:inference"


def anthropic_login(
    on_url: callable,
    on_prompt: callable,
    on_status: callable | None = None,
) -> dict:
    """Run the Anthropic OAuth PKCE flow.

    Parameters
    ----------
    on_url : callable(url: str)
        Called with the authorization URL (caller should open browser).
    on_prompt : callable(message: str) -> str
        Called to prompt the user for the authorization code.
    on_status : callable(message: str), optional
        Progress messages.

    Returns
    -------
    dict with keys: access, refresh, expires (epoch ms)
    """
    verifier, challenge = _generate_pkce()

    params = urlencode({
        "code": "true",
        "client_id": _ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTHROPIC_REDIRECT_URI,
        "scope": _ANTHROPIC_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    })
    auth_url = f"{_ANTHROPIC_AUTHORIZE_URL}?{params}"

    on_url(auth_url)

    if on_status:
        on_status("Waiting for authorization code…")

    raw = on_prompt("Paste the authorization code:")
    raw = raw.strip()

    # Format is "code#state"
    parts = raw.split("#", 1)
    code = parts[0]
    state = parts[1] if len(parts) > 1 else ""

    if on_status:
        on_status("Exchanging code for tokens…")

    resp = httpx.post(
        _ANTHROPIC_TOKEN_URL,
        json={
            "grant_type": "authorization_code",
            "client_id": _ANTHROPIC_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": _ANTHROPIC_REDIRECT_URI,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")

    data = resp.json()

    # 5-minute buffer before expiry
    expires_at = int(time.time() * 1000) + data["expires_in"] * 1000 - 5 * 60 * 1000

    return {
        "access": data["access_token"],
        "refresh": data["refresh_token"],
        "expires": expires_at,
    }


def anthropic_refresh(refresh_token: str) -> dict:
    """Refresh an Anthropic OAuth token.

    Returns dict with keys: access, refresh, expires.
    """
    resp = httpx.post(
        _ANTHROPIC_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "client_id": _ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Anthropic token refresh failed ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    return {
        "access": data["access_token"],
        "refresh": data["refresh_token"],
        "expires": int(time.time() * 1000) + data["expires_in"] * 1000 - 5 * 60 * 1000,
    }


# ── Auth Manager ─────────────────────────────────────────────────────

class AuthManager:
    """Manages credentials from auth.json, OAuth, and env vars.

    Resolution priority (per provider):
      1. OAuth token  → subscription, free at point of use
      2. auth.json API key → explicit file-based key
      3. Environment variable → ANTHROPIC_API_KEY etc.

    This ordering prevents surprise costs when a subscription is available.
    """

    def __init__(self, auth_path: Path = AUTH_FILE) -> None:
        self.auth_path = auth_path
        self._data: dict[str, Any] = {}
        self.reload()

    # ── file I/O ─────────────────────────────────────────────────

    def reload(self) -> None:
        """Load credentials from disk."""
        if self.auth_path.exists():
            try:
                self._data = json.loads(self.auth_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Write credentials to disk (0600 permissions)."""
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_path.write_text(json.dumps(self._data, indent=2))
        self.auth_path.chmod(0o600)

    # ── credential CRUD ──────────────────────────────────────────

    def get(self, provider: str) -> dict | None:
        return self._data.get(provider)

    def set(self, provider: str, credential: dict) -> None:
        self._data[provider] = credential
        self.save()

    def remove(self, provider: str) -> None:
        self._data.pop(provider, None)
        self.save()

    def has(self, provider: str) -> bool:
        return provider in self._data

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store a provider API key in auth.json for cross-session reuse."""
        self.set(provider, {"type": "api_key", "key": api_key.strip()})

    # ── OAuth login/logout ───────────────────────────────────────

    def login_anthropic(
        self,
        on_url: callable,
        on_prompt: callable,
        on_status: callable | None = None,
    ) -> None:
        """Run Anthropic OAuth and store the result."""
        creds = anthropic_login(on_url, on_prompt, on_status)
        self.set("anthropic", {"type": "oauth", **creds})

    def logout(self, provider: str) -> None:
        """Remove stored credentials for a provider."""
        self.remove(provider)

    # ── token refresh ────────────────────────────────────────────

    def _ensure_fresh(self, provider: str) -> dict | None:
        """If the OAuth token is expired, refresh it.

        Returns the (possibly updated) credential dict, or None.
        """
        cred = self.get(provider)
        if not cred or cred.get("type") != "oauth":
            return cred

        if time.time() * 1000 < cred.get("expires", 0):
            return cred  # still valid

        # Refresh
        log.info("Refreshing expired %s OAuth token…", provider)
        try:
            if provider == "anthropic":
                new_creds = anthropic_refresh(cred["refresh"])
                self.set(provider, {"type": "oauth", **new_creds})
                return self.get(provider)
            else:
                log.warning("Don't know how to refresh %s OAuth token", provider)
                return cred
        except Exception:
            log.exception("Failed to refresh %s token", provider)
            return None

    # ── key resolution ───────────────────────────────────────────

    def resolve_api_key(self, provider: str) -> AuthSource | None:
        """Resolve the API key for *provider* using priority ordering.

        Returns an ``AuthSource`` or ``None`` if no credentials found.

        Priority:
          1. OAuth token (subscription — free at point of use)
          2. API key from auth.json
          3. Environment variable
        """
        # 1. OAuth token
        cred = self._ensure_fresh(provider)
        if cred and cred.get("type") == "oauth":
            access = cred.get("access")
            if access:
                return AuthSource(
                    key=access,
                    source=self._oauth_label(provider),
                    provider=provider,
                    is_subscription=True,
                )

        # 2. API key from auth.json
        if cred and cred.get("type") == "api_key":
            key = cred.get("key", "")
            if key:
                return AuthSource(
                    key=key,
                    source="auth.json",
                    provider=provider,
                    is_subscription=False,
                )

        # 3. Environment variable
        env_var = ENV_KEY_MAP.get(provider)
        if env_var:
            val = os.environ.get(env_var)
            if val:
                return AuthSource(
                    key=val,
                    source=f"env ${env_var}",
                    provider=provider,
                    is_subscription=False,
                )

        return None

    def resolve_for_model(self, model_str: str) -> AuthSource | None:
        """Resolve auth for a litellm model string like ``anthropic/claude-...``."""
        provider = self._provider_from_model(model_str)
        if provider:
            return self.resolve_api_key(provider)
        return None

    # ── info ─────────────────────────────────────────────────────

    def auth_summary(self) -> list[dict[str, str]]:
        """Return a summary of all configured auth sources for display."""
        providers = set(ENV_KEY_MAP.keys()) | set(self._data.keys())
        results = []
        for p in sorted(providers):
            source = self.resolve_api_key(p)
            if source:
                results.append({
                    "provider": p,
                    "source": source.source,
                    "subscription": "yes" if source.is_subscription else "no",
                    "key_preview": source.key[:8] + "…" + source.key[-4:]
                    if len(source.key) > 16 else "***",
                })
        return results

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _oauth_label(provider: str) -> str:
        labels = {
            "anthropic": "oauth (Claude Pro/Max)",
        }
        return labels.get(provider, f"oauth ({provider})")

    @staticmethod
    def _provider_from_model(model_str: str) -> str | None:
        """Extract provider name from a litellm model string."""
        # "anthropic/claude-sonnet-4-20250514" → "anthropic"
        # "gemini/gemini-2.0-flash" → "gemini"
        # "openai/gpt-4o" → "openai"
        if "/" in model_str:
            return model_str.split("/", 1)[0]
        # Heuristic for bare model names
        lower = model_str.lower()
        if "claude" in lower:
            return "anthropic"
        if "gpt" in lower or "o1" in lower or "o3" in lower:
            return "openai"
        if "gemini" in lower:
            return "gemini"
        return None


# ── module singleton ─────────────────────────────────────────────────

_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get or create the global AuthManager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
