"""OAuth authentication and API key management.

Supports:
  - Anthropic OAuth (Claude Pro/Max subscription)
  - OpenAI Codex OAuth (ChatGPT Plus/Pro subscription)
  - API keys from ``~/.aiipython/auth.json``
  - Host credential import from Codex/Pi auth files
  - Environment variable fallback

Priority (highest → lowest):
  1. OAuth token from ``auth.json``  (subscription — free at point of use)
  2. API key from ``auth.json``       (explicit file-based key)
  3. Environment variable             (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
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
from threading import Event, Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

log = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────

AUTH_DIR = Path.home() / ".aiipython"
AUTH_FILE = AUTH_DIR / "auth.json"
PI_AUTH_FILE = Path.home() / ".pi" / "agent" / "auth.json"
CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"

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
    source: str
    provider: str
    is_subscription: bool = False

    @property
    def display(self) -> str:
        return f"🔑 {self.source}"


# ── utilities ────────────────────────────────────────────────────────

def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge) for PKCE S256."""
    verifier_bytes = secrets.token_bytes(32)
    verifier = _base64url(verifier_bytes)
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        pad = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + pad)
        data = json.loads(decoded.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _jwt_exp_ms(token: str) -> int | None:
    payload = _jwt_payload(token)
    if not payload:
        return None
    exp = payload.get("exp")
    if isinstance(exp, (int, float)):
        # Keep a 5-minute safety margin.
        return int(exp * 1000 - 5 * 60 * 1000)
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    """Run the Anthropic OAuth PKCE flow."""
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

    raw = on_prompt("Paste the authorization code:").strip()

    # Anthropic returns "code#state"
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
    expires_at = _now_ms() + data["expires_in"] * 1000 - 5 * 60 * 1000

    return {
        "access": data["access_token"],
        "refresh": data["refresh_token"],
        "expires": expires_at,
    }


def anthropic_refresh(refresh_token: str) -> dict:
    """Refresh an Anthropic OAuth token."""
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
        "expires": _now_ms() + data["expires_in"] * 1000 - 5 * 60 * 1000,
    }


# ── OpenAI Codex OAuth ───────────────────────────────────────────────

_OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
_OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
_OPENAI_CODEX_SCOPE = "openid profile email offline_access"
_OPENAI_JWT_CLAIM_PATH = "https://api.openai.com/auth"


class _OpenAICallbackServer:
    def __init__(self, expected_state: str) -> None:
        self.expected_state = expected_state
        self.code: str | None = None
        self._event = Event()
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> bool:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    if parsed.path != "/auth/callback":
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"Not found")
                        return

                    q = parse_qs(parsed.query)
                    state = (q.get("state") or [""])[0]
                    code = (q.get("code") or [""])[0]

                    if state != parent.expected_state:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"State mismatch")
                        return

                    if not code:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"Missing authorization code")
                        return

                    parent.code = code
                    parent._event.set()

                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><p>Authentication successful. Return to your terminal.</p></body></html>"
                    )
                except Exception:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"Internal error")

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        try:
            self._server = HTTPServer(("127.0.0.1", 1455), Handler)
        except OSError:
            return False

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return True

    def wait_for_code(self, timeout_s: float = 180.0) -> str | None:
        if self._event.wait(timeout=timeout_s):
            return self.code
        return None

    def close(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass


def _parse_openai_auth_input(raw: str) -> tuple[str | None, str | None]:
    value = raw.strip()
    if not value:
        return None, None

    # Full callback URL case.
    try:
        parsed = urlparse(value)
        q = parse_qs(parsed.query)
        code = (q.get("code") or [None])[0]
        state = (q.get("state") or [None])[0]
        if code:
            return code, state
    except Exception:
        pass

    # "code#state" format.
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None

    # querystring-like format: code=...&state=...
    if "code=" in value:
        q = parse_qs(value)
        return (q.get("code") or [None])[0], (q.get("state") or [None])[0]

    # Just the code.
    return value, None


def _openai_account_id_from_access(access_token: str) -> str | None:
    payload = _jwt_payload(access_token)
    if not payload:
        return None
    auth = payload.get(_OPENAI_JWT_CLAIM_PATH)
    if isinstance(auth, dict):
        account = auth.get("chatgpt_account_id")
        if isinstance(account, str) and account:
            return account
    return None


def openai_codex_login(
    on_url: callable,
    on_prompt: callable,
    on_status: callable | None = None,
) -> dict:
    """Run OpenAI Codex OAuth (ChatGPT Plus/Pro subscription)."""
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    params = urlencode({
        "response_type": "code",
        "client_id": _OPENAI_CODEX_CLIENT_ID,
        "redirect_uri": _OPENAI_CODEX_REDIRECT_URI,
        "scope": _OPENAI_CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "aiipython",
    })
    auth_url = f"{_OPENAI_CODEX_AUTHORIZE_URL}?{params}"

    callback_server = _OpenAICallbackServer(expected_state=state)
    callback_ready = callback_server.start()

    on_url(auth_url)

    if on_status:
        if callback_ready:
            on_status("Waiting for browser callback on http://127.0.0.1:1455/auth/callback…")
        else:
            on_status("Could not bind local callback server, falling back to manual code paste…")

    try:
        code = callback_server.wait_for_code(timeout_s=180) if callback_ready else None

        if not code:
            raw = on_prompt("Paste the authorization code (or full redirect URL):")
            parsed_code, parsed_state = _parse_openai_auth_input(raw)
            if parsed_state and parsed_state != state:
                raise RuntimeError("State mismatch")
            code = parsed_code

        if not code:
            raise RuntimeError("Missing authorization code")

        if on_status:
            on_status("Exchanging code for tokens…")

        resp = httpx.post(
            _OPENAI_CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": _OPENAI_CODEX_CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": _OPENAI_CODEX_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAI token exchange failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        if not data.get("access_token") or not data.get("refresh_token"):
            raise RuntimeError("OpenAI token exchange response missing required fields")

        access = data["access_token"]
        expires = _now_ms() + int(data.get("expires_in", 3600)) * 1000 - 5 * 60 * 1000

        out = {
            "access": access,
            "refresh": data["refresh_token"],
            "expires": expires,
        }
        account_id = _openai_account_id_from_access(access)
        if account_id:
            out["accountId"] = account_id
        return out
    finally:
        callback_server.close()


def openai_codex_refresh(refresh_token: str) -> dict:
    """Refresh an OpenAI Codex OAuth token."""
    resp = httpx.post(
        _OPENAI_CODEX_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OPENAI_CODEX_CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenAI token refresh failed ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    if not data.get("access_token") or not data.get("refresh_token"):
        raise RuntimeError("OpenAI token refresh response missing required fields")

    access = data["access_token"]
    out = {
        "access": access,
        "refresh": data["refresh_token"],
        "expires": _now_ms() + int(data.get("expires_in", 3600)) * 1000 - 5 * 60 * 1000,
    }
    account_id = _openai_account_id_from_access(access)
    if account_id:
        out["accountId"] = account_id
    return out


# ── Auth Manager ─────────────────────────────────────────────────────

class AuthManager:
    """Manages credentials from auth.json, OAuth, host auth files, and env vars."""

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
        if provider == "openai":
            return "openai" in self._data or "openai-codex" in self._data
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

    def login_openai(
        self,
        on_url: callable,
        on_prompt: callable,
        on_status: callable | None = None,
    ) -> None:
        """Run OpenAI Codex OAuth and store the result."""
        creds = openai_codex_login(on_url, on_prompt, on_status)
        self.set("openai-codex", {"type": "oauth", **creds})

    def logout(self, provider: str) -> None:
        """Log out a provider and disable automatic host-import fallback for it."""
        if provider == "openai":
            # Disable both OpenAI API-key and Codex OAuth imports.
            self._data["openai"] = {"type": "disabled"}
            self._data["openai-codex"] = {"type": "disabled"}
            self.save()
            return

        self._data[provider] = {"type": "disabled"}
        self.save()

    # ── host credential import ───────────────────────────────────

    @staticmethod
    def _normalize_oauth(raw: dict[str, Any]) -> dict[str, Any] | None:
        typ = str(raw.get("type", "")).lower()
        if typ not in {"oauth", "o_auth"}:
            return None

        access = raw.get("access") or raw.get("access_token")
        refresh = raw.get("refresh") or raw.get("refresh_token")
        expires = raw.get("expires")

        if not isinstance(access, str) or not access.strip():
            return None
        if not isinstance(refresh, str) or not refresh.strip():
            return None

        if not isinstance(expires, (int, float)):
            expires = _jwt_exp_ms(access) or (_now_ms() + 50 * 60 * 1000)

        out: dict[str, Any] = {
            "type": "oauth",
            "access": access,
            "refresh": refresh,
            "expires": int(expires),
        }

        account_id = raw.get("accountId") or raw.get("account_id")
        if isinstance(account_id, str) and account_id:
            out["accountId"] = account_id

        return out

    def _import_external_provider_if_missing(self, provider: str) -> None:
        if provider in self._data:
            return

        imported: dict[str, Any] | None = None

        if provider == "openai-codex":
            # Prefer Pi auth, then Codex auth.
            pi_auth = _read_json_file(PI_AUTH_FILE) or {}
            raw = pi_auth.get("openai-codex")
            if isinstance(raw, dict):
                imported = self._normalize_oauth(raw)

            if imported is None:
                codex_auth = _read_json_file(CODEX_AUTH_FILE) or {}
                tokens = codex_auth.get("tokens")
                if isinstance(tokens, dict):
                    access = tokens.get("access_token")
                    refresh = tokens.get("refresh_token")
                    if isinstance(access, str) and isinstance(refresh, str):
                        imported = {
                            "type": "oauth",
                            "access": access,
                            "refresh": refresh,
                            "expires": _jwt_exp_ms(access) or (_now_ms() + 50 * 60 * 1000),
                        }
                        account_id = tokens.get("account_id")
                        if isinstance(account_id, str) and account_id:
                            imported["accountId"] = account_id

        elif provider == "anthropic":
            # Pi stores Anthropic OAuth as type "o_auth".
            pi_auth = _read_json_file(PI_AUTH_FILE) or {}
            raw = pi_auth.get("anthropic")
            if isinstance(raw, dict):
                imported = self._normalize_oauth(raw)

        elif provider == "openai":
            # API key fallback from Pi/Codex auth files.
            pi_auth = _read_json_file(PI_AUTH_FILE) or {}
            raw = pi_auth.get("openai")
            if isinstance(raw, dict) and raw.get("type") == "api_key":
                key = raw.get("key")
                if isinstance(key, str) and key.strip():
                    imported = {"type": "api_key", "key": key.strip()}

            if imported is None:
                codex_auth = _read_json_file(CODEX_AUTH_FILE) or {}
                key = codex_auth.get("OPENAI_API_KEY")
                if isinstance(key, str) and key.strip():
                    imported = {"type": "api_key", "key": key.strip()}

        if imported is not None:
            self.set(provider, imported)

    # ── token refresh ────────────────────────────────────────────

    def _ensure_fresh(self, provider: str) -> dict | None:
        """If an OAuth token is expired, refresh it."""
        self._import_external_provider_if_missing(provider)

        cred = self.get(provider)
        if not cred or cred.get("type") != "oauth":
            return cred

        if time.time() * 1000 < cred.get("expires", 0):
            return cred

        log.info("Refreshing expired %s OAuth token…", provider)
        try:
            if provider == "anthropic":
                new_creds = anthropic_refresh(cred["refresh"])
            elif provider == "openai-codex":
                new_creds = openai_codex_refresh(cred["refresh"])
            elif provider == "openai":
                # Allow legacy storage under "openai" if present.
                new_creds = openai_codex_refresh(cred["refresh"])
            else:
                log.warning("Don't know how to refresh %s OAuth token", provider)
                return cred

            self.set(provider, {"type": "oauth", **new_creds})
            return self.get(provider)
        except Exception:
            log.exception("Failed to refresh %s token", provider)
            return None

    # ── key resolution ───────────────────────────────────────────

    def _resolve_openai_api_key(self) -> AuthSource | None:
        """Resolve OpenAI Platform API key (for litellm openai/* models)."""
        cred = self._ensure_fresh("openai")
        if cred and cred.get("type") == "api_key":
            key = cred.get("key", "")
            if key:
                return AuthSource(
                    key=key,
                    source="auth.json",
                    provider="openai",
                    is_subscription=False,
                )

        val = os.environ.get("OPENAI_API_KEY")
        if val:
            return AuthSource(
                key=val,
                source="env $OPENAI_API_KEY",
                provider="openai",
                is_subscription=False,
            )

        return None

    def _resolve_openai_codex_oauth(self) -> AuthSource | None:
        """Resolve ChatGPT Plus/Pro Codex OAuth token (pi-style provider)."""
        cred = self._ensure_fresh("openai-codex")
        if cred and cred.get("type") == "oauth":
            access = cred.get("access")
            if access:
                return AuthSource(
                    key=access,
                    source=self._oauth_label("openai-codex"),
                    provider="openai-codex",
                    is_subscription=True,
                )
        return None

    def resolve_api_key(self, provider: str) -> AuthSource | None:
        """Resolve API key for *provider* using priority ordering."""
        if provider == "openai":
            return self._resolve_openai_api_key()
        if provider == "openai-codex":
            return self._resolve_openai_codex_oauth()

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
                    if len(source.key) > 16
                    else "***",
                })
        return results

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _oauth_label(provider: str) -> str:
        labels = {
            "anthropic": "oauth (Claude Pro/Max)",
            "openai-codex": "oauth (ChatGPT Plus/Pro Codex)",
        }
        return labels.get(provider, f"oauth ({provider})")

    @staticmethod
    def _provider_from_model(model_str: str) -> str | None:
        """Extract provider name from a litellm model string."""
        if "/" in model_str:
            return model_str.split("/", 1)[0]

        lower = model_str.lower()
        if "claude" in lower:
            return "anthropic"
        if "gpt" in lower or "o1" in lower or "o3" in lower or "codex" in lower:
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
