"""Node Pi gateway process manager + HTTP client.

Provides an on-demand bridge to Pi's model/auth stack via a local Node server.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

_GATEWAY_LOCK = threading.Lock()
_GATEWAY_PROCESS: "PiGatewayProcess | None" = None


class PiGatewayProcess:
    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._base_url: str | None = None
        self._port: int | None = None

    @property
    def base_url(self) -> str:
        if not self._base_url:
            raise RuntimeError("Pi gateway is not running")
        return self._base_url

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self._base_url = None
        self._port = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def ensure_running(self) -> None:
        if self._proc and self._proc.poll() is None and self._base_url:
            return

        self.stop()

        node = shutil.which("node")
        if not node:
            raise RuntimeError(
                "Node.js is required for AIIPYTHON_LM_BACKEND=pi. "
                "Install Node.js or use AIIPYTHON_LM_BACKEND=litellm."
            )

        gateway_dir = _gateway_dir()
        _ensure_gateway_dependencies(gateway_dir)

        entry = gateway_dir / "pi_gateway_server.mjs"
        if not entry.exists():
            raise RuntimeError(f"Pi gateway server entry missing: {entry}")

        host = "127.0.0.1"
        port = _pick_free_port(host)
        env = os.environ.copy()
        env["AIIPYTHON_PI_GATEWAY_HOST"] = host
        env["AIIPYTHON_PI_GATEWAY_PORT"] = str(port)

        proc = subprocess.Popen(
            [node, str(entry)],
            cwd=str(gateway_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self._proc = proc
        self._port = port
        self._base_url = f"http://{host}:{port}"

        ready = self._wait_ready(proc, self._base_url)
        if not ready:
            stderr = ""
            try:
                if proc.stderr:
                    stderr = proc.stderr.read()[-2000:]
            except Exception:
                pass
            self.stop()
            raise RuntimeError(f"Failed to start Pi gateway.{(' stderr: ' + stderr) if stderr else ''}")

    def _wait_ready(self, proc: subprocess.Popen[str], base_url: str, timeout_s: float = 10.0) -> bool:
        deadline = time.time() + timeout_s

        # Fast path: parse stdout readiness line.
        if proc.stdout is not None:
            proc.stdout.flush()

        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            try:
                r = httpx.get(f"{base_url}/health", timeout=0.5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.15)

        return False


def _gateway_dir() -> Path:
    return Path(__file__).resolve().parent / "pi_gateway"


def _ensure_gateway_dependencies(gateway_dir: Path) -> None:
    marker = gateway_dir / "node_modules" / "@mariozechner" / "pi-ai"
    if marker.exists():
        return

    pkg = gateway_dir / "package.json"
    if not pkg.exists():
        raise RuntimeError(f"Pi gateway package.json missing: {pkg}")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError(
            "npm is required for AIIPYTHON_LM_BACKEND=pi. "
            "Install Node.js/npm or use AIIPYTHON_LM_BACKEND=litellm."
        )

    proc = subprocess.run(
        [npm, "install", "--no-audit", "--no-fund"],
        cwd=str(gateway_dir),
        check=False,
    )
    if proc.returncode != 0 or not marker.exists():
        raise RuntimeError(
            "Failed to install Pi gateway dependencies. "
            f"Try: cd {gateway_dir} && npm install"
        )


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _get_gateway_process() -> PiGatewayProcess:
    global _GATEWAY_PROCESS
    with _GATEWAY_LOCK:
        if _GATEWAY_PROCESS is None:
            _GATEWAY_PROCESS = PiGatewayProcess()
        _GATEWAY_PROCESS.ensure_running()
        return _GATEWAY_PROCESS


def stop_gateway() -> None:
    global _GATEWAY_PROCESS
    with _GATEWAY_LOCK:
        if _GATEWAY_PROCESS is not None:
            _GATEWAY_PROCESS.stop()
            _GATEWAY_PROCESS = None


atexit.register(stop_gateway)


class PiGatewayClient:
    def __init__(self) -> None:
        self._http = httpx.Client(timeout=300)

    def close(self) -> None:
        self._http.close()

    def health(self) -> dict[str, Any]:
        gp = _get_gateway_process()
        r = self._http.get(f"{gp.base_url}/health")
        r.raise_for_status()
        return r.json()

    def list_models(self) -> list[dict[str, Any]]:
        gp = _get_gateway_process()
        r = self._http.get(f"{gp.base_url}/models")
        r.raise_for_status()
        body = r.json()
        return body.get("models", []) if isinstance(body, dict) else []

    def auth_status(self) -> dict[str, Any]:
        gp = _get_gateway_process()
        r = self._http.get(f"{gp.base_url}/auth/status")
        r.raise_for_status()
        return r.json()

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        gp = _get_gateway_process()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options or {},
        }
        if api_key:
            payload["apiKey"] = api_key

        r = self._http.post(f"{gp.base_url}/lm/complete", json=payload)
        r.raise_for_status()
        body = r.json()
        if not isinstance(body, dict):
            raise RuntimeError("Invalid Pi gateway response")
        return body
