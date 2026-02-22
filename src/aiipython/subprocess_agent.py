"""Sub-agent proxy that runs ReactiveAgent in a dedicated child process."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import threading
import weakref
from typing import Any


_LIVE_PROXIES: "weakref.WeakSet[SubprocessAgentProxy]" = weakref.WeakSet()


def _configure_worker_lm(model: str) -> None:
    """Configure dspy LM inside the child process to match parent model."""
    import dspy

    from aiipython.lm_factory import create_lm

    lm = create_lm(model)

    dspy.configure(lm=lm)


def _worker_main(
    conn: Any,
    state_blob: bytes,
    initial_task: str | None,
    model: str,
) -> None:
    """Child-process RPC worker for a single sub-agent."""
    try:
        _configure_worker_lm(model)

        from IPython.core.interactiveshell import InteractiveShell

        from aiipython.agent import ReactiveAgent
        from aiipython.checkpoint import load_clone
        from aiipython.kernel import Kernel

        kernel = Kernel(shell=InteractiveShell(), enable_checkpoints=False)
        agent = ReactiveAgent(kernel)
        load_clone(kernel, state_blob, agent=agent)

        if initial_task:
            kernel.push_user_input(initial_task, source="spawn", is_clipboard=False)

        conn.send({"ok": True, "pid": os.getpid()})

        while True:
            req = conn.recv()
            op = str(req.get("op", ""))

            if op == "close":
                conn.send({"ok": True})
                break

            if op == "react":
                task = req.get("task")
                if isinstance(task, str) and task.strip():
                    kernel.push_user_input(task, source="spawn", is_clipboard=False)
                responses = agent.react()
                conn.send({"ok": True, "responses": responses})
                continue

            if op == "push_user_input":
                text = str(req.get("text", ""))
                source = str(req.get("source", "chat"))
                is_clipboard = bool(req.get("is_clipboard", False))
                kernel.push_user_input(text, source=source, is_clipboard=is_clipboard)
                conn.send({"ok": True})
                continue

            if op == "execute":
                code = str(req.get("code", ""))
                tag = req.get("tag")
                tag = str(tag) if tag is not None else None
                entry = kernel.execute(code, tag=tag)
                conn.send({"ok": True, "entry": entry})
                continue

            if op == "snapshot":
                conn.send({"ok": True, "snapshot": kernel.snapshot()})
                continue

            if op == "history":
                limit = int(req.get("limit", 50))
                conn.send({"ok": True, "history": kernel.history[-limit:]})
                continue

            conn.send({"ok": False, "error": f"Unknown op: {op}"})

    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _close_live_proxies() -> None:
    for proxy in list(_LIVE_PROXIES):
        try:
            proxy.close()
        except Exception:
            pass


atexit.register(_close_live_proxies)


class _KernelProxy:
    """Minimal remote kernel facade for compatibility."""

    def __init__(self, owner: "SubprocessAgentProxy") -> None:
        self._owner = owner

    def push_user_input(
        self,
        text: str,
        *,
        source: str = "chat",
        is_clipboard: bool = False,
    ) -> None:
        self._owner.push_user_input(text, source=source, is_clipboard=is_clipboard)

    def execute(self, code: str, tag: str | None = None) -> dict[str, Any]:
        return self._owner.execute(code, tag=tag)

    def snapshot(self) -> dict[str, str]:
        return self._owner.snapshot()

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._owner.history()


class SubprocessAgentProxy:
    """Client-side proxy for a sub-agent running in a dedicated process."""

    def __init__(self, *, state_blob: bytes, task: str | None, model: str) -> None:
        self._ctx = mp.get_context("spawn")
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, state_blob, task, model),
            daemon=True,
        )
        self._lock = threading.Lock()
        self._closed = False

        self._proc.start()
        child_conn.close()

        ready = self._recv(timeout=15.0)
        if not ready.get("ok"):
            raise RuntimeError(ready.get("error", "sub-agent failed to start"))

        self._pid = int(ready.get("pid", -1))
        self.kernel = _KernelProxy(self)
        _LIVE_PROXIES.add(self)

    @property
    def pid(self) -> int:
        return self._pid

    def _recv(self, *, timeout: float = 30.0) -> dict[str, Any]:
        if not self._conn.poll(timeout):
            if not self._proc.is_alive():
                raise RuntimeError("sub-agent process exited")
            raise TimeoutError("sub-agent response timeout")
        msg = self._conn.recv()
        if not isinstance(msg, dict):
            raise RuntimeError("invalid sub-agent response")
        return msg

    def _rpc(self, op: str, **kwargs: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("sub-agent is closed")

        with self._lock:
            self._conn.send({"op": op, **kwargs})
            resp = self._recv()

        if not resp.get("ok"):
            raise RuntimeError(str(resp.get("error", "sub-agent request failed")))
        return resp

    def react(self, task: str | None = None) -> list[str]:
        """Run one reactive cycle (optionally with a new task)."""
        resp = self._rpc("react", task=task)
        return list(resp.get("responses", []))

    def ask(self, text: str) -> list[str]:
        """Alias for react(task=text)."""
        return self.react(task=text)

    def push_user_input(
        self,
        text: str,
        *,
        source: str = "chat",
        is_clipboard: bool = False,
    ) -> None:
        self._rpc(
            "push_user_input",
            text=text,
            source=source,
            is_clipboard=is_clipboard,
        )

    def execute(self, code: str, tag: str | None = None) -> dict[str, Any]:
        resp = self._rpc("execute", code=code, tag=tag)
        return dict(resp.get("entry", {}))

    def snapshot(self) -> dict[str, str]:
        resp = self._rpc("snapshot")
        return dict(resp.get("snapshot", {}))

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        resp = self._rpc("history", limit=limit)
        return list(resp.get("history", []))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            with self._lock:
                if self._proc.is_alive():
                    self._conn.send({"op": "close"})
                    _ = self._recv(timeout=3.0)
        except Exception:
            pass

        try:
            if self._proc.is_alive():
                self._proc.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        except Exception:
            pass

        try:
            self._conn.close()
        except Exception:
            pass

        _LIVE_PROXIES.discard(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else ("alive" if self._proc.is_alive() else "dead")
        return f"SubprocessAgentProxy(pid={self._pid}, status={status})"
