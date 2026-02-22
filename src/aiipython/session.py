"""Session — persists kernel + agent state between TUI launches.

When you call ``chat()`` from IPython, exit the TUI, do things in the
REPL, then call ``chat()`` again, the same session is reused.  The
agent sees everything you did in between.
"""

from __future__ import annotations

import os
from typing import Any

import dspy

from aiipython.auth import AuthSource, get_auth_manager, ENV_KEY_MAP
from aiipython.kernel import Kernel
from aiipython.agent import ReactiveAgent
from aiipython.settings import get_settings


class Session:
    """Holds all state that survives TUI restarts."""

    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.agent = ReactiveAgent(kernel)
        self.model: str = getattr(dspy.settings, "lm", None) and dspy.settings.lm.model or "?"
        self.auth_source: AuthSource | None = self._resolve_auth()

    def _resolve_auth(self) -> AuthSource | None:
        """Resolve auth for the current model and inject into env."""
        auth = get_auth_manager()
        source = auth.resolve_for_model(self.model)
        if source:
            env_var = ENV_KEY_MAP.get(source.provider)
            if env_var:
                os.environ[env_var] = source.key
        return source

    def switch_model(self, model: str) -> None:
        """Hot-swap the LM without losing session state."""
        self.model = model
        get_settings().set_last_model(model)
        os.environ["AIIPYTHON_MODEL"] = model
        os.environ["PYCODE_MODEL"] = model

        from aiipython.lm_factory import create_lm
        from aiipython.mlflow_integration import configure_mlflow_from_env

        configure_mlflow_from_env()
        lm = create_lm(model)
        self.auth_source = self._resolve_auth()

        try:
            dspy.configure(lm=lm)
        except RuntimeError as exc:
            if "can only be changed by the thread that initially configured it" not in str(exc):
                raise
            # aiipython may route model changes from a worker/RPC thread.
            # Reset DSPy config ownership and retry in the current thread.
            try:
                import importlib

                _settings_mod = importlib.import_module("dspy.dsp.utils.settings")
                _settings_mod.config_owner_thread_id = None
                _settings_mod.config_owner_async_task = None
            except Exception:
                pass
            dspy.configure(lm=lm)

        # Rebuild the predict module with the new LM
        from aiipython.adapter import build_predict
        self.agent.predict = build_predict(self.agent)


# ── global singleton ────────────────────────────────────────────────

_session: Session | None = None


def get_session(shell=None) -> Session:
    """Get or create the global session.

    If *shell* is given (e.g. the running IPython shell), the kernel
    wraps it.  Otherwise ``InteractiveShell.instance()`` is used.
    """
    global _session
    if _session is None:
        kernel = Kernel(shell=shell)
        _session = Session(kernel)
    return _session


def reset_session() -> None:
    """Force a fresh session (for testing)."""
    global _session
    _session = None
