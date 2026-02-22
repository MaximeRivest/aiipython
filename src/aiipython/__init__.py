"""aiipython - A reactive AI chat assistant running inside IPython."""

from __future__ import annotations

__version__ = "0.5.0"


def _detect_terminal_background(timeout: float = 0.2) -> str | None:
    """Best-effort OSC 11 query for terminal background color.

    Returns #RRGGBB when available, otherwise None.
    """
    import os
    import re
    import select
    import sys
    import termios
    import time
    import tty

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    data = b""

    try:
        tty.setcbreak(fd)
        os.write(fd, b"\x1b]11;?\x07")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                break
            chunk = os.read(fd, 1024)
            if not chunk:
                break
            data += chunk
            if b"\x07" in data or b"\x1b\\" in data:
                break
    except Exception:
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass

    m = re.search(
        rb"\x1b\]11;rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})(?:\x07|\x1b\\)",
        data,
    )
    if not m:
        return None

    def to_8bit(raw: bytes) -> int:
        n = int(raw.decode("ascii"), 16)
        bits = len(raw) * 4
        return n if bits <= 8 else (n >> (bits - 8))

    r8 = to_8bit(m.group(1))
    g8 = to_8bit(m.group(2))
    b8 = to_8bit(m.group(3))
    return f"#{r8:02x}{g8:02x}{b8:02x}"


def _reset_dspy_config_ownership() -> None:
    """Reset DSPy configure ownership (best effort, internal API)."""
    try:
        import importlib

        _settings_mod = importlib.import_module("dspy.dsp.utils.settings")
        _settings_mod.config_owner_thread_id = None
        _settings_mod.config_owner_async_task = None
        try:
            _settings_mod.main_thread_config["lm"] = None
        except Exception:
            pass
    except Exception:
        pass


def _prepare_session(
    model: str | None = None,
    *,
    defer_lm_setup: bool = False,
):
    """Resolve model/auth and return the live aiipython session."""
    import os

    import dspy

    from aiipython.auth import ENV_KEY_MAP, get_auth_manager
    from aiipython.mlflow_integration import configure_mlflow_from_env
    from aiipython.settings import DEFAULT_MODEL, get_settings

    auth = get_auth_manager()
    settings = get_settings()

    if model:
        model_str = model
    else:
        candidates = [
            settings.get_last_model(),
            os.environ.get("AIIPYTHON_MODEL"),
            os.environ.get("PYCODE_MODEL"),
            DEFAULT_MODEL,
        ]
        model_str = next(
            (c for c in candidates if isinstance(c, str) and c.strip() and "/" in c),
            DEFAULT_MODEL,
        )

    settings.set_last_model(model_str)
    os.environ["AIIPYTHON_MODEL"] = model_str
    os.environ["PYCODE_MODEL"] = model_str

    auth_source = auth.resolve_for_model(model_str)
    if auth_source:
        env_var = ENV_KEY_MAP.get(auth_source.provider)
        if env_var:
            os.environ[env_var] = auth_source.key

    # Optional MLflow tracing for DSPy calls (must be enabled before dspy.configure).
    configure_mlflow_from_env()

    if defer_lm_setup:
        _reset_dspy_config_ownership()
    elif not hasattr(dspy.settings, "lm") or dspy.settings.lm is None or model:
        from aiipython.lm_factory import create_lm

        lm = create_lm(model_str)
        dspy.configure(lm=lm)

    shell = None
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        pass

    from aiipython.session import get_session

    session = get_session(shell=shell)
    if defer_lm_setup:
        session.model = model_str
        session.auth_source = session._resolve_auth()
    elif session.model != model_str:
        session.switch_model(model_str)

    return session


def chat(
    model: str | None = None,
    ui: str | None = None,
    mlflow: bool | None = None,
) -> None:
    """Launch aiipython from an IPython session.

    Args:
        model: Optional model override (``provider/model``).
        mlflow: Optional MLflow tracing toggle. ``True`` sets
            ``AIIPYTHON_MLFLOW=1`` for this process.
    Usage::

        In [1]: from aiipython import chat
        In [2]: chat()
        In [3]: chat("openai/gpt-4o-mini")
        In [4]: chat(mlflow=True)
    """
    import os
    import sys

    if mlflow is True:
        os.environ["AIIPYTHON_MLFLOW"] = "1"
    elif mlflow is False:
        os.environ["AIIPYTHON_MLFLOW"] = "0"

    session = _prepare_session(
        model=model,
        defer_lm_setup=True,
    )

    from aiipython.pi_native import run_pi_native

    try:
        run_pi_native(session)
    except Exception as exc:
        strict = (os.environ.get("AIIPYTHON_UI_STRICT") or "").lower() in {"1", "true", "yes"}
        if strict:
            raise
        print(
            f"[aiipython] pi-native frontend failed ({exc}).",
            file=sys.stderr,
        )
