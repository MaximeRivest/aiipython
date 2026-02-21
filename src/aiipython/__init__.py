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
    lm_backend: str | None = None,
    *,
    defer_lm_setup: bool = False,
):
    """Resolve model/auth and return the live aiipython session."""
    import os

    import dspy

    from aiipython.auth import ENV_KEY_MAP, get_auth_manager
    from aiipython.settings import DEFAULT_MODEL, get_settings

    auth = get_auth_manager()
    settings = get_settings()

    if lm_backend:
        os.environ["AIIPYTHON_LM_BACKEND"] = lm_backend

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

    if defer_lm_setup:
        _reset_dspy_config_ownership()
    elif not hasattr(dspy.settings, "lm") or dspy.settings.lm is None or model:
        from aiipython.lm_factory import create_lm

        lm = create_lm(model_str)
        dspy.configure(lm=lm)

    from aiipython.wire import get_wire_log

    get_wire_log()

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


def _run_textual(session) -> None:
    """Run legacy Textual frontend."""
    import os

    if "AIIPYTHON_BG" not in os.environ and "PYCODE_BG" not in os.environ:
        detected_bg = _detect_terminal_background()
        os.environ["AIIPYTHON_BG"] = detected_bg or "ansi_default"

    if "AIIPYTHON_BG" in os.environ:
        os.environ["PYCODE_BG"] = os.environ["AIIPYTHON_BG"]
    elif "PYCODE_BG" in os.environ:
        os.environ["AIIPYTHON_BG"] = os.environ["PYCODE_BG"]

    from aiipython.app import AiiPythonApp

    app = AiiPythonApp(session=session)

    inline_env = (
        os.environ.get("AIIPYTHON_INLINE")
        or os.environ.get("PYCODE_INLINE")
        or "1"
    ).lower()
    inline = inline_env in ("1", "true", "yes")

    os.environ["AIIPYTHON_INLINE"] = "1" if inline else "0"
    os.environ["PYCODE_INLINE"] = os.environ["AIIPYTHON_INLINE"]

    app.run(inline=inline, inline_no_clear=inline)


def chat(
    model: str | None = None,
    ui: str | None = None,
    lm_backend: str | None = None,
) -> None:
    """Launch aiipython from an IPython session.

    Args:
        model: Optional model override (``provider/model``).
        ui: Frontend selector:
            - ``"pi-native"`` (default, real Pi InteractiveMode)
            - ``"pi-tui"`` (custom frontend)
            - ``"textual"`` (legacy)
        lm_backend: Optional LM routing override:
            - ``"auto"`` (default behavior)
            - ``"pi"``
            - ``"litellm"``

    Usage::

        In [1]: from aiipython import chat
        In [2]: chat()
        In [3]: chat("openai/gpt-4o-mini")
        In [4]: chat(ui="textual")
        In [5]: chat(lm_backend="pi")
    """
    import os
    import sys

    ui_choice = (ui or os.environ.get("AIIPYTHON_UI") or "pi-native").strip().lower()

    session = _prepare_session(
        model=model,
        lm_backend=lm_backend,
        defer_lm_setup=ui_choice in {"pi-native", "native", "pi-full", "pi-interactive"},
    )

    if ui_choice in {"textual", "legacy", "pycode"}:
        _run_textual(session)
        return

    if ui_choice in {"pi-native", "native", "pi-full", "pi-interactive"}:
        from aiipython.pi_tui_bridge import run_pi_native

        try:
            run_pi_native(session)
            return
        except Exception as exc:
            strict = (os.environ.get("AIIPYTHON_UI_STRICT") or "").lower() in {"1", "true", "yes"}
            if strict:
                raise
            print(
                f"[aiipython] pi-native frontend failed ({exc}); falling back to pi-tui.",
                file=sys.stderr,
            )

            # pi-native may defer LM setup to the backend thread; ensure LM is ready
            # before using pi-tui/textual fallback paths.
            try:
                import dspy

                if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
                    session.switch_model(session.model)
            except Exception:
                pass

            from aiipython.pi_tui_bridge import run_pi_tui

            try:
                run_pi_tui(session)
                return
            except Exception as exc2:
                print(
                    f"[aiipython] pi-tui frontend failed ({exc2}); falling back to Textual.",
                    file=sys.stderr,
                )
                _run_textual(session)
                return

    if ui_choice in {"pi", "pi-tui", "pitui"}:
        from aiipython.pi_tui_bridge import run_pi_tui

        try:
            run_pi_tui(session)
            return
        except Exception as exc:
            strict = (os.environ.get("AIIPYTHON_UI_STRICT") or "").lower() in {"1", "true", "yes"}
            if strict:
                raise
            print(
                f"[aiipython] pi-tui frontend failed ({exc}); falling back to Textual.",
                file=sys.stderr,
            )
            _run_textual(session)
            return

    raise ValueError(f"Unknown ui '{ui_choice}'. Use 'pi-native', 'pi-tui' or 'textual'.")
