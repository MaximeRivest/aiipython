"""aiipython - A reactive AI chat assistant running inside IPython."""

__version__ = "0.2.3"


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
        # Query terminal background color (OSC 11)
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


def chat(model: str | None = None) -> None:
    """Launch the aiipython TUI from an IPython session.

    Your existing namespace is preserved.  Exit with ctrl+c, do things
    in the REPL, then call ``chat()`` again to continue.

    Usage::

        In [1]: import pandas as pd
        In [2]: df = pd.read_csv("data.csv")
        In [3]: from aiipython import chat
        In [4]: chat()             # TUI sees df, pd, etc.
        ...                        # ctrl+c to exit
        In [5]: df.describe()      # back in IPython
        In [6]: chat()             # re-enter, agent sees updated state
        In [7]: chat("openai/gpt-4o-mini")  # switch model
    """
    import os
    import dspy

    # ── Resolve auth ────────────────────────────────────────────
    # Inject the best available API key into the environment BEFORE
    # litellm/dspy see it.  Priority: OAuth > auth.json > env var.
    from aiipython.auth import get_auth_manager, ENV_KEY_MAP
    from aiipython.settings import get_settings, DEFAULT_MODEL
    auth = get_auth_manager()
    settings = get_settings()

    # Model precedence:
    #   explicit arg > persisted last_model > AIIPYTHON_MODEL env
    #   > PYCODE_MODEL env (legacy) > default
    #
    # Putting persisted state ahead of env vars ensures "latest model
    # you picked" survives new host sessions even if your shell exports
    # a stale AIIPYTHON_MODEL value.
    model_str = (
        model
        or settings.get_last_model()
        or os.environ.get("AIIPYTHON_MODEL")
        or os.environ.get("PYCODE_MODEL")  # legacy alias
        or DEFAULT_MODEL
    )
    # Persist whichever model we ended up using.
    settings.set_last_model(model_str)

    os.environ["AIIPYTHON_MODEL"] = model_str
    os.environ["PYCODE_MODEL"] = model_str  # legacy alias

    # Resolve and inject key for the active model's provider
    auth_source = auth.resolve_for_model(model_str)
    if auth_source:
        env_var = ENV_KEY_MAP.get(auth_source.provider)
        if env_var:
            os.environ[env_var] = auth_source.key

    # ── Configure LM ────────────────────────────────────────────
    if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None or model:
        from aiipython.streaming_lm import StreamingLM
        from aiipython.tabminion import is_tabminion_model, litellm_model_kwargs

        if is_tabminion_model(model_str):
            # Route through TabMinion browser bridge
            tm_kwargs = litellm_model_kwargs(model_str)
            lm = StreamingLM(
                tm_kwargs.pop("model"),
                api_base=tm_kwargs.pop("api_base"),
                api_key=tm_kwargs.pop("api_key"),
                cache=False,
            )
        else:
            lm = StreamingLM(model_str, cache=False)
        dspy.configure(lm=lm)

    from aiipython.wire import get_wire_log
    get_wire_log()

    # Get the current IPython shell if we're inside one
    shell = None
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except Exception:
        pass

    from aiipython.session import get_session
    session = get_session(shell=shell)
    if session.model != model_str:
        session.switch_model(model_str)

    # Background strategy (if caller didn't set AIIPYTHON_BG explicitly):
    # Try to detect terminal background via OSC 11 and use it as a
    # concrete color so Textual visually matches the host terminal.
    if "AIIPYTHON_BG" not in os.environ and "PYCODE_BG" not in os.environ:
        detected_bg = _detect_terminal_background()
        os.environ["AIIPYTHON_BG"] = detected_bg or "ansi_default"

    # Keep legacy alias in sync.
    if "AIIPYTHON_BG" in os.environ:
        os.environ["PYCODE_BG"] = os.environ["AIIPYTHON_BG"]
    elif "PYCODE_BG" in os.environ:
        os.environ["AIIPYTHON_BG"] = os.environ["PYCODE_BG"]

    from aiipython.app import AiiPythonApp
    app = AiiPythonApp(session=session)

    # Some terminals/compositors render alternate-screen apps with an
    # opaque background regardless of ANSI default colors.
    # Default to inline mode (better terminal-native background behavior).
    # Override with AIIPYTHON_INLINE=0/false/no for fullscreen alternate-screen.
    inline_env = (
        os.environ.get("AIIPYTHON_INLINE")
        or os.environ.get("PYCODE_INLINE")  # legacy alias
        or "1"
    ).lower()
    inline = inline_env in ("1", "true", "yes")

    # Keep legacy alias in sync.
    os.environ["AIIPYTHON_INLINE"] = "1" if inline else "0"
    os.environ["PYCODE_INLINE"] = os.environ["AIIPYTHON_INLINE"]

    app.run(inline=inline, inline_no_clear=inline)
