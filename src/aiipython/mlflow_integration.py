"""Optional MLflow DSPy tracing integration.

Enabled via environment variables so MLflow remains an optional dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MLFLOW_CONFIGURED = False
_MLFLOW_FAILURE_REPORTED = False


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in _TRUE_VALUES:
        return True
    if norm in _FALSE_VALUES:
        return False
    return default


def _default_tracking_uri() -> str:
    db_path = (Path.home() / ".aiipython" / "mlflow.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def resolve_mlflow_config() -> dict[str, str | bool]:
    """Resolve effective MLflow config from environment + defaults."""
    tracking_uri = (os.environ.get("AIIPYTHON_MLFLOW_TRACKING_URI") or "").strip()
    if not tracking_uri:
        tracking_uri = _default_tracking_uri()

    experiment = (os.environ.get("AIIPYTHON_MLFLOW_EXPERIMENT") or "").strip() or "aiipython"
    silent = _env_bool("AIIPYTHON_MLFLOW_SILENT", False)
    enabled = _env_bool("AIIPYTHON_MLFLOW", False)

    return {
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "silent": silent,
        "enabled": enabled,
    }


def _terminal_link(label: str, url: str) -> str:
    """Best-effort OSC-8 hyperlink for compatible terminals."""
    if not sys.stderr.isatty():
        return url
    esc = "\033]8;;"
    end = "\033\\"
    reset = "\033]8;;\033\\"
    return f"{esc}{url}{end}{label}{reset}"


def _emit_startup_hint(*, tracking_uri: str, experiment: str, silent: bool) -> None:
    if silent:
        return

    print("[aiipython] MLflow DSPy tracing enabled.", file=sys.stderr)
    print(f"[aiipython] Tracking URI: {tracking_uri}", file=sys.stderr)
    print(f"[aiipython] Experiment: {experiment}", file=sys.stderr)

    if tracking_uri.startswith(("http://", "https://")):
        print(f"[aiipython] MLflow UI URL: {tracking_uri}", file=sys.stderr)
        print(
            f"[aiipython] Open MLflow UI: {_terminal_link('click here', tracking_uri)}",
            file=sys.stderr,
        )
        return

    ui_url = "http://127.0.0.1:5000"
    print(f"[aiipython] Default MLflow UI URL: {ui_url}", file=sys.stderr)
    print(
        f"[aiipython] Open MLflow UI: {_terminal_link('click here', ui_url)}",
        file=sys.stderr,
    )
    print("[aiipython] In pi-native use /mlflow status for the exact running URL.", file=sys.stderr)


def configure_mlflow_from_env() -> bool:
    """Enable MLflow DSPy autologging when ``AIIPYTHON_MLFLOW`` is truthy.

    Environment variables:
      - AIIPYTHON_MLFLOW=1
      - AIIPYTHON_MLFLOW_TRACKING_URI=sqlite:///... (optional)
      - AIIPYTHON_MLFLOW_EXPERIMENT=aiipython (optional)
      - AIIPYTHON_MLFLOW_LOG_TRACES=1
      - AIIPYTHON_MLFLOW_LOG_TRACES_FROM_COMPILE=0
      - AIIPYTHON_MLFLOW_LOG_TRACES_FROM_EVAL=1
      - AIIPYTHON_MLFLOW_LOG_COMPILES=0
      - AIIPYTHON_MLFLOW_LOG_EVALS=0
      - AIIPYTHON_MLFLOW_SILENT=0
    """
    global _MLFLOW_CONFIGURED, _MLFLOW_FAILURE_REPORTED

    if _MLFLOW_CONFIGURED:
        return True

    if not _env_bool("AIIPYTHON_MLFLOW", False):
        return False

    try:
        import mlflow
    except Exception as exc:
        if not _MLFLOW_FAILURE_REPORTED:
            print(
                "[aiipython] AIIPYTHON_MLFLOW is enabled but `mlflow` is not installed "
                f"({exc}). Install it with `uv sync --extra mlflow`.",
                file=sys.stderr,
            )
            _MLFLOW_FAILURE_REPORTED = True
        return False

    try:
        cfg = resolve_mlflow_config()
        tracking_uri = str(cfg["tracking_uri"])
        experiment = str(cfg["experiment"])
        silent = bool(cfg["silent"])

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.dspy.autolog(
            log_traces=_env_bool("AIIPYTHON_MLFLOW_LOG_TRACES", True),
            log_traces_from_compile=_env_bool(
                "AIIPYTHON_MLFLOW_LOG_TRACES_FROM_COMPILE", False
            ),
            log_traces_from_eval=_env_bool(
                "AIIPYTHON_MLFLOW_LOG_TRACES_FROM_EVAL", True
            ),
            log_compiles=_env_bool("AIIPYTHON_MLFLOW_LOG_COMPILES", False),
            log_evals=_env_bool("AIIPYTHON_MLFLOW_LOG_EVALS", False),
            silent=silent,
        )

        _emit_startup_hint(tracking_uri=tracking_uri, experiment=experiment, silent=silent)
        _MLFLOW_CONFIGURED = True
        return True
    except Exception as exc:
        if not _MLFLOW_FAILURE_REPORTED:
            print(
                "[aiipython] Failed to enable MLflow DSPy autologging: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            _MLFLOW_FAILURE_REPORTED = True
        return False
