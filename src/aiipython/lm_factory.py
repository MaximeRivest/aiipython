"""LM backend selection for aiipython.

Backends:
- litellm: existing StreamingLM implementation
- pi: local Node gateway (Pi auth/model stack)
- auto: try pi first, fallback to litellm
"""

from __future__ import annotations

import os
from typing import Any


def _backend_choice() -> str:
    raw = (os.environ.get("AIIPYTHON_LM_BACKEND") or "auto").strip().lower()
    if raw in {"pi", "litellm", "auto"}:
        return raw
    return "auto"


def create_lm(model_str: str, **kwargs: Any):
    """Create the configured LM backend for *model_str*."""
    from aiipython.streaming_lm import StreamingLM
    from aiipython.tabminion import is_tabminion_model, litellm_model_kwargs

    # TabMinion is currently a litellm/OpenAI-compatible local transport.
    # Keep it on the existing path regardless of global backend choice.
    if is_tabminion_model(model_str):
        tm_kwargs = litellm_model_kwargs(model_str)
        model = tm_kwargs.pop("model")
        return StreamingLM(
            model,
            api_base=tm_kwargs.pop("api_base"),
            api_key=tm_kwargs.pop("api_key"),
            cache=False,
            **tm_kwargs,
        )

    backend = _backend_choice()

    if backend == "litellm":
        return StreamingLM(model_str, cache=False, **kwargs)

    if backend == "pi":
        from aiipython.pi_gateway_lm import PiGatewayLM

        return PiGatewayLM(model_str, cache=False, **kwargs)

    # auto
    try:
        from aiipython.pi_gateway_lm import PiGatewayLM

        lm = PiGatewayLM(model_str, cache=False, **kwargs)
        # Ensure gateway is actually reachable before committing.
        lm._client.health()
        return lm
    except Exception:
        return StreamingLM(model_str, cache=False, **kwargs)
