"""LM backend selection for aiipython (pi-native only)."""

from __future__ import annotations

from typing import Any


def create_lm(model_str: str, **kwargs: Any):
    """Create LM for *model_str* using the pi-native gateway backend."""
    from aiipython.pi_native_lm import PiNativeLM
    return PiNativeLM(model_str, cache=False, **kwargs)
