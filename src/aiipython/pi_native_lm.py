"""Compatibility wrapper for the pi-native DSPy LM."""

from __future__ import annotations

from aiipython.pi_gateway_lm import PiGatewayLM


class PiNativeLM(PiGatewayLM):
    """Alias of :class:`PiGatewayLM` under the new pi-native name."""

    pass
