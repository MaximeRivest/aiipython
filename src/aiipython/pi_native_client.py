"""Compatibility wrapper for the pi-native model/auth client."""

from __future__ import annotations

from aiipython.pi_gateway_client import PiGatewayClient


class PiNativeClient(PiGatewayClient):
    """Alias of :class:`PiGatewayClient` under the new pi-native name."""

    pass
