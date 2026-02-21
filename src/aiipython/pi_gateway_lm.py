"""DSPy LM adapter backed by the local Pi gateway."""

from __future__ import annotations

from typing import Any

import dspy
import litellm

from aiipython.pi_gateway_client import PiGatewayClient


class PiGatewayLM(dspy.LM):
    """Drop-in dspy.LM that delegates completion to the Node Pi gateway."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._client = PiGatewayClient()
        self._streaming_fallback_lm: Any = None

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        kwargs = dict(kwargs)
        kwargs.pop("cache", None)

        messages = messages or [{"role": "user", "content": prompt or ""}]

        if self.use_developer_role and self.model_type == "responses":
            messages = [
                {**m, "role": "developer"} if m.get("role") == "system" else m
                for m in messages
            ]

        merged = {**self.kwargs, **kwargs}
        merged.pop("rollout_id", None)

        # DSPy streamify uses settings.send_stream. The Pi gateway path is
        # currently non-streaming, so fall back to StreamingLM for real-time
        # token deltas when streaming is requested.
        if getattr(dspy.settings, "send_stream", None) is not None:
            if self._streaming_fallback_lm is None:
                from aiipython.streaming_lm import StreamingLM

                self._streaming_fallback_lm = StreamingLM(self.model, **self.kwargs)
            return self._streaming_fallback_lm.forward(
                prompt=prompt,
                messages=messages,
                **merged,
            )

        options: dict[str, Any] = {}
        if "temperature" in merged and merged["temperature"] is not None:
            options["temperature"] = merged["temperature"]

        max_tokens = merged.get("max_output_tokens") or merged.get("max_tokens")
        if isinstance(max_tokens, int):
            options["max_tokens"] = max_tokens

        if "thinking" in merged and merged["thinking"] is not None:
            options["thinking"] = merged["thinking"]

        # Additional provider-specific options can pass through.
        for k, v in merged.items():
            if k in {"temperature", "max_output_tokens", "max_tokens", "thinking"}:
                continue
            if v is None:
                continue
            options[k] = v

        resp = self._client.complete(
            model=self.model,
            messages=messages,
            options=options,
        )

        text = str(resp.get("text", ""))
        usage = resp.get("usage") or {}

        usage_litellm = {
            "prompt_tokens": int(usage.get("input", 0) or 0),
            "completion_tokens": int(usage.get("output", 0) or 0),
            "total_tokens": int(
                usage.get("totalTokens")
                or (int(usage.get("input", 0) or 0) + int(usage.get("output", 0) or 0))
            ),
            "cost": float((usage.get("cost") or {}).get("total", 0) or 0.0),
        }

        stop_reason = str(resp.get("stopReason", "stop") or "stop")
        if stop_reason not in {"stop", "length", "toolUse"}:
            stop_reason = "stop"

        result = litellm.ModelResponse(
            model=self.model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "length" if stop_reason == "length" else "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            usage=usage_litellm,
        )

        self._check_truncation(result)

        if (
            not getattr(result, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(result, "usage")
        ):
            from dspy.dsp.utils.settings import settings as dspy_settings

            dspy_settings.usage_tracker.add_usage(self.model, dict(result.usage))

        return result
