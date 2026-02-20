"""Streaming LM — forces litellm streaming so wire callbacks fire per-chunk.

By default, DSPy calls litellm.completion() without stream=True, so the
wire panel only sees the request and the final response — nothing in between.

This subclass overrides forward() to call litellm.completion(stream=True).
LiteLLM's CustomLogger callbacks fire in real time:
  - log_pre_api_call   → wire panel shows request immediately
  - log_stream_event   → wire panel shows each chunk as it arrives
  - log_success_event  → wire panel shows timing and usage

The complete response is assembled via stream_chunk_builder and returned
to DSPy as a normal ModelResponse — completely transparent to the caller.
"""

from __future__ import annotations

from typing import Any

import dspy
import litellm


class StreamingLM(dspy.LM):
    """Drop-in dspy.LM that forces streaming through litellm."""

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.pop("cache", None)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [
                {**m, "role": "developer"} if m.get("role") == "system" else m
                for m in messages
            ]
        kwargs = {**self.kwargs, **kwargs}
        kwargs.pop("rollout_id", None)

        # Force streaming — litellm CustomLogger callbacks fire per-chunk
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            num_retries=self.num_retries,
            cache={"no-cache": True, "no-store": True},
            **kwargs,
        )

        # Iterate chunks — litellm fires log_stream_event for each one,
        # then log_success_event when the stream completes.
        chunks = []
        for chunk in response:
            chunks.append(chunk)

        # Assemble into a standard ModelResponse for DSPy
        result = litellm.stream_chunk_builder(chunks)

        self._check_truncation(result)

        if (
            not getattr(result, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(result, "usage")
        ):
            from dspy.dsp.utils.settings import settings as dspy_settings

            dspy_settings.usage_tracker.add_usage(self.model, dict(result.usage))

        return result
