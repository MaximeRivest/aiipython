"""Streaming LM backends.

Default path uses LiteLLM chat-completions streaming.
For ``openai-codex/*`` models we use a custom transport compatible with
Pi/Codex OAuth subscription tokens (chatgpt.com backend API).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import dspy
import httpx
import litellm

from aiipython.auth import get_auth_manager

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        data = base64.urlsafe_b64decode(payload)
        obj = json.loads(data.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _codex_account_id(token: str, stored_cred: dict[str, Any] | None = None) -> str | None:
    if stored_cred:
        account = stored_cred.get("accountId") or stored_cred.get("account_id")
        if isinstance(account, str) and account:
            return account

    payload = _jwt_payload(token)
    if not payload:
        return None
    auth = payload.get(_CODEX_JWT_CLAIM_PATH)
    if isinstance(auth, dict):
        account = auth.get("chatgpt_account_id")
        if isinstance(account, str) and account:
            return account
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    out.append("[image]")
                else:
                    out.append(str(block))
            else:
                out.append(str(block))
        return "\n".join(out).strip()
    return str(content)


def _to_codex_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    out: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                out.append({"type": "input_text", "text": str(block)})
                continue

            btype = block.get("type")
            if btype == "text":
                out.append({"type": "input_text", "text": str(block.get("text", ""))})
            elif btype == "image_url":
                image_url = block.get("image_url", {}).get("url")
                if isinstance(image_url, str) and image_url:
                    out.append({"type": "input_image", "image_url": image_url})
            else:
                out.append({"type": "input_text", "text": str(block)})

    if not out:
        out.append({"type": "input_text", "text": _content_text(content)})
    return out


class StreamingLM(dspy.LM):
    """Drop-in dspy.LM with streaming support + Codex transport."""

    def _forward_openai_codex(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ):
        auth = get_auth_manager()
        source = auth.resolve_api_key("openai-codex")
        if not source:
            raise RuntimeError("No OpenAI Codex OAuth credentials. Use `/login openai`.")

        token = source.key
        cred = auth.get("openai-codex") or {}
        account_id = _codex_account_id(token, stored_cred=cred)
        if not account_id:
            raise RuntimeError("OpenAI Codex token missing account id")

        model_id = self.model.split("/", 1)[1] if "/" in self.model else self.model

        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")

            if role in {"system", "developer"}:
                text = _content_text(content).strip()
                if text:
                    instructions_parts.append(text)
                continue

            mapped_role = "assistant" if role == "assistant" else "user"
            input_items.append({
                "role": mapped_role,
                "content": _to_codex_content(content),
            })

        instructions = "\n\n".join(p for p in instructions_parts if p.strip())
        if not instructions:
            instructions = "You are a helpful coding assistant."

        if not input_items:
            input_items = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            ]

        body: dict[str, Any] = {
            "model": model_id,
            "store": False,
            "stream": True,
            "instructions": instructions,
            "input": input_items,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            body["temperature"] = kwargs["temperature"]

        max_tokens = kwargs.get("max_output_tokens") or kwargs.get("max_tokens")
        if isinstance(max_tokens, int):
            body["max_output_tokens"] = max_tokens

        if kwargs.get("tools") is not None:
            body["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            body["tool_choice"] = kwargs["tool_choice"]

        headers = {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "aiipython",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

        full_text = ""
        usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        finish_reason = "stop"

        with httpx.stream(
            "POST",
            _CODEX_BASE_URL,
            headers=headers,
            json=body,
            timeout=90,
        ) as resp:
            if resp.status_code != 200:
                try:
                    error_text = resp.read().decode("utf-8", errors="replace")
                except Exception:
                    error_text = ""
                raise RuntimeError(
                    f"OpenAI Codex request failed ({resp.status_code}): {error_text}"
                )

            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    event = json.loads(data)
                except Exception:
                    continue

                etype = event.get("type")
                if etype == "error":
                    msg = event.get("message") or event.get("code") or str(event)
                    raise RuntimeError(f"Codex error: {msg}")

                if etype == "response.output_text.delta":
                    full_text += str(event.get("delta", ""))
                    continue

                if etype in {"response.completed", "response.done"}:
                    response_obj = event.get("response") or {}
                    status = str(response_obj.get("status", "completed"))
                    if status in {"incomplete", "failed", "cancelled"}:
                        finish_reason = "length" if status == "incomplete" else "stop"

                    usage_obj = response_obj.get("usage") or {}
                    input_tokens = int(usage_obj.get("input_tokens", 0) or 0)
                    output_tokens = int(usage_obj.get("output_tokens", 0) or 0)
                    total_tokens = int(usage_obj.get("total_tokens", input_tokens + output_tokens) or 0)
                    usage = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }

                    if not full_text:
                        outputs = response_obj.get("output") or []
                        texts: list[str] = []
                        for item in outputs:
                            if not isinstance(item, dict) or item.get("type") != "message":
                                continue
                            for part in item.get("content") or []:
                                if isinstance(part, dict) and part.get("type") == "output_text":
                                    texts.append(str(part.get("text", "")))
                        full_text = "".join(texts)

        result = litellm.ModelResponse(
            model=self.model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": full_text},
                }
            ],
            usage=usage,
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

        # Pi/Codex transport (subscription OAuth over chatgpt.com backend).
        if self.model.startswith("openai-codex/"):
            return self._forward_openai_codex(messages=messages, kwargs=kwargs)

        # TabMinion custom params (OpenAI-compatible extra body fields)
        extra_body = dict(kwargs.get("extra_body") or {})
        if "conversation_mode" in kwargs:
            extra_body["conversation_mode"] = kwargs.pop("conversation_mode")
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Force streaming — litellm CustomLogger callbacks fire per-chunk
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            num_retries=self.num_retries,
            cache={"no-cache": True, "no-store": True},
            **kwargs,
        )

        chunks = []
        for chunk in response:
            chunks.append(chunk)

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
