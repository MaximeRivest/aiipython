"""Custom DSPy adapter for aiipython — full control over LM messages.

Uses dspy-template-adapter for exact prompt control. No hidden
rewriting — what you see in the templates below is what the LM gets.

Template elements
─────────────────
  {instruction}           Signature docstring (optimizable by MIPRO/COPRO)
  {session_summary()}     Custom helper: high-level timeline of the session
                          (first 100 chars of each user msg, AI response,
                          and code executed — lives in the system prompt so
                          the model always has context on what's been done)
  {agent_context()}       Custom helper: reads `agent_context` from the
                          namespace.  The AI writes to this variable to
                          persist notes, plans, and context across turns.
  {pinned_context()}      Human-managed context refs (files/text) added via
                          @mentions and /context; file refs are re-read live.
  {environment_state}     The compact namespace + activity log from the kernel
  {active_images()}       Custom helper: renders looked-at images inline
                          (each becomes an image_url content block via DSPy's
                          split_message_content_for_custom_types pipeline)

Parse mode
──────────
  Custom callable `parse_markdown_response` — the entire LM completion
  is returned as the `response` field.  Code block extraction is handled
  separately by aiipython.parser (not the adapter's job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
from dspy.signatures.signature import Signature
from dspy_template_adapter import TemplateAdapter, Predict

if TYPE_CHECKING:
    from aiipython.agent import ReactiveAgent


# ── Templates ───────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
{instruction}

## Response Format

Respond in **markdown**.  To execute code, use fenced python blocks:

```python
# your code here — runs in IPython
```

Multiple code blocks are executed in order.  IPython magic commands are supported:
- `%pip install package` — install packages (via uv)
- `!command` — run shell commands
- `%%bash` — bash cell magic
- `%timeit` — timing

## Available Tools (in the IPython namespace)

- `look_at(images["name"])` or `look_at(pil_img, "label")` — view an image
  (it will be included in your next call so you can see it)
- `spawn_agent(task="...")` — create a sub-agent with its own IPython kernel
- `print()` — inspect variable contents (the snapshot shows types/shapes only)

## Context Gardening

You have full control over your own context.  The variable `agent_context`
is a string that is injected into this system prompt on every call.
Write to it to persist notes, plans, working memory, or anything you want
to remember across turns:

```python
agent_context = \"\"\"
Working on: iris dataset analysis
Key findings: balanced classes (50 each), 4 numeric features
Next steps: user wants a distribution plot
Files loaded: none
\"\"\"
```

This is your scratchpad — curate it, update it, trim it.  It is the
primary way you maintain awareness across turns.

## Your Harness

You are running inside `aiipython`, a reactive agent loop:
1. User message → injected into `user_inputs` / `user_input`
2. Environment snapshot built (namespace shapes + recent activity)
3. This system prompt + snapshot sent to you (single-shot, no chat history)
4. Your markdown response is parsed for ```python blocks
5. Code blocks execute in IPython
6. If code ran → you are re-triggered (you see the results)
7. If no code → conversation pauses until user speaks

The session has automatic checkpointing.  Before each of your turns,
the full namespace is snapshot'd.  The user can navigate the checkpoint
tree to undo, branch, and restore:
  /undo — revert your last turn's effects
  /restore <id> — jump to any checkpoint
  /fork [label] — mark a named branch point
  /tree — view the checkpoint tree

If the user undoes your work and asks you to try again, check
agent_context and the activity log to understand what was tried before.

The session summary, agent_context, and pinned human context live in this
system prompt. The namespace snapshot and recent activity live in the user
message. You never see raw variable contents unless you print() them.

## Behavior

- If you need to act, include ```python blocks.
- When done, respond with just text (no code blocks).
- The user's messages are in `user_inputs` (latest last) and `user_input`.
- Update `agent_context` whenever your working context changes.
- Treat pinned context as user-prioritized and keep it in mind.
{agent_context()}{pinned_context()}{session_summary()}"""

USER_TEMPLATE = """\
{environment_state}{active_images()}"""


# ── Custom parser ───────────────────────────────────────────────────

def parse_markdown_response(
    signature: type[Signature], completion: str
) -> dict[str, Any]:
    """Custom parse function — entire completion is the response.

    Code-block extraction is handled downstream by aiipython.parser,
    not here.  The adapter's only job is to map the raw LM text to
    the `response` output field.
    """
    return {"response": completion.strip()}


# ── Signature ───────────────────────────────────────────────────────

class React(dspy.Signature):
    """You are an AI assistant embedded in a live IPython environment.

You receive a high-level snapshot of the environment: what variables
exist (with types and shapes, not raw contents), a compact execution
log showing who ran what (user / agent / system), and any images you
have asked to look at.  The user's latest message is the last entry
in `user_inputs`.

If you need to inspect a variable's actual contents, write code to
print it.  To persist context across turns, write to `agent_context`."""

    environment_state: str = dspy.InputField(
        desc="Compact snapshot of the IPython environment"
    )
    response: str = dspy.OutputField(
        desc="Markdown response; use ```python fenced blocks for code to execute"
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _trunc(text: str, limit: int = 100) -> str:
    """Truncate to *limit* chars, single line."""
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


# ── Adapter factory ─────────────────────────────────────────────────

def build_adapter(agent: ReactiveAgent) -> TemplateAdapter:
    """Create a TemplateAdapter wired to *agent*'s live state.

    Four custom helpers read from the agent at render time:
      {agent_context()}    — the AI's own scratchpad (system prompt)
      {pinned_context()}   — human-managed persistent refs (system prompt)
      {session_summary()}  — high-level session timeline (system prompt)
      {active_images()}    — multimodal image injection (user message)
    """
    adapter = TemplateAdapter(
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": USER_TEMPLATE},
        ],
        parse_mode=parse_markdown_response,
    )

    # ── helper: {agent_context()} ───────────────────────────────────
    # Reads the `agent_context` variable from the namespace.  The AI
    # writes to it from code; it appears in the system prompt.
    def agent_context_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        ac = agent.kernel.shell.user_ns.get("agent_context", "")
        if not ac or not ac.strip():
            return ""
        return f"\n## Agent Context (your notes)\n{ac.strip()}\n"

    # ── helper: {pinned_context()} ──────────────────────────────────
    def pinned_context_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        return agent.render_pinned_context()

    # ── helper: {session_summary()} ─────────────────────────────────
    def session_summary_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        ns = agent.kernel.shell.user_ns
        user_inputs = ns.get("user_inputs", [])
        ai_responses = ns.get("ai_responses", [])
        history = agent.kernel.history

        if not user_inputs and not ai_responses:
            return ""

        lines: list[str] = ["\n## Session Summary"]

        ui_idx = 0
        ar_idx = 0

        for h in history:
            tag = h.get("tag")
            code = h.get("code", "")

            if tag == "user" and code.startswith("user_input = <message"):
                if ui_idx < len(user_inputs):
                    lines.append(f"  👤 {_trunc(user_inputs[ui_idx])}")
                    ui_idx += 1
            elif tag == "agent" and code.startswith("ai_responses += <response"):
                if ar_idx < len(ai_responses):
                    lines.append(f"  🤖 {_trunc(ai_responses[ar_idx])}")
                    ar_idx += 1
            elif tag == "agent" and not code.startswith("ai_responses"):
                lines.append(f"  ⚡ {_trunc(code)}")

        return "\n".join(lines)

    # ── helper: {active_images()} ───────────────────────────────────
    def active_images_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        images = agent._active_images
        if not images:
            return ""
        parts: list[str] = []
        for label, dspy_img in images:
            parts.append(f"\n[Image: {label}]\n{dspy_img}")
        return "\n".join(parts)

    adapter.register_helper("agent_context", agent_context_helper)
    adapter.register_helper("pinned_context", pinned_context_helper)
    adapter.register_helper("session_summary", session_summary_helper)
    adapter.register_helper("active_images", active_images_helper)

    return adapter


def build_predict(agent: ReactiveAgent) -> Predict:
    """Return a Predict module bound to our adapter + signature."""
    adapter = build_adapter(agent)
    return Predict(React, adapter=adapter)
