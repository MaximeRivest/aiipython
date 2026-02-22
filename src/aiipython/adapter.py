"""Custom DSPy adapter for aiipython — full control over LM messages.

Uses dspy-template-adapter for exact prompt control. No hidden
rewriting — what you see in the templates below is what the LM gets.

Template elements
─────────────────
  {instruction}             Signature docstring (optimizable by MIPRO/COPRO)
  {latest_user_message()}   Guaranteed latest user message (full if small,
                            head+tail when oversized)
  {agent_context()}         Custom helper: reads `agent_context` from the
                            namespace. The AI writes notes/plans here.
  {pinned_context()}        Human-managed context refs (files/text) added via
                            @mentions and /context; file refs are re-read live.
  {recent_transcript()}     Budgeted transcript with user text + assistant prose
                            (non-code markdown) for stronger continuity.
  {environment_state}       Compact namespace + activity log from the kernel
  {active_images()}         Custom helper: renders looked-at images inline
                            (each becomes an image_url content block via DSPy's
                            split_message_content_for_custom_types pipeline)

Parse mode
──────────
  Custom callable `parse_markdown_response` — the entire LM completion
  is returned as the `response` field. Code block extraction is handled
  separately by aiipython.parser (not the adapter's job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
from dspy.signatures.signature import Signature
from dspy_template_adapter import TemplateAdapter, Predict

from aiipython.parser import strip_fenced_code
from aiipython.prompt_policy import (
    CLIPBOARD_INLINE_LIMIT,
    LATEST_USER_FULL_LIMIT,
    LATEST_USER_HEAD_TAIL,
    TRANSCRIPT_BUDGET,
)

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

Because execution happens in IPython, `%%bash` inside an executable
` ```python ` block is a first-class way to run shell workflows.  Use it
confidently whenever bash is the best tool for the job.

Use ` ```#py ` for reference-only code blocks that should **not** execute.

## Available Tools (in the IPython namespace)

- `look_at(images["name"])` or `look_at(pil_img, "label")` — view an image
  (it will be included in your next call so you can see it)
- `spawn_agent(task="...")` — create a subprocess sub-agent proxy
  (use `child.react()` or `child.ask("...")` to continue it statefully)
- `file_fingerprint(path)` — returns `{sha256, line_count, size_bytes}`
- `read_file_lines(path, start_line=1, end_line=None)` — numbered slice with
  `file_sha256` and `range_sha256`
- `edit_file_lines(path, start_line, end_line, new_content, expected_file_sha256=..., expected_range_sha256=...)`
  — precise line-range replacement with optimistic hash guards
- `edit_file(...)` — alias of `edit_file_lines(...)`
- `safe_patch(path, start_line, end_line, new_content, expected_old_text=None, max_retries=1)`
  — convenience wrapper that reads hashes then applies guarded edit (with retry)
- `context_stats()` — reliable pinned-context diagnostics (caps/truncation markers)
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

When `agent_context` is empty, the session goal is not yet defined, and
there is no meaningful pinned context, default to **context-gardening mode**:
ask focused clarifying questions, do lightweight exploration when useful, and
promote durable findings into context (update `agent_context`, and pin key
text/files with `context_add_text(...)` / `context_add_file(...)` when helpful).

## Your Harness

You are running inside `aiipython`, a reactive agent loop:
1. User message → injected into `user_inputs` / `user_input`
2. Environment snapshot built (namespace shapes + recent activity)
3. This system prompt + snapshot sent to you (stateless single-shot; no raw chat transcript)
4. Your markdown response is parsed for ```python blocks
5. Code blocks execute in IPython
6. If code ran → you are re-triggered (you see the results)
7. If no code → conversation pauses until user speaks

All markdown is streamed to the user before code runs.  Narrate briefly what
you're about to do, then run code. In multi-step chains, summarize what just
happened before executing the next block.

The session has automatic checkpointing.  Before each of your turns,
the full namespace is snapshot'd.  The user can navigate the checkpoint
tree to undo, branch, and restore:
  /undo — revert your last turn's effects
  /restore <id> — jump to any checkpoint
  /fork [label] — mark a named branch point
  /tree — view the checkpoint tree

If the user undoes your work and asks you to try again, check
agent_context and the activity log to understand what was tried before.

Large pasted/clipboard payloads may be clipped for context budget.  If a
payload appears clipped, inspect it strategically (head/tail/search/sampling)
instead of dumping everything at once.

## Pi Frontend (still active)

Your UI is Pi `InteractiveMode` (frontend), even though backend execution is
aiipython:
- Pi editor UX still applies: `@` file refs, path completion, image paste,
  queueing (`Enter` steer / `Alt+Enter` follow-up), and abort (`Escape`).
- Pi slash commands are handled by the frontend before model calls
  (for example `/model`, `/settings`, `/tree`, `/resume`, `/new`, `/hotkeys`).
- `!cmd` / `!!cmd` entered by the user are handled by the frontend bridge and
  executed against the bound IPython session.
- Pi's default built-in coding tools (`read`, `write`, `edit`, `bash`, etc.)
  are not your primary interface here; prefer aiipython namespace helpers and
  Python execution as documented above.

## Edit Protocol (follow this for code/file changes)

For non-trivial edits, use this exact sequence:
1. `file_fingerprint(path)`
2. `read_file_lines(path, start, end)` to confirm target lines
3. `edit_file_lines(...)` with `expected_file_sha256` + `expected_range_sha256`
4. If hashes drift, re-read and patch again (or use `safe_patch(...)`).

Prefer line-range edits over whole-file rewrites. Avoid large regex/string
surgery unless absolutely necessary.
Do not attempt giant multiline rewrites with one huge Python string literal;
use `read_file_lines(...)` + `edit_file_lines(...)`/`safe_patch(...)`.
When checking pinned-context truncation, use `context_stats()` instead of naive
substring searches (those can match text inside pinned source files).

## Behavior

- If you need to act, include ```python blocks.
- Be eager and action-oriented: do not be shy about producing runnable code
  blocks when execution is the best path to help the user.
- For actions that are clearly low-risk, easily reversible, or non-mutating,
  do not ask for permission first — just do them.
- Context updates are always allowed: changing `agent_context` and doing
  context-gardening work (including proactive pin management) does **not** count
  as risky change; do it freely and proactively whenever useful.
- Do not hesitate to define small helper functions/utilities in Python when
  that makes progress faster or more reliable for the current goal.
- When done, respond with just text (no code blocks).
- The user's messages are in `user_inputs` (latest last) and `user_input`.
- Update `agent_context` whenever your working context changes.
- Treat pinned context as user-prioritized and keep it in mind.
- Prefer targeted file edits: read `read_file_lines(...)`, then edit with
  `edit_file_lines(...)` using hash guards instead of brittle whole-file rewrites.
{latest_user_message()}{agent_context()}{pinned_context()}{recent_transcript()}"""

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

def _head_tail(text: str, *, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n"
        f"\n… [omitted {omitted} chars] …\n\n"
        f"{text[-tail:]}"
    )


# ── Adapter factory ─────────────────────────────────────────────────

def build_adapter(agent: ReactiveAgent) -> TemplateAdapter:
    """Create a TemplateAdapter wired to *agent*'s live state.

    Five custom helpers read from the agent at render time:
      {latest_user_message()} — latest user message (system prompt)
      {agent_context()}       — AI scratchpad (system prompt)
      {pinned_context()}      — human-managed persistent refs (system prompt)
      {recent_transcript()}   — budgeted transcript (system prompt)
      {active_images()}       — multimodal image injection (user message)
    """
    adapter = TemplateAdapter(
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": USER_TEMPLATE},
        ],
        parse_mode=parse_markdown_response,
    )

    # ── helper: {latest_user_message()} ────────────────────────────
    def latest_user_message_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        ns = agent.kernel.shell.user_ns
        latest = ns.get("user_input") or ""
        if not latest:
            return ""

        source = "chat"
        is_clipboard = False
        for entry in reversed(ns.get("conversation_log", [])):
            if entry.get("role") == "user":
                source = str(entry.get("source", "chat"))
                is_clipboard = bool(entry.get("is_clipboard", False))
                break

        if len(latest) <= LATEST_USER_FULL_LIMIT:
            body = latest
        else:
            body = _head_tail(
                latest,
                head=LATEST_USER_HEAD_TAIL,
                tail=LATEST_USER_HEAD_TAIL,
            )

        return (
            "\n## Latest User Message\n"
            f"source: {source}{' (clipboard)' if is_clipboard else ''}\n"
            "```text\n"
            f"{body}\n"
            "```\n"
        )

    # ── helper: {agent_context()} ───────────────────────────────────
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

    # ── helper: {recent_transcript()} ───────────────────────────────
    def recent_transcript_helper(
        ctx: dict, signature: type[Signature], demos: list, **kwargs
    ) -> str:
        ns = agent.kernel.shell.user_ns
        conv = ns.get("conversation_log", [])
        if not conv:
            return ""

        rendered: list[str] = []
        remaining = TRANSCRIPT_BUDGET
        omitted = 0

        for entry in reversed(conv):
            role = entry.get("role")
            title = ""
            body = ""

            if role == "user":
                source = str(entry.get("source", "chat"))
                is_clipboard = bool(entry.get("is_clipboard", False))
                body = str(entry.get("text", ""))
                if not body.strip():
                    continue
                if is_clipboard and len(body) > CLIPBOARD_INLINE_LIMIT:
                    half = CLIPBOARD_INLINE_LIMIT // 2
                    body = _head_tail(body, head=half, tail=half)
                title = f"👤 user ({source}{', clipboard' if is_clipboard else ''})"

            elif role == "assistant":
                prose = str(entry.get("prose", "") or "").strip()
                if not prose:
                    prose = strip_fenced_code(str(entry.get("markdown", "")))
                body = prose.strip()
                if not body:
                    continue
                title = "🤖 assistant prose"

            else:
                continue

            chunk = (
                f"\n### {title}\n"
                "```text\n"
                f"{body}\n"
                "```"
            )

            if len(chunk) > remaining:
                omitted += 1
                continue

            rendered.append(chunk)
            remaining -= len(chunk)

        if not rendered:
            return ""

        rendered.reverse()

        lines = ["\n## Recent Transcript (budgeted)"]
        lines.extend(rendered)
        if omitted > 0:
            lines.append(f"\n… {omitted} older transcript entries omitted (budget cap).")
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

    adapter.register_helper("latest_user_message", latest_user_message_helper)
    adapter.register_helper("agent_context", agent_context_helper)
    adapter.register_helper("pinned_context", pinned_context_helper)
    adapter.register_helper("recent_transcript", recent_transcript_helper)
    adapter.register_helper("active_images", active_images_helper)

    return adapter


def build_predict(agent: ReactiveAgent) -> Predict:
    """Return a Predict module bound to our adapter + signature."""
    adapter = build_adapter(agent)
    return Predict(React, adapter=adapter)
