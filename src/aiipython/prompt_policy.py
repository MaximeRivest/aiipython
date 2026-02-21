"""Prompt/context assembly policy constants."""

from __future__ import annotations

# Latest user message inclusion policy.
LATEST_USER_FULL_LIMIT = 12_000
LATEST_USER_HEAD_TAIL = 6_000

# Conversation transcript budget (characters).
TRANSCRIPT_BUDGET = 24_000

# Current reactive-turn execution trace budget (characters).
TURN_TRACE_BUDGET = 18_000

# Per executed block output/code clipping limit used in trace sections.
EXEC_OUTPUT_PER_BLOCK_LIMIT = 4_000

# Clipboard text inline limit for transcript rendering.
CLIPBOARD_INLINE_LIMIT = 4_000
