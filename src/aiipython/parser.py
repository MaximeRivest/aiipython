"""Parse fenced code blocks out of markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches ```lang\n...\n``` (with optional language tag)
_FENCE_RE = re.compile(
    r"```(?P<lang>\w*)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)


@dataclass
class CodeBlock:
    lang: str       # e.g. "python", "bash", "" — empty means untagged
    code: str       # the raw code inside the fence


def extract_code_blocks(markdown: str) -> list[CodeBlock]:
    """Return all fenced code blocks found in *markdown*."""
    return [
        CodeBlock(lang=m.group("lang").lower(), code=m.group("code").rstrip())
        for m in _FENCE_RE.finditer(markdown)
    ]


def executable_blocks(markdown: str) -> list[CodeBlock]:
    """Return only the blocks we should run in IPython.

    We run python blocks (including bare ``` with no lang tag) and
    IPython-magic blocks.  Everything else is treated as display-only.
    """
    runnable = {"python", "py", "ipython", ""}
    return [b for b in extract_code_blocks(markdown) if b.lang in runnable]
