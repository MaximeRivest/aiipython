"""Parse fenced code blocks out of markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches ```lang\n...\n``` (with optional language tag)
# Language tags are permissive so markers like "#py" are supported.
_FENCE_RE = re.compile(
    r"```(?P<lang>[^\s`]*)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)


@dataclass
class CodeBlock:
    lang: str       # e.g. "python", "bash", "", "#py"
    code: str       # the raw code inside the fence


def extract_code_blocks(markdown: str) -> list[CodeBlock]:
    """Return all fenced code blocks found in *markdown*."""
    return [
        CodeBlock(lang=m.group("lang").lower(), code=m.group("code").rstrip())
        for m in _FENCE_RE.finditer(markdown)
    ]


def is_executable_lang(lang: str) -> bool:
    """Whether a fenced language should be executed in IPython."""
    lang = (lang or "").lower()
    if lang.startswith("#"):
        return False
    runnable = {"python", "py", "ipython", ""}
    return lang in runnable


def executable_blocks(markdown: str) -> list[CodeBlock]:
    """Return only the blocks we should run in IPython.

    We run python blocks (including bare ``` with no lang tag) and
    IPython-magic blocks.  Everything else is treated as display-only.
    """
    return [b for b in extract_code_blocks(markdown) if is_executable_lang(b.lang)]


def strip_fenced_code(markdown: str) -> str:
    """Return markdown with all fenced code blocks removed."""
    return _FENCE_RE.sub("", markdown).strip()


def strip_executable_fenced_code(markdown: str) -> str:
    """Return markdown with runnable fenced blocks removed.

    Non-runnable blocks (e.g. ```#py) are preserved for display.
    """

    def _replace(match: re.Match[str]) -> str:
        lang = (match.group("lang") or "").lower()
        if is_executable_lang(lang):
            return ""
        return match.group(0)

    return _FENCE_RE.sub(_replace, markdown).strip()
