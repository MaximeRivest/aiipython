"""Human-managed persistent context items (files, text snippets, clipboard).

These items are stored in the IPython namespace so they survive TUI restarts,
are checkpointed, and can be managed from both UI commands and the REPL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONTEXT_ITEMS_KEY = "context_items"
_CONTEXT_NEXT_ID_KEY = "context_next_id"

_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


def ensure_context_namespace(ns: dict[str, Any]) -> None:
    if _CONTEXT_ITEMS_KEY not in ns or not isinstance(ns[_CONTEXT_ITEMS_KEY], list):
        ns[_CONTEXT_ITEMS_KEY] = []
    if _CONTEXT_NEXT_ID_KEY not in ns or not isinstance(ns[_CONTEXT_NEXT_ID_KEY], int):
        ns[_CONTEXT_NEXT_ID_KEY] = 1


def list_items(ns: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_context_namespace(ns)
    return list(ns[_CONTEXT_ITEMS_KEY])


def add_text_item(
    ns: dict[str, Any],
    text: str,
    *,
    label: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    ensure_context_namespace(ns)
    item_id = f"ctx{ns[_CONTEXT_NEXT_ID_KEY]:03d}"
    ns[_CONTEXT_NEXT_ID_KEY] += 1

    item = {
        "id": item_id,
        "kind": "text",
        "label": label or f"text {item_id}",
        "source": source,
        "text": text,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ns[_CONTEXT_ITEMS_KEY].append(item)
    return item


def add_file_item(
    ns: dict[str, Any],
    path: str | Path,
    *,
    label: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    ensure_context_namespace(ns)
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))

    item_id = f"ctx{ns[_CONTEXT_NEXT_ID_KEY]:03d}"
    ns[_CONTEXT_NEXT_ID_KEY] += 1

    kind = "image" if p.suffix.lower() in _IMAGE_EXTS else "file"
    item = {
        "id": item_id,
        "kind": kind,
        "label": label or p.name,
        "source": source,
        "path": str(p),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ns[_CONTEXT_ITEMS_KEY].append(item)
    return item


def remove_item(ns: dict[str, Any], item_id: str) -> bool:
    ensure_context_namespace(ns)
    items = ns[_CONTEXT_ITEMS_KEY]
    for i, item in enumerate(items):
        if str(item.get("id")) == str(item_id):
            del items[i]
            return True
    return False


def clear_items(ns: dict[str, Any]) -> int:
    ensure_context_namespace(ns)
    n = len(ns[_CONTEXT_ITEMS_KEY])
    ns[_CONTEXT_ITEMS_KEY].clear()
    return n


def render_for_prompt(
    ns: dict[str, Any],
    *,
    max_items: int = 12,
    max_item_chars: int = 2500,
    max_total_chars: int = 12000,
) -> str:
    """Render live context for interpolation into the system prompt."""
    ensure_context_namespace(ns)
    items = ns[_CONTEXT_ITEMS_KEY]
    if not items:
        return ""

    lines: list[str] = ["\n## Pinned Context (human-managed, live)"]
    remaining = max_total_chars

    for item in items[:max_items]:
        if remaining <= 0:
            break

        item_id = item.get("id", "ctx?")
        kind = item.get("kind", "text")
        label = item.get("label", item_id)
        source = item.get("source", "manual")

        lines.append(f"\n### {item_id} [{kind}] {label}")
        lines.append(f"source: {source}")

        body = _resolve_item_body(item, max_chars=min(max_item_chars, remaining))
        lines.append("```text")
        lines.append(body)
        lines.append("```")

        remaining -= len(body)

    extra = len(items) - max_items
    if extra > 0:
        lines.append(f"\n… {extra} more pinned context items not shown (budget cap).")

    return "\n".join(lines)


def summarize_items(items: list[dict[str, Any]], *, preview_chars: int = 80) -> str:
    """Compact markdown summary for /context UI output."""
    if not items:
        return "(none)"

    out: list[str] = []
    for item in items:
        item_id = item.get("id", "ctx?")
        kind = item.get("kind", "text")
        label = item.get("label", "")
        source = item.get("source", "manual")

        if kind in {"file", "image"}:
            path = item.get("path", "")
            preview = path
        else:
            text = str(item.get("text", "")).replace("\n", " ").strip()
            preview = text[:preview_chars] + ("…" if len(text) > preview_chars else "")

        out.append(f"- `{item_id}` [{kind}] **{label}** — {preview} _(source: {source})_")

    return "\n".join(out)


def _resolve_item_body(item: dict[str, Any], *, max_chars: int) -> str:
    kind = item.get("kind")
    if kind in {"file", "image"}:
        path = Path(str(item.get("path", "")))
        return _read_file_live(path, max_chars=max_chars)

    text = str(item.get("text", ""))
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n… [{len(text) - max_chars} more chars]"


def _read_file_live(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return f"<missing file: {path}>"
    if not path.is_file():
        return f"<not a regular file: {path}>"

    try:
        if path.suffix.lower() in _IMAGE_EXTS:
            return _describe_image(path)

        raw = path.read_bytes()
        if b"\x00" in raw[:2048]:
            return f"<binary file: {path} ({len(raw)} bytes)>"

        text = raw.decode("utf-8", errors="replace")
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n… [{len(text) - max_chars} more chars]"
    except Exception as exc:
        return f"<error reading {path}: {type(exc).__name__}: {exc}>"


def _describe_image(path: Path) -> str:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return (
                f"Image file: {path}\n"
                f"size: {img.size[0]}x{img.size[1]}\n"
                f"mode: {img.mode}\n"
                "(Use look_at(images[...]) for visual analysis after loading into images dict.)"
            )
    except Exception:
        return f"Image file: {path}"
