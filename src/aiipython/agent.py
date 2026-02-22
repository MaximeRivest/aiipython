"""DSPy reactive agent — single-shot, fully stateful system prompt."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dspy

from aiipython.checkpoint import clone_state
from aiipython.context import (
    add_file_item,
    add_text_item,
    clear_items,
    ensure_context_namespace,
    list_items,
    remove_item,
    render_for_prompt,
)
from aiipython.kernel import Kernel
from aiipython.parser import CodeBlock, executable_blocks, strip_fenced_code
from aiipython.prompt_policy import (
    TURN_TRACE_BUDGET,
    EXEC_OUTPUT_PER_BLOCK_LIMIT,
)


@dataclass
class ReactionStep:
    """One agent iteration: markdown + executed blocks + their results."""

    iteration: int
    markdown: str
    blocks: list[CodeBlock]
    executions: list[dict[str, Any]]
    is_final: bool


class PromptAborted(RuntimeError):
    """Raised when a reactive turn is cancelled by the user."""


class ReactiveAgent:
    """Reacts to state changes in the kernel.

    No conversation history is passed to the LM — the entire context is
    baked into a compact environment snapshot.  Uses our custom
    TemplateAdapter (aiipython.adapter) so the exact messages sent to the
    LM are fully specified — no hidden rewriting.
    """

    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel

        # Accumulated images for the session — never cleared automatically.
        # The adapter reads this list at render time via {active_images()}.
        self._active_images: list[tuple[str, dspy.Image]] = []

        # Build the adapter + predict module (must come after _active_images
        # exists because the adapter captures a reference to it)
        from aiipython.adapter import build_predict
        self.predict = build_predict(self)

        # Expose helpers inside the kernel so the AI's code can use them
        ensure_context_namespace(self.kernel.shell.user_ns)
        self.kernel.push(
            agent=self,
            spawn_agent=self.spawn,
            look_at=self.look_at,
            file_fingerprint=self.file_fingerprint,
            read_file_lines=self.read_file_lines,
            edit_file_lines=self.edit_file_lines,
            edit_file=self.edit_file_lines,
            safe_patch=self.safe_patch,
            context_stats=self.context_stats,
            context_add=self.context_add,
            context_add_file=self.context_add_file,
            context_add_text=self.context_add_text,
            context_remove=self.context_remove,
            context_clear=self.context_clear,
            context_list=self.context_list,
        )

    # ── image management ────────────────────────────────────────────
    def look_at(self, image: Any, label: str | None = None) -> str:
        """Register an image so the LM can see it in all future calls.

        Call from IPython code the AI writes:
            look_at(images["photo"])
            look_at(images["chart"], "sales chart Q4")
            look_at(some_pil_var, "screenshot")

        Accepts:
          - A PIL Image directly
          - A string key into the `images` dict
        """
        if isinstance(image, str):
            images_dict = self.kernel.shell.user_ns.get("images", {})
            if image not in images_dict:
                return f"error: no image named '{image}' in images dict"
            label = label or image
            image = images_dict[image]

        if label is None:
            label = f"image_{len(self._active_images)}"

        dspy_img = dspy.Image(url=image)
        self._active_images.append((label, dspy_img))

        summary = (
            f"👁 Now looking at \"{label}\" "
            f"({len(self._active_images)} active images)"
        )
        print(summary)
        return summary

    @property
    def active_image_labels(self) -> list[str]:
        return [label for label, _ in self._active_images]

    # ── file inspection / editing helpers ───────────────────────────
    def _resolve_file_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p)
        p = p.resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        return p

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def file_fingerprint(self, path: str) -> dict[str, Any]:
        """Return stable file metadata for safe line-based editing."""
        p = self._resolve_file_path(path)
        raw = p.read_bytes()
        if b"\x00" in raw[:2048]:
            raise ValueError(f"Binary file not supported: {p}")

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        return {
            "path": str(p),
            "size_bytes": len(raw),
            "line_count": len(lines),
            "sha256": self._sha256_text(text),
        }

    def read_file_lines(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        *,
        max_chars: int = 12_000,
    ) -> str:
        """Read a numbered line range with file/range hashes for precise edits."""
        p = self._resolve_file_path(path)
        raw = p.read_bytes()
        if b"\x00" in raw[:2048]:
            raise ValueError(f"Binary file not supported: {p}")

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        line_count = len(lines)
        if line_count == 0:
            return f"# file: {p}\n# sha256: {self._sha256_text(text)}\n# lines: 0\n"

        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line is not None else min(line_count, start + 199)
        start = min(start, line_count)
        end = max(start, min(end, line_count))

        segment_lines = lines[start - 1 : end]
        segment = "".join(segment_lines)
        file_sha = self._sha256_text(text)
        segment_sha = self._sha256_text(segment)

        rendered = [
            f"# file: {p}",
            f"# file_sha256: {file_sha}",
            f"# line_range: {start}-{end} / {line_count}",
            f"# range_sha256: {segment_sha}",
        ]
        for idx, line in enumerate(segment_lines, start=start):
            rendered.append(f"{idx:5d} | {line.rstrip(chr(10)).rstrip(chr(13))}")

        out = "\n".join(rendered)
        if len(out) <= max_chars:
            return out
        clipped = out[:max_chars]
        omitted = len(out) - max_chars
        return f"{clipped}\n… [{omitted} more chars clipped]"

    def edit_file_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
        *,
        expected_file_sha256: str | None = None,
        expected_range_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Replace a line range (inclusive) with optional hash guards."""
        p = self._resolve_file_path(path)
        raw = p.read_bytes()
        if b"\x00" in raw[:2048]:
            raise ValueError(f"Binary file not supported: {p}")

        old_text = raw.decode("utf-8", errors="replace")
        old_lines = old_text.splitlines(keepends=True)
        line_count = len(old_lines)

        if line_count == 0:
            raise ValueError("Cannot edit an empty file by line range; write the file directly.")

        start = int(start_line)
        end = int(end_line)
        if start < 1 or end < start or end > line_count:
            raise ValueError(f"Invalid line range {start}-{end}; file has {line_count} line(s).")

        old_file_sha = self._sha256_text(old_text)
        old_segment = "".join(old_lines[start - 1 : end])
        old_segment_sha = self._sha256_text(old_segment)

        if expected_file_sha256 and expected_file_sha256 != old_file_sha:
            raise ValueError(
                "File hash mismatch; file changed since it was inspected. "
                f"expected={expected_file_sha256} actual={old_file_sha}"
            )
        if expected_range_sha256 and expected_range_sha256 != old_segment_sha:
            raise ValueError(
                "Range hash mismatch; targeted lines changed. "
                f"expected={expected_range_sha256} actual={old_segment_sha}"
            )

        replacement = str(new_content or "")
        if replacement and old_segment.endswith("\n") and not replacement.endswith("\n"):
            replacement += "\n"
        replacement_lines = replacement.splitlines(keepends=True)

        new_lines = old_lines[: start - 1] + replacement_lines + old_lines[end:]
        new_text = "".join(new_lines)
        p.write_text(new_text, encoding="utf-8")

        return {
            "path": str(p),
            "replaced_range": f"{start}-{end}",
            "old_file_sha256": old_file_sha,
            "new_file_sha256": self._sha256_text(new_text),
            "old_range_sha256": old_segment_sha,
            "new_range_sha256": self._sha256_text(replacement),
            "old_line_count": line_count,
            "new_line_count": len(new_lines),
        }

    def _parse_read_lines_meta(self, rendered: str) -> dict[str, Any]:
        file_sha = ""
        range_sha = ""
        start = end = total = 0

        for line in rendered.splitlines():
            if line.startswith("# file_sha256:"):
                file_sha = line.split(":", 1)[1].strip()
            elif line.startswith("# range_sha256:"):
                range_sha = line.split(":", 1)[1].strip()
            elif line.startswith("# line_range:"):
                # format: "# line_range: <start>-<end> / <total>"
                payload = line.split(":", 1)[1].strip()
                left, _, right = payload.partition("/")
                left = left.strip()
                right = right.strip()
                if "-" in left:
                    s_raw, e_raw = left.split("-", 1)
                    start = int(s_raw.strip())
                    end = int(e_raw.strip())
                if right:
                    total = int(right)

        if not file_sha or not range_sha or start <= 0 or end < start:
            raise ValueError("Could not parse read_file_lines metadata for safe_patch")

        return {
            "file_sha256": file_sha,
            "range_sha256": range_sha,
            "start_line": start,
            "end_line": end,
            "line_count": total,
        }

    def _read_exact_lines(self, path: str, start_line: int, end_line: int) -> str:
        p = self._resolve_file_path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError(f"Invalid line range {start_line}-{end_line}")
        return "".join(lines[start_line - 1 : end_line])

    def _locate_text_line_range(self, path: str, expected_old_text: str) -> tuple[int, int] | None:
        needle = str(expected_old_text or "")
        if not needle:
            return None

        p = self._resolve_file_path(path)
        haystack = p.read_text(encoding="utf-8", errors="replace")

        idx = haystack.find(needle)
        if idx < 0:
            return None
        if haystack.find(needle, idx + 1) >= 0:
            # Ambiguous occurrence; don't guess.
            return None

        start_line = haystack.count("\n", 0, idx) + 1
        line_span = needle.count("\n")
        if not needle.endswith("\n"):
            line_span += 1
        end_line = start_line + max(1, line_span) - 1
        return (start_line, end_line)

    def safe_patch(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
        *,
        expected_old_text: str | None = None,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Convenience wrapper: fingerprint/read/hash-guarded line patch with retry.

        Flow per attempt:
          1) read_file_lines(path, start, end) -> parse hashes
          2) edit_file_lines(..., expected_file_sha256=..., expected_range_sha256=...)
          3) on mismatch, optionally relocate by expected_old_text and retry
        """
        target_start = int(start_line)
        target_end = int(end_line)
        attempts = max(0, int(max_retries)) + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                rendered = self.read_file_lines(path, target_start, target_end, max_chars=120_000)
                meta = self._parse_read_lines_meta(rendered)

                if expected_old_text is not None:
                    current_segment = self._read_exact_lines(path, meta["start_line"], meta["end_line"])
                    if current_segment != expected_old_text:
                        relocated = self._locate_text_line_range(path, expected_old_text)
                        if relocated is None:
                            raise ValueError(
                                "safe_patch: expected_old_text no longer matches target range "
                                "and could not be uniquely relocated"
                            )
                        target_start, target_end = relocated
                        rendered = self.read_file_lines(path, target_start, target_end, max_chars=120_000)
                        meta = self._parse_read_lines_meta(rendered)

                result = self.edit_file_lines(
                    path,
                    meta["start_line"],
                    meta["end_line"],
                    new_content,
                    expected_file_sha256=meta["file_sha256"],
                    expected_range_sha256=meta["range_sha256"],
                )
                result["safe_patch_attempts"] = attempt
                result["safe_patch_requested_range"] = f"{start_line}-{end_line}"
                result["safe_patch_applied_range"] = result.get("replaced_range")
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                if expected_old_text is not None:
                    relocated = self._locate_text_line_range(path, expected_old_text)
                    if relocated is not None:
                        target_start, target_end = relocated

        if last_error is not None:
            raise last_error
        raise RuntimeError("safe_patch failed")

    # ── pinned context (human-managed) ──────────────────────────────
    def context_list(self) -> list[dict[str, Any]]:
        return list_items(self.kernel.shell.user_ns)

    def context_add_text(
        self, text: str, label: str | None = None, source: str = "manual"
    ) -> dict[str, Any]:
        return add_text_item(
            self.kernel.shell.user_ns,
            text,
            label=label,
            source=source,
        )

    def context_add_file(
        self, path: str, label: str | None = None, source: str = "manual"
    ) -> dict[str, Any]:
        return add_file_item(
            self.kernel.shell.user_ns,
            path,
            label=label,
            source=source,
        )

    def context_add(self, value: str, label: str | None = None) -> str:
        """Add text or file path to pinned context (REPL helper)."""
        from pathlib import Path

        p = Path(value).expanduser()
        if p.is_file():
            item = self.context_add_file(str(p), label=label, source="repl")
            return f"pinned {item['id']} [file] {item['label']}"

        item = self.context_add_text(value, label=label, source="repl")
        return f"pinned {item['id']} [text] {item['label']}"

    def context_remove(self, item_id: str) -> bool:
        return remove_item(self.kernel.shell.user_ns, item_id)

    def context_clear(self) -> int:
        return clear_items(self.kernel.shell.user_ns)

    def render_pinned_context(self) -> str:
        return render_for_prompt(self.kernel.shell.user_ns)

    def context_stats(self) -> dict[str, Any]:
        """Return reliable pinned-context diagnostics (avoids false-positive substring checks)."""
        rendered = self.render_pinned_context()
        lines = rendered.splitlines()

        shown_item_headers = sum(1 for line in lines if line.startswith("### ctx"))
        has_top_level_item_cap = any(
            line.startswith("… ") and "more pinned context items not shown" in line
            for line in lines
        )
        top_level_line_truncation_count = sum(
            1 for line in lines if line.startswith("… [truncated ")
        )

        return {
            "total_pinned_items": len(self.context_list()),
            "rendered_chars": len(rendered),
            "shown_item_headers": shown_item_headers,
            "item_cap_reached": has_top_level_item_cap,
            "line_truncation_markers": top_level_line_truncation_count,
        }

    # ── build the compact environment state ─────────────────────────
    def build_environment_state(
        self,
        *,
        turn_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        snap = self.kernel.snapshot()
        history = self.kernel.history

        sections: list[str] = []

        # ── namespace overview ──────────────────────────────────────
        if snap:
            var_lines = "\n".join(f"  {k}: {v}" for k, v in snap.items())
            sections.append(f"## Namespace\n{var_lines}")

        # ── images in dict ──────────────────────────────────────────
        images = self.kernel.shell.user_ns.get("images", {})
        if images:
            img_lines = []
            for name, im in images.items():
                size = getattr(im, "size", "?")
                mode = getattr(im, "mode", "?")
                looked = "👁" if name in self.active_image_labels else "  "
                img_lines.append(
                    f"  {looked} \"{name}\": {size[0]}×{size[1]}, {mode}"
                )
            sections.append("## Images\n" + "\n".join(img_lines))

        # ── active images (what the LM can currently see) ───────────
        if self._active_images:
            labels = ", ".join(f'"{l}"' for l, _ in self._active_images)
            sections.append(
                f"## Active Images (you can see these)\n"
                f"  {len(self._active_images)} images: [{labels}]\n"
                f"  Use look_at() to add more."
            )

        # ── human-pinned context refs ───────────────────────────────
        ctx_items = self.context_list()
        if ctx_items:
            ctx_lines = []
            for item in ctx_items[-12:]:
                kind = item.get("kind", "text")
                label = item.get("label", item.get("id", "ctx?"))
                item_id = item.get("id", "ctx?")
                if kind in {"file", "image"}:
                    src = item.get("path", "")
                else:
                    txt = str(item.get("text", "")).replace("\n", " ").strip()
                    src = txt[:60] + ("…" if len(txt) > 60 else "")
                ctx_lines.append(f"  [{item_id}] {kind} {label} — {src}")
            sections.append("## Pinned Context Refs\n" + "\n".join(ctx_lines))

        # ── current reactive-turn trace ─────────────────────────────
        if turn_trace:
            lines: list[str] = [
                "## Current Turn Trace (code/results this reactive turn)",
            ]
            remaining = TURN_TRACE_BUDGET

            for idx, t in enumerate(turn_trace, start=1):
                if remaining <= 0:
                    break

                code = self._clip_text(
                    t.get("code", ""),
                    limit=min(EXEC_OUTPUT_PER_BLOCK_LIMIT, remaining),
                )
                lines.append(f"\n### step {idx} (iteration {t.get('iteration', '?')})")
                lines.append("```py")
                lines.append(code)
                lines.append("```")
                remaining -= len(code)

                for label, key in [
                    ("stdout", "stdout"),
                    ("stderr", "stderr"),
                    ("result", "result"),
                    ("error", "error"),
                ]:
                    val = str(t.get(key, "") or "").strip()
                    if not val or remaining <= 0:
                        continue
                    clipped = self._clip_text(
                        val,
                        limit=min(EXEC_OUTPUT_PER_BLOCK_LIMIT, remaining),
                    )
                    lines.append(f"{label}:")
                    lines.append("```text")
                    lines.append(clipped)
                    lines.append("```")
                    remaining -= len(clipped)

                if t.get("truncated"):
                    lines.append(
                        "⚠ output/code was truncated for context budget. "
                        "If needed, inspect targeted slices with print()."
                    )

            if remaining <= 0 and len(turn_trace) > 0:
                lines.append(
                    "\n… older steps omitted due to turn-trace budget."
                )

            sections.append("\n".join(lines))

        # ── compact activity log ────────────────────────────────────
        # Only metadata: what happened and whether it worked.
        # Large outputs are never shown — the model must print() to
        # inspect.  Short outputs (≤80 chars) are shown inline.
        meaningful = [
            h for h in history if h.get("tag") in ("user", "agent")
        ][-20:]

        if meaningful:
            log_lines: list[str] = []
            for h in meaningful:
                tag = h.get("tag", "system")
                line = f"  [{tag}] {h['code']}"
                # Show the short preview if one was recorded
                if h.get("summary"):
                    line += f"\n    \"{h['summary']}\""
                # Execution output — only if short
                stdout = h.get("stdout", "").strip()
                stderr = h.get("stderr", "").strip()
                result = h.get("result", "")
                if stdout:
                    if len(stdout) <= 80:
                        line += f"\n    → {stdout}"
                    else:
                        line += f"\n    → ({len(stdout)} chars, use print() to see)"
                if stderr:
                    if len(stderr) <= 80:
                        line += f"\n    ⚠ {stderr}"
                    else:
                        line += f"\n    ⚠ ({len(stderr)} chars of stderr)"
                if result:
                    if len(result) <= 80:
                        line += f"\n    = {result}"
                    else:
                        line += f"\n    = ({len(result)} chars)"
                if not h.get("success"):
                    error_msg = h.get("error", "")
                    if error_msg:
                        line += f"\n    ✗ {error_msg}"
                    else:
                        line += "\n    ✗ FAILED"
                log_lines.append(line)
            sections.append(
                "## Recent Activity (oldest → newest)\n"
                + "\n".join(log_lines)
            )

        return "\n\n".join(sections)

    @staticmethod
    def _clip_text(text: str, *, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n… [{len(text) - limit} more chars]"

    def _predict_response_markdown(
        self,
        *,
        environment_state: str,
        on_stream_chunk: Callable[[str, bool], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> str:
        """Run one Predict call and return markdown, optionally streaming chunks."""
        if should_abort and should_abort():
            raise PromptAborted("Prompt aborted")

        if on_stream_chunk is None:
            prediction = self.predict(environment_state=environment_state)
            return str(getattr(prediction, "response", "") or "")

        def _extract_delta_text(item: Any) -> str:
            if isinstance(item, bytes):
                try:
                    return item.decode("utf-8", errors="ignore")
                except Exception:
                    return ""
            if isinstance(item, str):
                return item

            try:
                choices = getattr(item, "choices", None)
                if not choices:
                    return ""
                first = choices[0]
                delta = getattr(first, "delta", None)
                if delta is None and isinstance(first, dict):
                    delta = first.get("delta")
                if isinstance(delta, dict):
                    return str(delta.get("content", "") or "")
                return str(getattr(delta, "content", "") or "")
            except Exception:
                return ""

        try:
            from dspy.streaming import streamify

            stream_predict = streamify(
                self.predict,
                stream_listeners=[],
                include_final_prediction_in_output_stream=True,
                async_streaming=False,
            )

            prediction = None
            streamed_parts: list[str] = []

            for item in stream_predict(environment_state=environment_state):
                if should_abort and should_abort():
                    raise PromptAborted("Prompt aborted")

                if isinstance(item, dspy.Prediction):
                    prediction = item
                    continue

                chunk = _extract_delta_text(item)
                if chunk:
                    streamed_parts.append(chunk)
                    on_stream_chunk(chunk, False)

            if prediction is not None:
                final_markdown = str(getattr(prediction, "response", "") or "")
                if final_markdown:
                    return final_markdown

            return "".join(streamed_parts)

        except PromptAborted:
            raise
        except Exception:
            if should_abort and should_abort():
                raise PromptAborted("Prompt aborted")
            prediction = self.predict(environment_state=environment_state)
            return str(getattr(prediction, "response", "") or "")

    # ── react loop ──────────────────────────────────────────────────
    MAX_ITERATIONS = 5

    def react(
        self,
        on_response: Callable[[str], None] | None = None,
        on_step: Callable[[ReactionStep], None] | None = None,
        on_stream_chunk: Callable[[str, bool], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[str]:
        """React loop: run the LM, execute code, re-trigger if code ran.

        Stops when:
          - The AI responds with no code blocks (just talks), or
          - MAX_ITERATIONS is hit (safety cap).

        If provided, callbacks are invoked *after* all runnable blocks from
        that LM response have finished executing.
        """
        responses: list[str] = []

        # Auto-checkpoint before the AI modifies anything.
        # This is the save point that /undo restores to.
        if self.kernel.checkpoints is not None:
            user_input = self.kernel.shell.user_ns.get("user_input", "") or ""
            ckpt_label = (
                f"pre: {user_input[:50]}" if user_input else "checkpoint"
            )
            self.kernel.checkpoints.checkpoint(
                self.kernel, agent=self, label=ckpt_label,
            )

        turn_trace: list[dict[str, Any]] = []

        for i in range(self.MAX_ITERATIONS):
            if should_abort and should_abort():
                raise PromptAborted("Prompt aborted")

            env_state = self.build_environment_state(turn_trace=turn_trace)

            markdown = self._predict_response_markdown(
                environment_state=env_state,
                on_stream_chunk=on_stream_chunk,
                should_abort=should_abort,
            )
            responses.append(markdown)

            # Execute every runnable code block before the next model turn.
            blocks = executable_blocks(markdown)
            executions: list[dict[str, Any]] = []
            for block in blocks:
                if should_abort and should_abort():
                    raise PromptAborted("Prompt aborted")
                entry = self.kernel.execute(block.code, tag="agent")
                executions.append(entry)
                turn_trace.append(
                    {
                        "iteration": i + 1,
                        "code": block.code,
                        "stdout": entry.get("stdout", ""),
                        "stderr": entry.get("stderr", ""),
                        "result": entry.get("result", ""),
                        "error": entry.get("error", ""),
                        "success": entry.get("success", True),
                        "truncated": any(
                            len(str(entry.get(k, "") or ""))
                            > EXEC_OUTPUT_PER_BLOCK_LIMIT
                            for k in ("stdout", "stderr", "result", "error")
                        ) or len(block.code) > EXEC_OUTPUT_PER_BLOCK_LIMIT,
                    }
                )

            # Store the response in the namespace
            self.kernel.push_ai_response(
                markdown,
                prose=strip_fenced_code(markdown),
            )

            step = ReactionStep(
                iteration=i,
                markdown=markdown,
                blocks=blocks,
                executions=executions,
                is_final=not blocks,
            )
            if on_step:
                on_step(step)
            if on_response:
                on_response(markdown)

            # If no code was executed, the AI is just talking — we're done
            if step.is_final:
                break

        return responses

    # ── spawn ───────────────────────────────────────────────────────
    def spawn(self, task: str | None = None) -> Any:
        """Create a subprocess child agent with a cloned copy of state.

        Returns a ``SubprocessAgentProxy``. The child runs in a dedicated
        Python process with its own IPython shell, namespace, and history.
        Changes in the child never affect the parent process.
        """
        blob = clone_state(self.kernel, agent=self)

        lm = getattr(dspy.settings, "lm", None)
        model = getattr(lm, "model", None) or "gemini/gemini-3-flash-preview"

        from aiipython.subprocess_agent import SubprocessAgentProxy

        return SubprocessAgentProxy(state_blob=blob, task=task, model=model)
