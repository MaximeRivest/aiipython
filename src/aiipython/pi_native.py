"""pi-native bridge (Python backend <-> Node Pi InteractiveMode host)."""

from __future__ import annotations

import ast
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from aiipython.agent import PromptAborted, ReactionStep
from aiipython.parser import strip_executable_fenced_code
from aiipython.session import Session

_UI_MAX_LINE_CHARS = 140
_UI_MAX_BLOCK_LINES = 120
_SNAPSHOT_TEXT_CHAR_LIMIT = 20_000
_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}


def _abbrev_line(line: str, max_len: int = _UI_MAX_LINE_CHARS) -> str:
    if len(line) <= max_len:
        return line
    keep = max(20, max_len - 22)
    return f"{line[:keep]}… [{len(line)} chars]"


def _abbrev_multiline(
    text: str,
    *,
    max_len: int = _UI_MAX_LINE_CHARS,
    max_lines: int = _UI_MAX_BLOCK_LINES,
) -> str:
    lines = text.splitlines() or [""]
    out = [_abbrev_line(line, max_len=max_len) for line in lines[:max_lines]]
    if len(lines) > max_lines:
        out.append(f"… [{len(lines) - max_lines} more lines]")
    return "\n".join(out)


def _without_fenced_code(markdown: str) -> str:
    return strip_executable_fenced_code(markdown)


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


class PiNativeBackend:
    def __init__(self, session: Session) -> None:
        self.session = session
        self._emit_lock = threading.Lock()
        self._emit_raw: Callable[[dict[str, Any]], None] | None = None

        self._prompt_lock = threading.Lock()
        self._prompt_cancel = threading.Event()

        self._mlflow_proc: subprocess.Popen[str] | None = None
        self._mlflow_ui_url: str | None = None

    @property
    def kernel(self):
        return self.session.kernel

    @property
    def agent(self):
        return self.session.agent

    def attach_transport(self, emit_raw: Callable[[dict[str, Any]], None]) -> None:
        with self._emit_lock:
            self._emit_raw = emit_raw

    def detach_transport(self) -> None:
        with self._emit_lock:
            self._emit_raw = None

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        with self._emit_lock:
            fn = self._emit_raw
        if not fn:
            return
        try:
            fn({"type": "event", "event": event, "data": data})
        except Exception:
            pass

    def hello(self) -> dict[str, Any]:
        return {
            "model": self.session.model,
            "cwd": str(Path.cwd()),
            "version": "1",
        }

    def get_inspector(self) -> dict[str, Any]:
        return {
            "snapshot": self.kernel.snapshot(),
            "history": self.kernel.history[-80:],
        }

    def list_pins(self, include_stats: bool = False) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for item in self.agent.context_list():
            kind = str(item.get("kind", "text"))
            row: dict[str, Any] = {
                "id": str(item.get("id", "")),
                "kind": kind,
                "label": str(item.get("label", "")),
                "source": str(item.get("source", "manual")),
            }

            if kind in {"file", "image"}:
                path = str(item.get("path", ""))
                p = Path(path)
                row["path"] = path
                try:
                    row["relpath"] = str(p.resolve().relative_to(Path.cwd()))
                except Exception:
                    row["relpath"] = path
                row["ext"] = p.suffix.lower()

                if include_stats and kind == "file" and p.exists() and p.is_file():
                    try:
                        text = p.read_text(encoding="utf-8", errors="replace")
                        row["line_count"] = len(text.splitlines())
                    except Exception:
                        pass
            else:
                text = str(item.get("text", ""))
                row["preview"] = (text[:120] + "…") if len(text) > 120 else text

            rows.append(row)

        return {"items": rows}

    def remove_pin(self, item_id: str) -> dict[str, Any]:
        removed = self.agent.context_remove(str(item_id))
        return {"removed": bool(removed), "id": str(item_id)}

    def add_pin(self, path: str, *, source: str = "/pin") -> dict[str, Any]:
        raw = str(path or "").strip()
        if not raw:
            raise ValueError("path is required")

        item = self.agent.context_add_file(raw, source=source)
        out: dict[str, Any] = {
            "id": str(item.get("id", "")),
            "kind": str(item.get("kind", "file")),
            "label": str(item.get("label", "")),
            "source": str(item.get("source", source)),
            "path": str(item.get("path", raw)),
        }

        try:
            p = Path(str(out["path"]))
            out["relpath"] = str(p.resolve().relative_to(Path.cwd()))
            out["ext"] = p.suffix.lower()
        except Exception:
            out["relpath"] = out["path"]
            out["ext"] = ""

        return {"item": out}

    def abort_prompt(self) -> dict[str, Any]:
        self._prompt_cancel.set()
        return {"requested": True}

    def _mlflow_poll_running(self) -> bool:
        proc = self._mlflow_proc
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        self._mlflow_proc = None
        self._mlflow_ui_url = None
        return False

    def mlflow_status(self) -> dict[str, Any]:
        from aiipython.mlflow_integration import resolve_mlflow_config

        cfg = resolve_mlflow_config()
        running = self._mlflow_poll_running()
        return {
            "enabled": bool(cfg["enabled"]),
            "tracking_uri": str(cfg["tracking_uri"]),
            "experiment": str(cfg["experiment"]),
            "ui_running": running,
            "ui_url": self._mlflow_ui_url or "",
            "pid": int(self._mlflow_proc.pid) if running and self._mlflow_proc else None,
        }

    def _open_url(self, url: str) -> bool:
        try:
            import webbrowser

            if webbrowser.open(url, new=2):
                return True
        except Exception:
            pass

        for opener in ("xdg-open", "open", "start"):
            exe = shutil.which(opener)
            if not exe:
                continue
            try:
                subprocess.Popen(
                    [exe, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                continue

        return False

    def mlflow_start(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        *,
        open_browser: bool = True,
    ) -> dict[str, Any]:
        host = (host or "127.0.0.1").strip() or "127.0.0.1"

        if self._mlflow_poll_running():
            status = self.mlflow_status()
            status["already_running"] = True
            if open_browser and status.get("ui_url"):
                status["browser_opened"] = self._open_url(str(status["ui_url"]))
            else:
                status["browser_opened"] = False
            return status

        if port is None:
            candidate_port = 5000
            if not _port_available(host, candidate_port):
                candidate_port = _pick_free_port(host)
        else:
            candidate_port = int(port)
            if candidate_port <= 0 or candidate_port > 65535:
                raise ValueError(f"Invalid port: {candidate_port}")
            if not _port_available(host, candidate_port):
                raise RuntimeError(f"Port {candidate_port} is already in use on {host}")

        os.environ["AIIPYTHON_MLFLOW"] = "1"

        from aiipython.mlflow_integration import configure_mlflow_from_env, resolve_mlflow_config

        cfg = resolve_mlflow_config()
        tracking_uri = str(cfg["tracking_uri"])
        experiment = str(cfg["experiment"])

        if not configure_mlflow_from_env():
            raise RuntimeError(
                "Could not enable MLflow tracing. Install MLflow with `uv sync --extra mlflow`."
            )

        cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_uri,
            "--host",
            host,
            "--port",
            str(candidate_port),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        self._mlflow_proc = proc
        self._mlflow_ui_url = f"http://{host}:{candidate_port}"

        time.sleep(0.6)
        if proc.poll() is not None:
            self._mlflow_proc = None
            self._mlflow_ui_url = None
            raise RuntimeError(
                "Failed to start MLflow UI process. Ensure MLflow is installed in this environment."
            )

        browser_opened = False
        if open_browser and self._mlflow_ui_url:
            browser_opened = self._open_url(self._mlflow_ui_url)

        return {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "experiment": experiment,
            "ui_running": True,
            "ui_url": self._mlflow_ui_url,
            "pid": int(proc.pid),
            "already_running": False,
            "browser_opened": browser_opened,
        }

    def mlflow_stop(self) -> dict[str, Any]:
        proc = self._mlflow_proc
        self._mlflow_proc = None
        self._mlflow_ui_url = None

        if proc is None:
            return {"stopped": False, "was_running": False}

        was_running = proc.poll() is None
        if was_running:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        return {"stopped": True, "was_running": was_running}

    def shutdown(self) -> None:
        try:
            self.mlflow_stop()
        except Exception:
            pass

    def _ensure_lm_ready(self, model: str | None) -> None:
        import dspy

        target_model = model.strip() if isinstance(model, str) else ""
        needs_lm = (not hasattr(dspy.settings, "lm")) or (dspy.settings.lm is None)

        if target_model and (needs_lm or target_model != self.session.model):
            self.session.switch_model(target_model)
            return

        if needs_lm:
            current = self.session.model if isinstance(self.session.model, str) else ""
            if not current or "/" not in current:
                current = os.environ.get("AIIPYTHON_MODEL") or "gemini/gemini-3-flash-preview"
            self.session.switch_model(current)

    def transform_at_refs(self, raw_text: str) -> dict[str, Any]:
        """Apply aiipython @-reference behavior to one input string.

        Returns:
            {
              "handled": bool,     # true => no model call should be made
              "text": str,         # transformed user text
              "messages": [str],   # informational/warning lines for UI
            }
        """
        raw = raw_text or ""
        text = raw.strip()
        if not text:
            return {"handled": False, "text": "", "messages": []}

        # Slash/shell commands are handled separately and do not pass through
        # @-pinning rewrite.
        if text.startswith("/") or text.startswith("!"):
            return {"handled": False, "text": text, "messages": []}

        raw_lstrip = raw.lstrip()
        if raw_lstrip.startswith("@ "):
            payload = raw_lstrip[2:]
            msg = self._context_add_snapshot_payload(payload)
            return {"handled": True, "text": "", "messages": [msg]}

        transformed, messages = self._consume_at_context_refs(text)
        handled = not bool(transformed)
        return {
            "handled": handled,
            "text": transformed,
            "messages": messages,
        }

    def _consume_at_context_refs(self, text: str) -> tuple[str, list[str]]:
        messages: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            token = raw.rstrip(",.;:!?)")
            suffix = raw[len(token):]
            if not token:
                return match.group(0)

            lower = token.lower()
            if lower in {"clip", "clipboard"}:
                msg = self._context_add_from_clipboard(snapshot=True)
                messages.append(msg)
                return suffix

            resolved = self._resolve_at_file_token(token)
            if isinstance(resolved, list):
                shown = ", ".join(str(p.relative_to(Path.cwd())) for p in resolved[:5])
                messages.append(f"⚠ `@{token}` matched multiple files. Be specific: {shown}")
                return match.group(0)

            if resolved is None:
                return match.group(0)

            try:
                item = self.agent.context_add_file(str(resolved), source=f"@{token}")
                messages.append(f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**")
                return suffix
            except Exception as exc:
                messages.append(f"⚠ failed to pin `@{token}`: {exc}")
                return match.group(0)

        updated = re.sub(r"(?<!\S)@([^\s]+)", _replace, text)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        return updated, messages

    def _resolve_at_file_token(self, token: str) -> Path | list[Path] | None:
        raw = token.strip().strip('"').strip("'")
        if not raw:
            return None

        if raw.startswith("file://"):
            parsed = urlparse(raw)
            raw = unquote(parsed.path)

        p = Path(raw).expanduser()
        if p.is_file():
            return p.resolve()

        rel = (Path.cwd() / raw).expanduser()
        if rel.is_file():
            return rel.resolve()

        if "/" not in raw and "\\" not in raw:
            matches = self._find_cwd_file_candidates(raw, limit=8)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return matches

        return None

    def _find_cwd_file_candidates(self, needle: str, *, limit: int = 8) -> list[Path]:
        out: list[Path] = []
        root = Path.cwd()
        skip_dirs = {
            ".git", "node_modules", ".venv", "dist", "build", ".cache",
            "__pycache__", ".aiipython_checkpoints",
        }
        needle_l = needle.lower()

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            depth = len(current.relative_to(root).parts)
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            if depth >= 6:
                dirnames[:] = []

            for fname in filenames:
                if needle_l not in fname.lower():
                    continue
                out.append((current / fname).resolve())
                if len(out) >= limit:
                    return out

        return out

    def _snapshot_context_dir(self) -> Path:
        d = Path.cwd() / ".aiipython_context"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _snapshot_file_to_context(self, path: Path, *, source: str) -> dict[str, Any]:
        p = path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))

        if p.suffix.lower() in _IMAGE_EXTS:
            snap = self._snapshot_context_dir() / f"snapshot_{int(time.time())}_{p.name}"
            shutil.copy2(p, snap)
            return self.agent.context_add_file(str(snap), source=source)

        raw = p.read_bytes()
        if b"\x00" in raw[:2048]:
            text = f"<binary snapshot: {p} ({len(raw)} bytes)>"
        else:
            text = raw.decode("utf-8", errors="replace")
            if len(text) > _SNAPSHOT_TEXT_CHAR_LIMIT:
                text = (
                    text[:_SNAPSHOT_TEXT_CHAR_LIMIT]
                    + f"\n… [{len(text) - _SNAPSHOT_TEXT_CHAR_LIMIT} more chars]"
                )

        return self.agent.context_add_text(
            text,
            label=f"{p.name} snapshot",
            source=source,
        )

    def _context_add_snapshot_payload(self, payload: str) -> str:
        raw = payload.rstrip("\n")
        if not raw.strip():
            return self._context_add_from_clipboard(snapshot=True)

        token = raw.strip().strip('"').strip("'")
        if token.startswith("file://"):
            parsed = urlparse(token)
            token = unquote(parsed.path)

        if "\n" not in token:
            p = Path(token).expanduser()
            if p.is_file():
                item = self._snapshot_file_to_context(p, source="@paste:file")
                return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}** (snapshot)"

        item = self.agent.context_add_text(raw, label="pasted", source="@paste:text")
        return f"📌 pinned `{item['id']}` [text] **{item['label']}** (snapshot)"

    def _read_clipboard_text(self) -> str | None:
        commands = [
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["pbpaste"],
        ]
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
            except Exception:
                continue
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.strip()
        return None

    def _context_add_from_clipboard(self, *, snapshot: bool = False) -> str:
        try:
            from PIL import Image, ImageGrab

            grabbed = ImageGrab.grabclipboard()
            if isinstance(grabbed, Image.Image):
                img_path = self._snapshot_context_dir() / f"clipboard_{int(time.time())}.png"
                grabbed.save(img_path)

                item = self.agent.context_add_file(str(img_path), source="@clip:image")
                name = img_path.stem
                self.kernel.push_image(name, grabbed)
                return (
                    f"📌 pinned `{item['id']}` [image] **{item['label']}** "
                    f"and loaded `images['{name}']`"
                )

            if isinstance(grabbed, list) and grabbed:
                added: list[str] = []
                for path_like in grabbed[:3]:
                    p = Path(path_like).expanduser()
                    if not p.is_file():
                        continue

                    if snapshot:
                        item = self._snapshot_file_to_context(p, source="@clip:path-snapshot")
                    else:
                        item = self.agent.context_add_file(str(p), source="@clip:path")
                    added.append(f"`{item['id']}` {item['label']}")
                if added:
                    suffix = " (snapshot)" if snapshot else ""
                    return "📌 pinned clipboard paths" + suffix + ": " + ", ".join(added)
        except Exception:
            pass

        clip = self._read_clipboard_text()
        if not clip:
            return "⚠ clipboard is empty or unreadable"

        clip = clip.strip()
        resolved = self._resolve_at_file_token(clip)
        if isinstance(resolved, Path):
            if snapshot:
                item = self._snapshot_file_to_context(resolved, source="@clip:path-snapshot")
                return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}** (snapshot)"
            item = self.agent.context_add_file(str(resolved), source="@clip:path")
            return f"📌 pinned `{item['id']}` [{item['kind']}] **{item['label']}**"

        item = self.agent.context_add_text(clip, label="clipboard", source="@clip:text")
        return f"📌 pinned `{item['id']}` [text] **{item['label']}**"

    def _prefer_python_input(self, cmd: str) -> bool:
        """Heuristic for `!`/`!!` bridge: allow direct Python when obvious.

        Pi emits `user_bash` events for `!`/`!!`. We keep shell ergonomics for
        common commands (`ls`, `git ...`, etc.) but let users run Python directly
        (for example `x = 1` or `import pandas as pd`) in the bound IPython
        namespace.
        """
        if not cmd:
            return False
        if cmd.startswith(("!", "%", "?")):
            return True

        try:
            tree = ast.parse(cmd, mode="exec")
        except SyntaxError:
            return False

        # Single bare identifier like `ls` or `git` is often intended as shell.
        if (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Name)
            and shutil.which(tree.body[0].value.id)
        ):
            return False

        return True

    def run_ipython_shell(self, command: str, *, exclude_from_context: bool = False) -> dict[str, Any]:
        cmd = (command or "").strip()
        if not cmd:
            return {
                "output": "",
                "exit_code": 0,
                "cancelled": False,
                "truncated": False,
                "exclude_from_context": bool(exclude_from_context),
            }

        code = cmd if self._prefer_python_input(cmd) else f"!{cmd}"

        with self._prompt_lock:
            entry = self.kernel.execute(code, tag="user")
            out = self._format_exec_output(entry)
            exit_code = 0 if bool(entry.get("success", True)) else 1

        return {
            "output": out,
            "exit_code": exit_code,
            "cancelled": False,
            "truncated": False,
            "exclude_from_context": bool(exclude_from_context),
        }

    def _format_exec_output(self, entry: dict[str, Any]) -> str:
        parts: list[str] = []

        stdout = (entry.get("stdout") or "").rstrip()
        stderr = (entry.get("stderr") or "").rstrip()
        result = entry.get("result") or ""
        error = entry.get("error") or ""

        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"stderr: {stderr}")
        if result:
            parts.append(f"→ {result}")
        if error:
            parts.append(f"✗ {error}")
        if not parts:
            parts.append("✓ (no output)")

        return _abbrev_multiline("\n".join(parts))

    def _build_prompt_result(
        self,
        *,
        steps: list[ReactionStep],
        aborted: bool,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        assistant_markdown = steps[-1].markdown if steps else ""

        display_parts: list[str] = []
        react_cells: list[dict[str, str]] = []

        for step in steps:
            shown_markdown = step.markdown
            if step.blocks:
                prose = _without_fenced_code(step.markdown)
                shown_markdown = prose or "Running requested python blocks…"
            shown_markdown = (shown_markdown or "").strip()
            if shown_markdown:
                display_parts.append(shown_markdown)

            for idx, block in enumerate(step.blocks, start=1):
                entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                out = self._format_exec_output(entry)
                code = _abbrev_multiline(block.code)
                react_cells.append({"code": code, "output": out})
                display_parts.append(
                    f"▶ python #{idx}\n"
                    f"```python\n{code}\n```\n"
                    f"```text\n{out}\n```"
                )

        display_markdown = "\n\n".join(p for p in display_parts if p).strip() or assistant_markdown

        out: dict[str, Any] = {
            "assistant_markdown": assistant_markdown,
            "display_markdown": display_markdown,
            "react_cells": react_cells,
            "aborted": aborted,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
            },
            "model": self.session.model,
            "iterations": len(steps),
        }
        if stream_id is not None:
            out["stream_id"] = stream_id
        return out

    def prompt_once(self, text: str, model: str | None = None) -> dict[str, Any]:
        clean = (text or "").strip()
        if not clean:
            return self._build_prompt_result(steps=[], aborted=False)

        with self._prompt_lock:
            self._ensure_lm_ready(model)
            self._prompt_cancel.clear()

            steps: list[ReactionStep] = []
            aborted = False

            self.kernel.push_user_input(clean)
            try:
                self.agent.react(
                    on_step=lambda s: steps.append(s),
                    should_abort=self._prompt_cancel.is_set,
                )
            except PromptAborted:
                aborted = True

            return self._build_prompt_result(steps=steps, aborted=aborted)

    def prompt_stream(
        self,
        text: str,
        model: str | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        clean = (text or "").strip()
        sid = (stream_id or "").strip() or secrets.token_hex(8)
        if not clean:
            return self._build_prompt_result(steps=[], aborted=False, stream_id=sid)

        with self._prompt_lock:
            self._ensure_lm_ready(model)
            self._prompt_cancel.clear()

            steps: list[ReactionStep] = []
            aborted = False

            self.kernel.push_user_input(clean)
            self._emit("prompt_stream_start", {"stream_id": sid})

            def _on_stream_chunk(chunk: str, is_last: bool) -> None:
                self._emit(
                    "prompt_stream_delta",
                    {
                        "stream_id": sid,
                        "chunk": chunk,
                        "is_last": bool(is_last),
                    },
                )

            def _on_step(step: ReactionStep) -> None:
                steps.append(step)
                blocks_payload: list[dict[str, Any]] = []
                for idx, block in enumerate(step.blocks, start=1):
                    entry = step.executions[idx - 1] if idx - 1 < len(step.executions) else {}
                    blocks_payload.append(
                        {
                            "index": idx,
                            "code": _abbrev_multiline(block.code),
                            "output": self._format_exec_output(entry),
                            "success": bool(entry.get("success", True)),
                        }
                    )
                self._emit(
                    "prompt_stream_step",
                    {
                        "stream_id": sid,
                        "iteration": int(step.iteration),
                        "is_final": bool(step.is_final),
                        "blocks": blocks_payload,
                    },
                )

            try:
                self.agent.react(
                    on_step=_on_step,
                    on_stream_chunk=_on_stream_chunk,
                    should_abort=self._prompt_cancel.is_set,
                )
            except PromptAborted:
                aborted = True
            finally:
                self._emit("prompt_stream_end", {"stream_id": sid})

            return self._build_prompt_result(steps=steps, aborted=aborted, stream_id=sid)


class PiRpcServer:
    """Single-client newline-delimited JSON-RPC-over-TCP server."""

    def __init__(self, backend: PiNativeBackend, host: str = "127.0.0.1", port: int = 0) -> None:
        self.backend = backend
        self.host = host
        self.port = port

        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._conn_lock = threading.Lock()
        self._conn_file_w: Any = None

    def start(self) -> tuple[str, int]:
        if self._sock is not None:
            raise RuntimeError("server already started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(0.5)

        self._sock = sock
        self.host, self.port = sock.getsockname()[0], int(sock.getsockname()[1])

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        return self.host, self.port

    def stop(self) -> None:
        self._stop.set()

        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        with self._conn_lock:
            fw = self._conn_file_w
            self._conn_file_w = None
        if fw is not None:
            try:
                fw.close()
            except Exception:
                pass

        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)

        self.backend.detach_transport()
        self.backend.shutdown()

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                self._serve_connection(conn)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                self.backend.detach_transport()

    def _serve_connection(self, conn: socket.socket) -> None:
        file_r = conn.makefile("r", encoding="utf-8", newline="\n")
        file_w = conn.makefile("w", encoding="utf-8", newline="\n")

        def send_raw(obj: dict[str, Any]) -> None:
            text = json.dumps(obj, ensure_ascii=False, default=str)
            with self._conn_lock:
                if self._conn_file_w is None:
                    return
                self._conn_file_w.write(text + "\n")
                self._conn_file_w.flush()

        with self._conn_lock:
            self._conn_file_w = file_w
        self.backend.attach_transport(send_raw)

        def _handle_request(req_id: Any, method: str, params: dict[str, Any]) -> None:
            try:
                result = self._dispatch(method, params)
                send_raw({"type": "response", "id": req_id, "ok": True, "result": result})
            except Exception as exc:
                send_raw(
                    {
                        "type": "response",
                        "id": req_id,
                        "ok": False,
                        "error": {"message": str(exc), "type": type(exc).__name__},
                    }
                )

        while not self._stop.is_set():
            line = file_r.readline()
            if not line:
                break

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") != "request":
                continue

            req_id = msg.get("id")
            method = str(msg.get("method", ""))
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                params = {}

            if method in {"prompt_once", "prompt_stream"}:
                threading.Thread(
                    target=_handle_request,
                    args=(req_id, method, params),
                    daemon=True,
                ).start()
            else:
                _handle_request(req_id, method, params)

        with self._conn_lock:
            self._conn_file_w = None

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "hello":
            return self.backend.hello()
        if method == "ping":
            return {"pong": True}
        if method == "prompt_once":
            model = params.get("model")
            return self.backend.prompt_once(
                text=str(params.get("text", "")),
                model=str(model) if model is not None else None,
            )
        if method == "prompt_stream":
            model = params.get("model")
            return self.backend.prompt_stream(
                text=str(params.get("text", "")),
                model=str(model) if model is not None else None,
                stream_id=str(params.get("stream_id", "") or ""),
            )
        if method == "abort_prompt":
            return self.backend.abort_prompt()
        if method == "transform_at_refs":
            return self.backend.transform_at_refs(str(params.get("text", "")))
        if method == "run_ipython_shell":
            return self.backend.run_ipython_shell(
                str(params.get("command", "")),
                exclude_from_context=bool(params.get("excludeFromContext", False)),
            )
        if method == "get_inspector":
            return self.backend.get_inspector()
        if method == "list_pins":
            return self.backend.list_pins(include_stats=bool(params.get("include_stats", False)))
        if method == "remove_pin":
            return self.backend.remove_pin(str(params.get("id", "")))
        if method == "add_pin":
            return self.backend.add_pin(
                str(params.get("path", "")),
                source=str(params.get("source", "/pin") or "/pin"),
            )
        if method == "mlflow_status":
            return self.backend.mlflow_status()
        if method == "mlflow_start":
            host = str(params.get("host", "127.0.0.1") or "127.0.0.1")
            port = params.get("port")
            open_browser = bool(params.get("open_browser", True))
            return self.backend.mlflow_start(
                host=host,
                port=int(port) if port is not None else None,
                open_browser=open_browser,
            )
        if method == "mlflow_stop":
            return self.backend.mlflow_stop()
        if method == "provide_prompt":
            return {"accepted": False}
        raise RuntimeError(f"Unknown method: {method}")


def _snapshot_terminal_state() -> tuple[int, Any] | None:
    if not sys.stdin.isatty():
        return None
    try:
        import termios

        fd = sys.stdin.fileno()
        return (fd, termios.tcgetattr(fd))
    except Exception:
        return None


def _restore_terminal_state(snapshot: tuple[int, Any] | None) -> None:
    try:
        if sys.stdout.isatty():
            sys.stdout.write(
                "\x1b[?2026l"
                "\x1b[?2004l"
                "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1004l\x1b[?1006l"
                "\x1b[?47l\x1b[?1047l\x1b[?1049l"
                "\x1b[?25h"
            )
            sys.stdout.flush()
    except Exception:
        pass

    if snapshot is not None:
        fd, attrs = snapshot
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception:
            pass

    stty = shutil.which("stty")
    if stty and sys.stdin.isatty():
        try:
            subprocess.run([stty, "sane"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def _run_node_ui(node: str, entry: Path, env: dict[str, str]) -> None:
    tty_snapshot = _snapshot_terminal_state()
    proc = subprocess.Popen([node, str(entry)], env=env)
    try:
        proc.wait()
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _restore_terminal_state(tty_snapshot)


def _native_host_dir() -> Path:
    return Path(__file__).resolve().parent / "pi_native_host"


def _ensure_native_host_dependencies(native_dir: Path) -> None:
    pkg = native_dir / "package.json"
    deps_marker = native_dir / "node_modules" / "@mariozechner" / "pi-coding-agent"
    if deps_marker.exists():
        return

    if not pkg.exists():
        raise RuntimeError(f"pi-native host package.json missing: {pkg}")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm is required for AIIPYTHON_UI=pi-native.")

    proc = subprocess.run(
        [npm, "install", "--no-audit", "--no-fund"],
        cwd=str(native_dir),
        check=False,
    )
    if proc.returncode != 0 or not deps_marker.exists():
        raise RuntimeError(
            "Failed to install pi-native host dependencies. "
            f"Try manually: cd {native_dir} && npm install"
        )


def run_pi_native(session: Session) -> None:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for AIIPYTHON_UI=pi-native.")

    native_dir = _native_host_dir()
    _ensure_native_host_dependencies(native_dir)

    entry = native_dir / "pi_native_host.mjs"
    if not entry.exists():
        raise RuntimeError(f"pi-native host entry not found: {entry}")

    backend = PiNativeBackend(session)

    # If MLflow tracing is enabled, auto-start the UI and try opening browser.
    if _env_bool("AIIPYTHON_MLFLOW", False) and _env_bool("AIIPYTHON_MLFLOW_AUTO_UI", True):
        try:
            status = backend.mlflow_start(
                host=os.environ.get("AIIPYTHON_MLFLOW_UI_HOST", "127.0.0.1"),
                port=(
                    int(os.environ["AIIPYTHON_MLFLOW_UI_PORT"])
                    if os.environ.get("AIIPYTHON_MLFLOW_UI_PORT")
                    else None
                ),
                open_browser=_env_bool("AIIPYTHON_MLFLOW_OPEN_BROWSER", True),
            )
            ui_url = str(status.get("ui_url", "") or "")
            if ui_url:
                print(f"[aiipython] MLflow UI: {ui_url}", file=sys.stderr)
                print(ui_url, file=sys.stderr)
        except Exception as exc:
            print(f"[aiipython] Could not auto-start MLflow UI: {exc}", file=sys.stderr)

    server = PiRpcServer(backend)
    host, port = server.start()

    env = os.environ.copy()
    env["AIIPYTHON_RPC_HOST"] = host
    env["AIIPYTHON_RPC_PORT"] = str(port)

    try:
        _run_node_ui(node, entry, env)
    finally:
        server.stop()
