"""Checkpoint tree — snapshot, fork, and navigate IPython session states.

Each checkpoint captures:
  - The full user namespace (serialized with dill)
  - The kernel activity history
  - The agent's active image labels
  - Metadata (timestamp, label, parent/child relationships)

Checkpoints form a tree.  Auto-checkpoint before each AI turn so
you can ``/undo``, ``/restore``, ``/fork``, and ``/tree`` freely.

Usage from the REPL::

    %tree             # show checkpoint tree
    %undo             # revert AI's last effects
    %restore 0003     # jump to any checkpoint
    %fork my idea     # mark a named branch point
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import dill

if TYPE_CHECKING:
    from aiipython.kernel import Kernel

log = logging.getLogger(__name__)


# Names to always skip when serializing the namespace.
# These are IPython internals or agent-injected callables that get
# re-created on restore.
_SKIP_NAMES = frozenset({
    "In", "Out", "get_ipython", "exit", "quit", "open",
    "_kernel", "_oh", "_dh", "_ih", "_ii", "_iii", "_i",
    "terminal_history", "chat",
    # Agent-injected callables — rebuilt by ReactiveAgent.__init__
    "agent", "spawn_agent", "look_at",
})


# ── Tree node ────────────────────────────────────────────────────────

@dataclass
class Node:
    """A single checkpoint in the tree."""

    id: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    label: str = ""
    timestamp: float = 0.0
    skipped_vars: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "parent": self.parent,
            "children": list(self.children),
            "label": self.label,
            "timestamp": self.timestamp,
        }
        if self.skipped_vars:
            d["skipped_vars"] = self.skipped_vars
        return d


# ── Checkpoint tree ──────────────────────────────────────────────────

class CheckpointTree:
    """Manages a tree of serialized IPython session snapshots."""

    def __init__(self, root_dir: Path | str = ".aiipython_checkpoints") -> None:
        self.root = Path(root_dir)
        self.nodes_dir = self.root / "nodes"
        self.tree_file = self.root / "tree.json"
        self.nodes: dict[str, Node] = {}
        self.current: str | None = None
        self._load_or_init()

    # ── persistence ──────────────────────────────────────────────

    def _load_or_init(self) -> None:
        if self.tree_file.exists():
            data = json.loads(self.tree_file.read_text())
            self.current = data.get("current")
            for nid, meta in data.get("nodes", {}).items():
                self.nodes[nid] = Node(id=nid, **meta)
        else:
            self.nodes_dir.mkdir(parents=True, exist_ok=True)

    def _save_tree(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "current": self.current,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        self.tree_file.write_text(json.dumps(data, indent=2))

    def _next_id(self) -> str:
        return f"{len(self.nodes):04d}"

    # ── core operations ──────────────────────────────────────────

    def checkpoint(
        self,
        kernel: Kernel,
        agent: Any = None,
        label: str = "",
    ) -> str:
        """Snapshot current state as a new child of the current node.

        Returns the new node id.
        """
        nid = self._next_id()
        node_dir = self.nodes_dir / nid
        node_dir.mkdir(parents=True, exist_ok=True)

        # 1. Serialize namespace (skip what we can't)
        ns, skipped = _safe_namespace(kernel)
        (node_dir / "namespace.dill").write_bytes(dill.dumps(ns))

        # 2. Save history log
        (node_dir / "history.json").write_text(
            json.dumps(kernel.history, indent=2, default=str)
        )

        # 3. Save active image labels (if agent available)
        active_labels: list[str] = []
        if agent is not None:
            active_labels = [lbl for lbl, _ in agent._active_images]
        (node_dir / "active_image_labels.json").write_text(
            json.dumps(active_labels)
        )

        # 4. Create node
        node = Node(
            id=nid,
            parent=self.current,
            label=label or f"checkpoint {nid}",
            timestamp=time.time(),
            skipped_vars=skipped,
        )
        self.nodes[nid] = node
        if self.current is not None and self.current in self.nodes:
            self.nodes[self.current].children.append(nid)
        self.current = nid
        self._save_tree()

        log.info("Checkpoint %s: %s", nid, label)
        return nid

    def restore(
        self,
        kernel: Kernel,
        node_id: str,
        agent: Any = None,
    ) -> None:
        """Restore kernel (and optionally agent) state from a checkpoint.

        Raises ``KeyError`` if *node_id* doesn't exist.
        """
        if node_id not in self.nodes:
            raise KeyError(f"No checkpoint '{node_id}'")

        node_dir = self.nodes_dir / node_id

        # 1. Restore namespace
        ns = dill.loads((node_dir / "namespace.dill").read_bytes())
        _reset_namespace(kernel, ns)

        # 2. Restore history
        kernel.history = json.loads(
            (node_dir / "history.json").read_text()
        )
        kernel.shell.user_ns["terminal_history"] = kernel.history

        # 3. Restore active images on agent (if available)
        if agent is not None:
            labels_file = node_dir / "active_image_labels.json"
            if labels_file.exists():
                labels = json.loads(labels_file.read_text())
                _restore_active_images(agent, labels)

        # 4. Update current pointer
        self.current = node_id
        self._save_tree()

        log.info("Restored to checkpoint %s", node_id)

    def undo(
        self,
        kernel: Kernel,
        agent: Any = None,
    ) -> str | None:
        """Restore to the current checkpoint — undo AI's last effects.

        The auto-checkpoint before each ``react()`` captures the state
        *before* the AI ran.  Calling undo restores that snapshot,
        effectively reverting everything the AI did this turn.

        Returns the checkpoint id, or ``None`` if there's nothing to undo.
        """
        if self.current is None:
            return None
        self.restore(kernel, self.current, agent=agent)
        return self.current

    def fork(
        self,
        kernel: Kernel,
        agent: Any = None,
        label: str = "",
    ) -> str:
        """Create a named branch point from the current state.

        Equivalent to ``checkpoint()`` with a descriptive label.
        """
        return self.checkpoint(
            kernel,
            agent=agent,
            label=label or f"fork from {self.current or 'root'}",
        )

    # ── display ──────────────────────────────────────────────────

    def show_tree(self) -> str:
        """Render an ASCII-art checkpoint tree."""
        if not self.nodes:
            return "(no checkpoints yet)"

        roots = [n for n in self.nodes.values() if n.parent is None]
        if not roots:
            return "(no checkpoints yet)"

        lines: list[str] = ["Checkpoint tree:"]
        for i, root in enumerate(roots):
            self._render(root.id, "", i == len(roots) - 1, lines)
        return "\n".join(lines)

    def _render(
        self, nid: str, prefix: str, last: bool, lines: list[str],
    ) -> None:
        node = self.nodes[nid]
        marker = "●" if nid == self.current else "○"
        connector = "└── " if last else "├── "
        ts = time.strftime("%H:%M:%S", time.localtime(node.timestamp))
        lines.append(
            f"{prefix}{connector}[{nid}] {marker} {node.label}  ({ts})"
        )
        child_prefix = prefix + ("    " if last else "│   ")
        for i, cid in enumerate(node.children):
            self._render(
                cid, child_prefix, i == len(node.children) - 1, lines,
            )

    # ── info ─────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def current_label(self) -> str:
        if self.current and self.current in self.nodes:
            return self.nodes[self.current].label
        return ""


# ── Sub-agent cloning (tree-independent) ─────────────────────────────

def clone_state(kernel: Kernel, agent: Any = None) -> bytes:
    """Serialize current state for loading into a sub-agent's kernel.

    Returns an opaque ``bytes`` blob.  Does *not* create a tree node.
    """
    ns, _ = _safe_namespace(kernel)
    blob = {
        "namespace": ns,
        "history": kernel.history.copy(),
        "active_image_labels": (
            [lbl for lbl, _ in agent._active_images]
            if agent else []
        ),
    }
    return dill.dumps(blob)


def load_clone(kernel: Kernel, data: bytes, agent: Any = None) -> None:
    """Load a cloned state blob into a (fresh) kernel."""
    blob = dill.loads(data)
    kernel.shell.user_ns.update(blob["namespace"])
    kernel.history = blob["history"]
    kernel.shell.user_ns["terminal_history"] = kernel.history
    if agent is not None:
        _restore_active_images(agent, blob.get("active_image_labels", []))


# ── internal helpers ─────────────────────────────────────────────────

def _safe_namespace(kernel: Kernel) -> tuple[dict[str, Any], list[str]]:
    """Extract serializable variables from the namespace.

    Returns ``(namespace_dict, list_of_skipped_var_names)``.

    Strategy: try the whole dict first (fast path).  If that fails,
    fall back to per-item testing so one bad object doesn't block
    everything.
    """
    candidates: dict[str, Any] = {}
    for k, v in kernel.shell.user_ns.items():
        if k.startswith("_") and not k.startswith("_user"):
            continue
        if k in _SKIP_NAMES:
            continue
        candidates[k] = v

    # Fast path — try everything at once
    try:
        dill.dumps(candidates)
        return candidates, []
    except Exception:
        pass

    # Slow path — test each item individually
    ns: dict[str, Any] = {}
    skipped: list[str] = []
    for k, v in candidates.items():
        try:
            dill.dumps(v)
            ns[k] = v
        except Exception:
            skipped.append(k)
            log.debug(
                "Skipping unserializable var: %s (%s)", k, type(v).__name__,
            )
    return ns, skipped


def _reset_namespace(kernel: Kernel, ns: dict[str, Any]) -> None:
    """Replace user namespace with restored values.

    Keeps IPython internals and agent-injected names intact.
    """
    keep = {"In", "Out", "get_ipython", "exit", "quit", "open"} | _SKIP_NAMES
    to_remove = [
        k for k in kernel.shell.user_ns
        if not k.startswith("_") and k not in keep
    ]
    for k in to_remove:
        del kernel.shell.user_ns[k]
    kernel.shell.user_ns.update(ns)


def _restore_active_images(agent: Any, labels: list[str]) -> None:
    """Rebuild ``agent._active_images`` from labels + the images dict."""
    import dspy as _dspy

    images_dict = agent.kernel.shell.user_ns.get("images", {})
    restored: list[tuple[str, Any]] = []
    for label in labels:
        if label in images_dict:
            try:
                restored.append((label, _dspy.Image(url=images_dict[label])))
            except Exception:
                log.debug("Could not restore active image: %s", label)
    agent._active_images = restored
