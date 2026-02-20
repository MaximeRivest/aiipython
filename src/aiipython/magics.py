"""Custom IPython magics — %pip delegates to uv, checkpoint navigation."""

from __future__ import annotations

import shutil
import subprocess
import sys

from IPython.core.magic import Magics, magics_class, line_magic


@magics_class
class UvMagics(Magics):
    """Override %pip so it runs through uv instead."""

    @line_magic
    def pip(self, line: str) -> None:
        """%pip install foo  →  uv pip install foo

        Supports all pip subcommands (install, uninstall, list, show, …)
        by forwarding the full argument line to `uv pip`.
        """
        uv = shutil.which("uv")
        if uv is None:
            print("error: `uv` not found on PATH", file=sys.stderr)
            return

        cmd = [uv, "pip", *line.split()]
        print(f"$ uv pip {line}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            # uv prints progress on stderr — show it as normal output
            print(result.stderr.rstrip())
        if result.returncode != 0:
            print(f"\nuv pip exited with code {result.returncode}")


@magics_class
class CheckpointMagics(Magics):
    """Checkpoint tree navigation from the REPL.

    Usage::

        %tree             # show checkpoint tree
        %undo             # revert AI's last effects
        %restore 0003     # jump to any checkpoint
        %fork my idea     # mark a named branch point
    """

    def _get_kernel(self):
        return self.shell.user_ns.get("_kernel")

    @line_magic
    def tree(self, line: str) -> None:
        """Show the checkpoint tree."""
        kernel = self._get_kernel()
        if kernel and kernel.checkpoints:
            print(kernel.checkpoints.show_tree())
        else:
            print("(checkpoints not available)")

    @line_magic
    def undo(self, line: str) -> None:
        """Undo to the last checkpoint (revert AI's last turn)."""
        kernel = self._get_kernel()
        if not kernel or not kernel.checkpoints:
            print("(checkpoints not available)")
            return
        agent = self.shell.user_ns.get("agent")
        nid = kernel.checkpoints.undo(kernel, agent=agent)
        if nid:
            node = kernel.checkpoints.nodes[nid]
            print(f"✓ Restored to [{nid}]: {node.label}")
        else:
            print("Nothing to undo — no checkpoints yet.")

    @line_magic
    def restore(self, line: str) -> None:
        """Restore a specific checkpoint: ``%restore 0003``"""
        kernel = self._get_kernel()
        if not kernel or not kernel.checkpoints:
            print("(checkpoints not available)")
            return
        nid = line.strip()
        if not nid:
            print("Usage: %restore <checkpoint_id>")
            return
        try:
            agent = self.shell.user_ns.get("agent")
            kernel.checkpoints.restore(kernel, nid, agent=agent)
            node = kernel.checkpoints.nodes[nid]
            print(f"✓ Restored to [{nid}]: {node.label}")
        except KeyError as exc:
            print(f"Error: {exc}")

    @line_magic
    def fork(self, line: str) -> None:
        """Create a named branch point: ``%fork my experiment``"""
        kernel = self._get_kernel()
        if not kernel or not kernel.checkpoints:
            print("(checkpoints not available)")
            return
        agent = self.shell.user_ns.get("agent")
        label = line.strip() or None
        nid = kernel.checkpoints.fork(kernel, agent=agent, label=label)
        node = kernel.checkpoints.nodes[nid]
        print(f"✓ Forked: [{nid}] {node.label}")


def register(shell) -> None:
    """Register our magics on the given InteractiveShell."""
    shell.register_magics(UvMagics)
    shell.register_magics(CheckpointMagics)
