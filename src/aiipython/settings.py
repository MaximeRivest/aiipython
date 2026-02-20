"""Persistent aiipython settings.

Stores lightweight user preferences in ``~/.aiipython/settings.json``.
Currently used for:
  - Remembering the last selected model across launches
  - Keeping a small recent-models list for interactive menus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_DIR = Path.home() / ".aiipython"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
DEFAULT_MODEL = "gemini/gemini-3-flash-preview"


class SettingsManager:
    def __init__(self, path: Path = SETTINGS_FILE) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
        self.path.chmod(0o600)

    def get_last_model(self) -> str | None:
        v = self._data.get("last_model")
        return v if isinstance(v, str) and v.strip() else None

    def set_last_model(self, model: str) -> None:
        self._data["last_model"] = model
        self.add_recent_model(model)
        self.save()

    def get_recent_models(self) -> list[str]:
        values = self._data.get("recent_models", [])
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for v in values:
            if isinstance(v, str) and v.strip() and v not in out:
                out.append(v)
        return out

    def add_recent_model(self, model: str, limit: int = 12) -> None:
        models = [m for m in self.get_recent_models() if m != model]
        models.insert(0, model)
        self._data["recent_models"] = models[:limit]


_settings: SettingsManager | None = None


def get_settings() -> SettingsManager:
    global _settings
    if _settings is None:
        _settings = SettingsManager()
    return _settings
