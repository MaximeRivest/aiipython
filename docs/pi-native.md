# pi-native frontend

`pi-native` runs aiipython using Pi's real `InteractiveMode` UI (`@mariozechner/pi-coding-agent`) while keeping aiipython backend semantics in Python.

## What you get

- Pi editor behavior (autocomplete, `@` path UX, history up/down, keybindings)
- Pi slash command UI (`/model`, `/login`, `/settings`, etc.)
- Pi status/footer/queue rendering
- Pi selectors and interaction patterns

## Backend bridge

Pi model calls are intercepted via provider `streamSimple` overrides and routed to Python `prompt_once` over local JSON-RPC.

Python side:

- maintains IPython kernel state
- applies aiipython system prompt logic
- executes markdown Python blocks reactively
- returns assistant markdown + usage to Pi

## Selecting frontend

- `AIIPYTHON_UI=pi-native`
- `chat(ui="pi-native")`
- `uv run aiipython --ui pi-native`

## Notes

- `pi-native` is the highest-fidelity Pi UX mode.
- `pi-tui` remains available as a custom frontend fallback.
- `textual` remains as legacy fallback.
- Current implementation routes each Pi model call to Python using the latest user message.
  Pi session-branch operations (`/tree` navigation/rewind) are not yet fully mirrored into
  Python kernel checkpoint state.
