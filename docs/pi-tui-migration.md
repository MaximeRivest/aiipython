# aiipython Pi-TUI Migration Architecture

See also [`docs/pi-native.md`](pi-native.md) for the full-Pi InteractiveMode path.

This document describes the full frontend migration from Textual (Python) to Pi-TUI (Node.js) while keeping the Python/IPython execution core.

## Goals

Related document: [`docs/pi-lm-gateway.md`](pi-lm-gateway.md) for model/auth transport routing through Pi.

- Keep aiipython's Python backend semantics:
  - live IPython namespace
  - reactive markdown→code-block execution loop
  - checkpoints, context, auth, model switching
- Replace frontend rendering with Pi's terminal UI stack (`@mariozechner/pi-tui`)
- Preserve `chat()` from IPython and `uv run aiipython` CLI entry points
- Make migration incremental and reversible (`--ui textual` fallback)

## High-level topology

```text
+--------------------------+            JSON-RPC over localhost TCP
| Python process           | <--------------------------------------> +-----------------------+
| (aiipython core)         |                                          | Node process          |
|                          |                                          | (pi-tui frontend)     |
| - Kernel (IPython)       |                                          | - TUI rendering       |
| - ReactiveAgent          |                                          | - keyboard handling   |
| - Session/Auth/Context   |                                          | - inspector/wire panes|
| - PiTuiBackend bridge    |                                          | - input prompt UI     |
+--------------------------+                                          +-----------------------+
```

## Implemented modules

### Python

- `src/aiipython/pi_tui_bridge.py`
  - `PiTuiBackend`: command handling + agent orchestration + event emission
  - `PiRpcServer`: local JSON-RPC server for one frontend client
  - `run_pi_tui(session)`: launcher, dependency check, server lifecycle, Node frontend spawn

- `src/aiipython/__init__.py`
  - `chat(model=None, ui=None)` now supports frontend selection
  - default frontend is `pi-tui`
  - `textual` remains available as fallback

- `src/aiipython/cli.py`
  - new `--ui {pi-tui,textual}` flag

### Node

- `src/aiipython/pi_tui_frontend/package.json`
- `src/aiipython/pi_tui_frontend/pi_tui_app.mjs`
  - JSON-RPC client
  - Pi-TUI root component
  - input handling + hotkeys
  - chat/debug/wire/inspector panels

## Runtime lifecycle

1. `chat()` resolves model/auth/session exactly as before.
2. If UI is `pi-tui`:
   - Python starts `PiRpcServer` on `127.0.0.1:<ephemeral>`
   - Python spawns Node frontend process with host/port env vars
3. Frontend connects and sends `hello`.
4. Backend emits welcome + status events.
5. User input is sent via `submit_input` RPC.
6. Backend processes commands or runs reactive agent loop.
7. Backend streams events (`chat_message`, `status`, `thinking`, `wire_*`, etc.).
8. Frontend renders events live.
9. Exiting frontend (`ctrl+c`) terminates Node process; Python server shuts down cleanly.

## JSON-RPC protocol

Transport: newline-delimited JSON over TCP.

### Request envelope

```json
{ "type": "request", "id": 1, "method": "submit_input", "params": { "text": "hello" } }
```

### Response envelope

```json
{ "type": "response", "id": 1, "ok": true, "result": { "accepted": true } }
```

or

```json
{ "type": "response", "id": 1, "ok": false, "error": { "message": "...", "type": "RuntimeError" } }
```

### Event envelope

```json
{ "type": "event", "event": "chat_message", "data": { ... } }
```

### Methods

- `hello`
- `submit_input`
- `provide_prompt` (OAuth prompt response)
- `get_inspector`
- `ping`

### Event types

- `chat_message`
- `reaction_step` (assistant prose + executable block payload)
- `status`
- `queue`
- `thinking`
- `auth_prompt`
- `wire_request`
- `wire_chunk`
- `wire_done`

## Command coverage in backend bridge

Implemented in `PiTuiBackend`:

- `/model`
- `/image`
- `/login` (OAuth + API keys)
- `/logout`
- `/auth`
- `/tabminion`
- `/context`
- `/tree`, `/undo`, `/restore`, `/fork`
- `@file`, `@clip`, `@ + paste` context pinning behavior

## Feature parity strategy

The migration is designed to keep the execution semantics in Python and move UI concerns to Node.

### Preserved

- session/state preservation across `chat()` calls
- reactive execution loop
- checkpoint tree operations
- auth model resolution rules
- context pinning and snapshot modes
- wire logging and token/cost status accumulation

### Frontend-level differences (intentional)

- Pi-TUI layout is now rendered by Node and can evolve independently.
- Message queue behavior is currently sequential queueing (no steer/follow-up split yet).

## Ops and troubleshooting

### Node dependency bootstrap

`run_pi_tui()` checks for `@mariozechner/pi-tui`; if missing it runs:

```bash
npm install --no-audit --no-fund
```

in `src/aiipython/pi_tui_frontend/`.

### Force frontend

- `AIIPYTHON_UI=pi-tui`
- `AIIPYTHON_UI=textual`
- strict mode: `AIIPYTHON_UI_STRICT=1` (do not fallback to Textual)

## Next migration phases

1. **Phase 2:** explicit steer/follow-up queue semantics in RPC API.
2. **Phase 3:** richer tool-exec style rendering (collapsible call/result blocks).
3. **Phase 4:** slash-command selector overlays (`/model`, `/login`) using `SelectList`.
4. **Phase 5:** full session tree navigator UI in frontend.
5. **Phase 6:** optional SDK split for backend API stability and external clients.

## Why this architecture

- Keeps Python as source-of-truth for IPython execution.
- Allows Pi-class frontend ergonomics and extensibility.
- Cleanly separates frontend rendering from backend agent semantics.
- Enables future alternate frontends (web, remote, headless) with same RPC backend.
