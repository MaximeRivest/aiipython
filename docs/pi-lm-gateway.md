# aiipython Pi LM Gateway

This document describes the new LM backend bridge that lets DSPy calls route through Pi's auth/model stack.

## Why

See also [`docs/pi-native.md`](pi-native.md) for full Pi InteractiveMode frontend integration.

aiipython's Python execution paradigm remains in place, but model transport can now use:

- Pi's `AuthStorage` (OAuth/API-key resolution)
- Pi's `ModelRegistry` (provider/model metadata)
- Pi AI transport (`@mariozechner/pi-ai`)

This avoids reimplementing provider auth/model logic in Python.

## Components

## Python side

- `src/aiipython/pi_gateway_client.py`
  - starts/stops Node gateway process on demand
  - ensures npm dependencies are installed
  - exposes HTTP calls (`health`, `list_models`, `auth_status`, `complete`)

- `src/aiipython/pi_gateway_lm.py`
  - `PiGatewayLM(dspy.LM)` adapter
  - converts DSPy calls to gateway `/lm/complete`
  - maps usage back into `litellm.ModelResponse`

- `src/aiipython/lm_factory.py`
  - backend selection (`auto`, `pi`, `litellm`)
  - keeps `tabminion/*` on legacy StreamingLM path

## Node side

- `src/aiipython/pi_gateway/pi_gateway_server.mjs`
  - local HTTP server (localhost ephemeral port)
  - endpoints:
    - `GET /health`
    - `GET /models`
    - `GET /auth/status`
    - `POST /lm/complete`

- `src/aiipython/pi_gateway/package.json`
  - dependencies:
    - `@mariozechner/pi-ai`
    - `@mariozechner/pi-coding-agent`

## Backend selection

Set with `AIIPYTHON_LM_BACKEND` or CLI `--lm-backend`:

- `auto` (default): try Pi gateway first; fallback to legacy litellm transport
- `pi`: require Pi gateway
- `litellm`: force legacy Python transport (`StreamingLM`)

## Data flow

1. DSPy calls `LM.forward(...)`
2. `PiGatewayLM` posts `messages + model + options` to `/lm/complete`
3. Gateway resolves model via `ModelRegistry`
4. Gateway resolves auth via `AuthStorage`/env
5. Gateway calls `completeSimple(...)`
6. Gateway returns text + usage
7. `PiGatewayLM` returns a `litellm.ModelResponse`-compatible object to DSPy

## Notes

- This phase focuses on LM routing + model/auth discovery.
- aiipython command handlers (`/login`, `/auth`) still use aiipython's Python auth manager today.
- Full unification with Pi's interactive command UX is the next migration phase.
