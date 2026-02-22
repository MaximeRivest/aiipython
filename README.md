# aiipython

A reactive AI chat assistant running inside IPython.

## Quick start

```bash
# Default frontend is pi-native (real Pi InteractiveMode).
# Node.js + npm must be installed (deps auto-install on first run).

# From the terminal
uv run aiipython
uv run aiipython -m openai/gpt-4o-mini
# (after publish/install) uvx aiipython

# Optional: enable MLflow DSPy tracing with defaults
uv sync --extra mlflow
uv run aiipython --mlflow

# Optional: force LM routing through Pi model/auth stack
AIIPYTHON_LM_BACKEND=pi uv run aiipython

# Optional: force custom pi-tui frontend
AIIPYTHON_UI=pi-tui uv run aiipython

# Optional: force legacy Textual frontend
AIIPYTHON_UI=textual uv run aiipython

# From IPython — your namespace is preserved
$ ipython
In [1]: import pandas as pd
In [2]: df = pd.read_csv("data.csv")
In [3]: from aiipython import chat
In [4]: chat()                          # pi-native (real Pi UI) sees df, pd, etc.
        # ... ctrl+c to exit ...
In [5]: df.describe()                   # back in IPython
In [6]: chat()                          # re-enter, state preserved
In [7]: chat("openai/gpt-4o-mini")     # switch model
In [8]: chat(ui="textual")             # optional legacy frontend
In [9]: chat(lm_backend="pi")          # force Pi model/auth gateway
```

## Frontends

aiipython now supports three frontends:

- **pi-native (default)** — real Pi `InteractiveMode` UI from `@mariozechner/pi-coding-agent`, wired to aiipython backend semantics.
- **pi-tui** — custom Node.js UI based on `@mariozechner/pi-tui`.
- **textual (legacy)** — previous pure-Python Textual UI.

Select frontend with:

- Environment: `AIIPYTHON_UI=pi-native|pi-tui|textual`
- Python API: `chat(ui="pi-native")`, `chat(ui="pi-tui")`, or `chat(ui="textual")`
- CLI: `uv run aiipython --ui pi-native|pi-tui|textual`
- LM backend override: `chat(lm_backend="pi")` or `uv run aiipython --lm-backend pi`

Architecture details: [`docs/pi-tui-migration.md`](docs/pi-tui-migration.md)

- Full Pi mode: [`docs/pi-native.md`](docs/pi-native.md)
- LM routing via Pi stack: [`docs/pi-lm-gateway.md`](docs/pi-lm-gateway.md)

`pi-native` is the mode that gives full Pi UX parity (autocomplete, history navigation,
interactive selectors, queue semantics, footer pricing/status rendering).

Backend-state mirroring for Pi branch navigation (`/tree` rewinds) is still being hardened;
see [`docs/pi-native.md`](docs/pi-native.md).

## LM Backends

aiipython supports three LM backend modes (`AIIPYTHON_LM_BACKEND`):

- `auto` (default): try Pi gateway first, fall back to litellm backend
- `pi`: force Node Pi gateway (`@mariozechner/pi-ai` + `ModelRegistry` + `AuthStorage`)
- `litellm`: force legacy Python litellm transport (`StreamingLM`)

CLI equivalent:

```bash
uv run aiipython --lm-backend auto|pi|litellm
```

> Note: this phase routes LM transport via Pi. aiipython `/login` and `/auth`
> commands are still backed by aiipython's Python auth manager.

Pi gateway architecture details: [`docs/pi-lm-gateway.md`](docs/pi-lm-gateway.md)

## Commands

| Command | Description |
|---|---|
| `/model` | Show discovered models (live discovery + recents + TabMinion) |
| `/model refresh` | Force-refresh live model discovery |
| `/model <provider/name>` | Switch model directly |
| `/image <path>` | Load image into `images` dict |
| `/login` | Show login usage/options |
| `/login anthropic` | OAuth login (Claude Pro/Max subscription) |
| `/login openai` | OAuth login (ChatGPT Plus/Pro via Codex OAuth; Pi-compatible token import/login) |
| `/login <provider> <api_key>` | Persist API key in `~/.aiipython/auth.json` (e.g. `openai`, `gemini`) |
| `/logout <provider>` | Remove stored credentials |
| `/auth` | Show auth sources and what's active |
| `/tabminion` | Show TabMinion status and available browser AIs |
| `/mlflow` | Start MLflow UI, auto-enable tracing, and open browser |
| `/mlflow status` | Show MLflow tracing + UI status |
| `/mlflow stop` | Stop the MLflow UI process |
| `/tree` | Show checkpoint tree |
| `/undo` | Revert AI's last turn (restore pre-turn snapshot) |
| `/restore <id>` | Jump to any checkpoint by id |
| `/fork [label]` | Mark a named branch point |

### Input shortcuts

- `!command` — run as IPython input in the bound aiipython session
- `!!command` — same as `!command` (also routed as IPython input)
- Model code fences:
  - ` ```py ` (or ` ```python `) runs
  - ` ```#py ` is reference-only (displayed, not executed)

## Checkpoints

Every AI turn is automatically snapshot'd before execution.  This
lets you freely explore, undo mistakes, and branch into alternatives.

```
Checkpoint tree:
└── [0000] ● pre: analyze the data  (14:30:01)
    ├── [0001] ○ pre: make a bar chart  (14:30:45)
    └── [0002] ○ pre: try a scatter plot instead  (14:31:12)
```

From IPython you can also use magics: `%tree`, `%undo`, `%restore 0003`,
`%fork my experiment`.

Sub-agents created with `spawn_agent()` now run in a **separate Python
process** and receive a full clone of the parent's namespace. They can
explore independently without affecting the main session. Keep the
returned proxy (for example `child`) and call `child.react()` /
`child.ask("next task")` to continue that same child statefully.

Checkpoints are stored in `.aiipython_checkpoints/` (git-ignored by default).
Large namespaces (big DataFrames, etc.) will produce proportionally larger
checkpoint files.

## Keybindings

| Key | Action |
|---|---|
| `ctrl+i` | Toggle inspector (namespace + REPL) |
| `ctrl+b` | Toggle debug (raw AI markdown) |
| `ctrl+w` | Toggle wire (raw API requests/responses) |
| `ctrl+c` | Quit (back to REPL) |

## Authentication

aiipython resolves API keys with a clear priority:

1. **OAuth token** (`~/.aiipython/auth.json`) — subscription, free at point of use
2. **API key from auth.json** — explicit file-based key
3. **Environment variable** — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.

This ensures your subscription is used when available, preventing surprise
costs from billed API keys in your environment.

```
/login anthropic             # OAuth login → Claude Pro/Max subscription
/login openai                # OAuth login → ChatGPT Plus/Pro (Codex)
/login openai sk-...         # (optional) Persist OpenAI API key in auth.json
/login gemini AIza...        # Persist Gemini key in auth.json
/auth                        # Show what's active and where each key comes from
/logout openai               # Remove stored OpenAI/Codex credentials
```

The footer always shows the active auth source so you know what you're
using. Credentials are stored in `~/.aiipython/auth.json` with `0600`
permissions.

On first use, aiipython also auto-imports compatible credentials from
`~/.codex/auth.json` and `~/.pi/agent/auth.json` (when present), so your
existing Codex/Pi OAuth logins can be reused.

Pi's `openai-codex/*` models use a custom ChatGPT backend transport
(`chatgpt.com/backend-api`) that is different from standard OpenAI Platform
`openai/*` API calls. aiipython now supports both paths:

- `openai-codex/*` → ChatGPT Plus/Pro OAuth subscription transport
- `openai/*` → OpenAI Platform API key transport

Example:

```
/model openai-codex/gpt-5.3-codex
```

## Remembered Model

aiipython remembers your last selected model in `~/.aiipython/settings.json`.
On next launch, default resolution is:

1. `-m/--model` argument
2. remembered `last_model`
3. `AIIPYTHON_MODEL` environment variable
4. fallback `gemini/gemini-3-flash-preview`

`/model` performs live, auth-aware discovery (Anthropic/OpenAI/Gemini)
plus TabMinion services when available, then caches results briefly for speed.
Use `/model refresh` to force a rescan.

## TabMinion (Browser Subscriptions)

If [TabMinion](https://github.com/…/tabminion) is running, you can use
your browser AI subscriptions (ChatGPT Plus, Claude Pro, Grok, Gemini)
as free LM backends — no API keys needed:

```
/tabminion                  # show status and available browser tabs
/model tabminion/claude     # switch to Claude via browser
/model tabminion/chatgpt    # switch to ChatGPT via browser
```

TabMinion drives the actual browser UI through a Firefox/Zen extension,
so it uses your existing subscription. The footer shows
`🔑 tabminion (browser)` when active.

When using `tabminion/*` models, aiipython now sends
`conversation_mode="new"` on every call (fresh browser conversation per turn)
for more deterministic runs and less context bleed from prior chats.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AIIPYTHON_MODEL` | `gemini/gemini-3-flash-preview` | Default LM (used when no remembered model) |
| `AIIPYTHON_UI` | `pi-native` | Frontend selection: `pi-native`, `pi-tui`, or `textual` |
| `AIIPYTHON_UI_STRICT` | `0` | If `1`, fail instead of falling back to Textual when pi-tui startup fails |
| `AIIPYTHON_LM_BACKEND` | `auto` | LM routing: `auto`, `pi`, or `litellm` |
| `AIIPYTHON_MLFLOW` | `0` | Set to `1` to enable `mlflow.dspy.autolog()` tracing |
| `AIIPYTHON_MLFLOW_TRACKING_URI` | `sqlite:///$HOME/.aiipython/mlflow.db` | Optional MLflow tracking URI override |
| `AIIPYTHON_MLFLOW_EXPERIMENT` | `aiipython` | Optional MLflow experiment name override |
| `AIIPYTHON_MLFLOW_SILENT` | `0` | If `1`, suppress MLflow autologging warnings/events |
| `AIIPYTHON_MLFLOW_AUTO_UI` | `1` | If `AIIPYTHON_MLFLOW=1`, auto-start MLflow UI on launch |
| `AIIPYTHON_MLFLOW_OPEN_BROWSER` | `1` | Auto-open MLflow UI URL in your browser when starting UI |
| `GEMINI_API_KEY` | — | Gemini key |
| `OPENAI_API_KEY` | — | OpenAI key |
| `ANTHROPIC_API_KEY` | — | Anthropic key |
| `TABMINION_WINDOW_ID` | — | Optional: force TabMinion OpenAI proxy to use a specific browser window ID |
| `TABMINION_ACTIVATE` | `1` | Optional: set `0` to avoid tab activation/focus-stealing in TabMinion proxy |
| `AIIPYTHON_INLINE` | `1` | Textual-only: run inline mode (`1/true/yes`) to preserve terminal-native transparency. Set `0/false/no` to force fullscreen alternate-screen mode |
| `AIIPYTHON_BG` | auto-detected | Textual-only: background color strategy. Accepts `ansi_default` or `#RRGGBB` |

To enable MLflow tracing with defaults:

```bash
uv sync --extra mlflow
uv run aiipython --mlflow
```

With `--mlflow`, aiipython now auto-starts the MLflow UI and attempts to open your browser.
Inside pi-native, you can also run `/mlflow` (or `/mlflow --no-open`) and `/mlflow status`.
(Equivalent to `AIIPYTHON_MLFLOW=1 uv run aiipython`.)
