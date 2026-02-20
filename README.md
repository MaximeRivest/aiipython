# aiipython

A reactive AI chat assistant running inside IPython.

## Quick start

```bash
# From the terminal
uv run aiipython
uv run aiipython -m openai/gpt-4o-mini

# From IPython — your namespace is preserved
$ ipython
In [1]: import pandas as pd
In [2]: df = pd.read_csv("data.csv")
In [3]: from aiipython import chat
In [4]: chat()                          # TUI sees df, pd, etc.
        # ... ctrl+c to exit ...
In [5]: df.describe()                   # back in IPython
In [6]: chat()                          # re-enter, state preserved
In [7]: chat("openai/gpt-4o-mini")     # switch model
```

## Commands

| Command | Description |
|---|---|
| `/model` | Interactive model picker menu (live discovery + recents + TabMinion) |
| `/model refresh` | Force-refresh live model discovery |
| `/model <provider/name>` | Switch model directly |
| `/image <path>` | Load image into `images` dict |
| `/login` | Interactive login provider menu |
| `/login anthropic` | OAuth login (Claude Pro/Max subscription) |
| `/login openai` | OAuth login (ChatGPT Plus/Pro via Codex OAuth; Pi-compatible token import/login) |
| `/login <provider> <api_key>` | Persist API key in `~/.aiipython/auth.json` (e.g. `openai`, `gemini`) |
| `/logout <provider>` | Remove stored credentials |
| `/auth` | Show auth sources and what's active |
| `/tabminion` | Show TabMinion status and available browser AIs |
| `/tree` | Show checkpoint tree |
| `/undo` | Revert AI's last turn (restore pre-turn snapshot) |
| `/restore <id>` | Jump to any checkpoint by id |
| `/fork [label]` | Mark a named branch point |

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

Sub-agents created with `spawn_agent()` receive a full clone of the
parent's namespace — they can explore independently without affecting
the main session.

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

## Remembered Model & Menus

aiipython remembers your last selected model in `~/.aiipython/settings.json`.
On next launch, default resolution is:

1. `-m/--model` argument
2. remembered `last_model`
3. `AIIPYTHON_MODEL` environment variable
4. fallback `gemini/gemini-3-flash-preview`

`/model` and `/login` both open interactive numbered menus. Type a number,
paste a value directly, or `q` to cancel.

`/model` also performs live, auth-aware discovery (Anthropic/OpenAI/Gemini)
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

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AIIPYTHON_MODEL` | `gemini/gemini-3-flash-preview` | Default LM (used when no remembered model) |
| `GEMINI_API_KEY` | — | Gemini key |
| `OPENAI_API_KEY` | — | OpenAI key |
| `ANTHROPIC_API_KEY` | — | Anthropic key |
| `AIIPYTHON_INLINE` | `1` | Run Textual inline mode (`1/true/yes`) to preserve terminal-native transparency. Set `0/false/no` to force fullscreen alternate-screen mode |
| `AIIPYTHON_BG` | auto-detected | Background color strategy. Accepts `ansi_default` or `#RRGGBB`. If unset, aiipython queries terminal background (OSC 11) and uses that color |
