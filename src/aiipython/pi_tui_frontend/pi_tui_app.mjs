import net from "node:net";
import process from "node:process";
import {
  Box,
  Input,
  Key,
  Markdown,
  ProcessTerminal,
  TUI,
  Text,
  matchesKey,
  truncateToWidth,
  visibleWidth,
  wrapTextWithAnsi,
} from "@mariozechner/pi-tui";

const host = process.env.AIIPYTHON_RPC_HOST || "127.0.0.1";
const port = Number(process.env.AIIPYTHON_RPC_PORT || "0");
if (!port) {
  process.stderr.write("Missing AIIPYTHON_RPC_PORT\n");
  process.exit(2);
}

class RpcClient {
  constructor(host, port) {
    this.host = host;
    this.port = port;
    this.socket = null;
    this.buffer = "";
    this.nextId = 1;
    this.pending = new Map();
    this.onEvent = null;
    this.onClose = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      const socket = net.createConnection({ host: this.host, port: this.port }, () => {
        this.socket = socket;
        resolve();
      });

      socket.setEncoding("utf8");

      socket.on("data", (chunk) => {
        this.buffer += chunk;
        for (;;) {
          const idx = this.buffer.indexOf("\n");
          if (idx < 0) break;
          const line = this.buffer.slice(0, idx);
          this.buffer = this.buffer.slice(idx + 1);
          if (!line.trim()) continue;
          let msg;
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          this._handleMessage(msg);
        }
      });

      socket.on("error", (err) => {
        if (!this.socket) reject(err);
      });

      socket.on("close", () => {
        for (const [, pending] of this.pending) {
          pending.reject(new Error("RPC connection closed"));
        }
        this.pending.clear();
        if (this.onClose) this.onClose();
      });
    });
  }

  close() {
    if (this.socket) {
      try {
        this.socket.end();
      } catch {}
      this.socket = null;
    }
  }

  request(method, params = {}) {
    if (!this.socket) return Promise.reject(new Error("RPC not connected"));

    const id = this.nextId++;
    const payload = {
      type: "request",
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.write(JSON.stringify(payload) + "\n", "utf8", (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  _handleMessage(msg) {
    if (msg?.type === "response") {
      const pending = this.pending.get(msg.id);
      if (!pending) return;
      this.pending.delete(msg.id);
      if (msg.ok) pending.resolve(msg.result ?? {});
      else pending.reject(new Error(msg?.error?.message || "RPC error"));
      return;
    }

    if (msg?.type === "event" && this.onEvent) {
      this.onEvent(msg.event, msg.data || {});
    }
  }
}

// ── Styling (Pi dark theme parity) ────────────────────────────────

const C = {
  cyan: "#00d7ff",
  blue: "#5f87ff",
  green: "#b5bd68",
  red: "#cc6666",
  yellow: "#ffff00",
  gray: "#808080",
  dimGray: "#666666",
  darkGray: "#505050",
  accent: "#8abeb7",
  selectedBg: "#3a3a4a",
  userMsgBg: "#343541",
  toolPendingBg: "#282832",
  toolSuccessBg: "#283228",
  toolErrorBg: "#3c2828",
  customMsgBg: "#2d2838",
  mdHeading: "#f0c674",
  mdLink: "#81a2be",
};

function hexToRgb(hex) {
  const h = String(hex || "").replace(/^#/, "").trim();
  if (h.length !== 6) return [255, 255, 255];
  const r = Number.parseInt(h.slice(0, 2), 16);
  const g = Number.parseInt(h.slice(2, 4), 16);
  const b = Number.parseInt(h.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
  return [r, g, b];
}

function fg(hex, text) {
  if (!hex) return text;
  const [r, g, b] = hexToRgb(hex);
  return `\x1b[38;2;${r};${g};${b}m${text}\x1b[39m`;
}

function bg(hex, text) {
  const [r, g, b] = hexToRgb(hex);
  return `\x1b[48;2;${r};${g};${b}m${text}\x1b[49m`;
}

function bold(text) {
  return `\x1b[1m${text}\x1b[22m`;
}

function dim(text) {
  return `\x1b[2m${text}\x1b[22m`;
}

function italic(text) {
  return `\x1b[3m${text}\x1b[23m`;
}

function markdownTheme() {
  return {
    heading: (t) => bold(fg(C.mdHeading, t)),
    link: (t) => fg(C.mdLink, t),
    linkUrl: (t) => fg(C.dimGray, t),
    code: (t) => fg(C.accent, t),
    codeBlock: (t) => fg(C.green, t),
    codeBlockBorder: (t) => fg(C.gray, t),
    quote: (t) => fg(C.gray, t),
    quoteBorder: (t) => fg(C.gray, t),
    hr: (t) => fg(C.gray, t),
    listBullet: (t) => fg(C.accent, t),
    bold: (t) => bold(t),
    italic: (t) => italic(t),
    strikethrough: (t) => `\x1b[9m${t}\x1b[29m`,
    underline: (t) => `\x1b[4m${t}\x1b[24m`,
  };
}

function shortenMiddle(text, width) {
  if (visibleWidth(text) <= width) return text;
  if (width <= 8) return truncateToWidth(text, width);
  const half = Math.floor((width - 1) / 2);
  const left = text.slice(0, half);
  const right = text.slice(-half + 1);
  return `${left}…${right}`;
}

class MessageEntry {
  constructor(kind, role, component, debugSummary) {
    this.kind = kind;
    this.role = role;
    this.component = component;
    this.debugSummary = debugSummary || "";
  }

  invalidate() {
    if (this.component && this.component.invalidate) this.component.invalidate();
  }

  render(width) {
    if (!this.component || !this.component.render) return [];
    return this.component.render(width);
  }
}

function createUserMessage(content, mdTheme) {
  return new Markdown(content, 1, 1, mdTheme, {
    bgColor: (text) => bg(C.userMsgBg, text),
    color: (text) => text,
  });
}

function createAssistantMessage(content, mdTheme) {
  return new Markdown(content, 1, 0, mdTheme);
}

function createSystemMarkdown(content, mdTheme) {
  return new Markdown(content, 1, 0, mdTheme, {
    color: (text) => fg(C.gray, text),
  });
}

function createSystemText(content) {
  return new Text(fg(C.gray, content), 1, 0);
}

function createErrorText(content) {
  return new Text(fg(C.red, content), 1, 0);
}

function createToolCallComponent(index, code) {
  const box = new Box(1, 1, (text) => bg(C.toolPendingBg, text));
  const title = fg(C.accent, bold(`python #${index}`));
  const body = String(code || "")
    .split("\n")
    .map((line) => fg(C.green, line))
    .join("\n");
  box.addChild(new Text(`${title}\n\n${body}`, 0, 0));
  return box;
}

function createToolResultComponent(index, success, output) {
  const boxBg = success ? C.toolSuccessBg : C.toolErrorBg;
  const box = new Box(1, 1, (text) => bg(boxBg, text));
  const title = success
    ? fg(C.green, bold(`execution #${index} ✓`))
    : fg(C.red, bold(`execution #${index} ✗`));
  const body = String(output || "")
    .split("\n")
    .map((line) => fg(C.gray, line))
    .join("\n");
  box.addChild(new Text(`${title}\n\n${body}`, 0, 0));
  return box;
}

class ChatComponent {
  constructor(tui, rpc) {
    this.tui = tui;
    this.rpc = rpc;
    this.messages = [];
    this.debugLines = [];
    this.wireLines = [];
    this.wireChunks = new Map();
    this.status = {
      model: "",
      prompt_tokens: 0,
      completion_tokens: 0,
      total_cost: 0,
      duration_ms: 0,
      auth_label: "",
      stats_compact: "",
      cwd: process.cwd(),
      busy: false,
    };
    this.queueDepth = 0;
    this.thinking = false;
    this.awaitingPrompt = null;
    this.showInspector = false;
    this.showDebug = false;
    this.showWire = false;
    this.inspector = { snapshot: {}, history: [] };
    this.mdTheme = markdownTheme();

    this.input = new Input();
    this.input.onSubmit = async (value) => {
      const raw = value ?? "";
      if (!raw.trim()) return;
      this.input.setValue("");
      this.tui.requestRender();

      if (this.awaitingPrompt) {
        const token = this.awaitingPrompt.token;
        try {
          await this.rpc.request("provide_prompt", { token, value: raw });
          this.awaitingPrompt = null;
        } catch (err) {
          this._appendError(`Prompt submission failed: ${String(err)}`);
        }
        this.tui.requestRender();
        return;
      }

      try {
        await this.rpc.request("submit_input", { text: raw });
      } catch (err) {
        this._appendError(`Submit failed: ${String(err)}`);
      }
      this.tui.requestRender();
    };
  }

  handleEvent(event, data) {
    switch (event) {
      case "chat_message":
        this._appendChatMessage(data.role || "system", data.content || "", data.format || "text");
        break;
      case "reaction_step":
        this._appendReactionStep(data || {});
        break;
      case "status":
        this.status = { ...this.status, ...data };
        break;
      case "queue":
        this.queueDepth = Number(data.depth || 0);
        break;
      case "thinking":
        this.thinking = Boolean(data.active);
        break;
      case "auth_prompt":
        this.awaitingPrompt = {
          token: data.token,
          message: data.message || "Paste authorization code",
        };
        this._appendSystemMarkdown(`**${this.awaitingPrompt.message}**\n\nPaste code and press Enter.`);
        break;
      case "wire_request": {
        const id = data.id;
        this.wireChunks.set(id, "");
        this.wireLines.push(`▶ #${id} ${data.model || "?"}`);
        break;
      }
      case "wire_chunk": {
        const id = data.id;
        const prev = this.wireChunks.get(id) || "";
        this.wireChunks.set(id, prev + (data.chunk || ""));
        break;
      }
      case "wire_done": {
        const id = data.id;
        const body = this.wireChunks.get(id) || "";
        const usage = data.usage || {};
        this.wireLines.push(
          `◀ #${id} ${Math.round(Number(data.duration_ms || 0))}ms ↑${usage.prompt_tokens || 0} ↓${usage.completion_tokens || 0}`
        );
        if (body) this.wireLines.push(body);
        if (data.error) this.wireLines.push(`ERROR: ${data.error}`);
        this.wireLines.push("─".repeat(72));
        if (this.wireLines.length > 800) this.wireLines = this.wireLines.slice(-800);
        this.wireChunks.delete(id);
        break;
      }
      default:
        break;
    }

    this.tui.requestRender();
  }

  async toggleInspector() {
    this.showInspector = !this.showInspector;
    if (this.showInspector) {
      this.showDebug = false;
      this.showWire = false;
      try {
        this.inspector = await this.rpc.request("get_inspector", {});
      } catch (err) {
        this._appendError(`Inspector failed: ${String(err)}`);
        this.showInspector = false;
      }
    }
    this.tui.requestRender();
  }

  toggleDebug() {
    this.showDebug = !this.showDebug;
    if (this.showDebug) {
      this.showInspector = false;
      this.showWire = false;
    }
    this.tui.requestRender();
  }

  toggleWire() {
    this.showWire = !this.showWire;
    if (this.showWire) {
      this.showInspector = false;
      this.showDebug = false;
    }
    this.tui.requestRender();
  }

  _pushMessage(entry) {
    this.messages.push(entry);
    if (this.messages.length > 800) this.messages = this.messages.slice(-800);

    this.debugLines.push(entry.debugSummary || `[${entry.role}]`);
    if (this.debugLines.length > 800) this.debugLines = this.debugLines.slice(-800);
  }

  _appendUser(content) {
    const comp = createUserMessage(content, this.mdTheme);
    this._pushMessage(new MessageEntry("user", "user", comp, `[user] ${String(content).replace(/\s+/g, " ")}`));
  }

  _appendAssistantMarkdown(content) {
    const comp = createAssistantMessage(content, this.mdTheme);
    this._pushMessage(
      new MessageEntry("assistant", "assistant", comp, `[assistant] ${String(content).replace(/\s+/g, " ")}`)
    );
  }

  _appendSystemMarkdown(content) {
    const comp = createSystemMarkdown(content, this.mdTheme);
    this._pushMessage(new MessageEntry("system-md", "system", comp, `[system] ${String(content).replace(/\s+/g, " ")}`));
  }

  _appendSystemText(content) {
    const comp = createSystemText(content);
    this._pushMessage(new MessageEntry("system", "system", comp, `[system] ${String(content).replace(/\s+/g, " ")}`));
  }

  _appendError(content) {
    const comp = createErrorText(content);
    this._pushMessage(new MessageEntry("error", "error", comp, `[error] ${String(content).replace(/\s+/g, " ")}`));
  }

  _appendToolCall(index, code) {
    const comp = createToolCallComponent(index, code);
    this._pushMessage(new MessageEntry("tool-call", "system", comp, `[tool-call] python #${index}`));
  }

  _appendToolResult(index, success, output) {
    const comp = createToolResultComponent(index, success, output);
    this._pushMessage(new MessageEntry("tool-result", "system", comp, `[tool-result] python #${index} ${success ? "ok" : "err"}`));
  }

  _appendChatMessage(role, content, format) {
    if (role === "user") {
      this._appendUser(content);
      return;
    }
    if (role === "assistant") {
      this._appendAssistantMarkdown(content);
      return;
    }
    if (role === "error") {
      this._appendError(content);
      return;
    }

    if (format === "markdown") this._appendSystemMarkdown(content);
    else this._appendSystemText(content);
  }

  _appendReactionStep(step) {
    const assistantMarkdown = String(step.assistant_markdown || "").trim();
    if (assistantMarkdown) this._appendAssistantMarkdown(assistantMarkdown);

    const blocks = Array.isArray(step.blocks) ? step.blocks : [];
    for (const block of blocks) {
      const index = Number(block.index || 0) || 1;
      this._appendToolCall(index, block.code || "");
      this._appendToolResult(index, Boolean(block.success), block.output || "");
    }
  }

  invalidate() {
    for (const m of this.messages) m.invalidate();
  }

  render(width) {
    const lines = [];

    lines.push(this._headerLine(width));
    lines.push(fg(C.darkGray, "─".repeat(width)));

    if (this.showInspector) lines.push(...this._renderInspector(width));
    else if (this.showWire) lines.push(...this._renderWire(width));
    else if (this.showDebug) lines.push(...this._renderDebug(width));
    else lines.push(...this._renderChat(width));

    if (this.thinking || this.status.busy) {
      lines.push(fg(C.gray, italic("⏳ thinking…")));
    }

    lines.push(fg(C.darkGray, "─".repeat(width)));

    if (this.awaitingPrompt) {
      lines.push(truncateToWidth(fg(C.yellow, "OAuth prompt active: paste code and press Enter"), width));
    } else {
      lines.push(
        truncateToWidth(
          fg(C.dimGray, "Enter send · ctrl+i inspector · ctrl+b debug · ctrl+w wire · ctrl+c quit"),
          width
        )
      );
    }

    lines.push(...this.input.render(width));
    lines.push(...this._footer(width));

    return lines;
  }

  _headerLine(width) {
    const left = `${fg(C.accent, bold("aiipython"))} ${fg(C.dimGray, "pi-tui")}`;
    const mode = this.showInspector ? "INSPECTOR" : this.showWire ? "WIRE" : this.showDebug ? "DEBUG" : "CHAT";
    const rightParts = [fg(C.gray, mode)];
    if (this.queueDepth > 0) rightParts.push(fg(C.yellow, `queue:${this.queueDepth}`));
    if (this.status.busy) rightParts.push(fg(C.green, "busy"));
    const right = rightParts.join(" ");
    return this._twoCol(left, right, width);
  }

  _renderChat(width) {
    const lines = [];
    for (const msg of this.messages) {
      lines.push(...msg.render(width));
      lines.push(" ".repeat(width));
    }
    return lines;
  }

  _renderInspector(width) {
    const lines = [];
    lines.push(truncateToWidth(fg(C.accent, bold("Inspector")), width));
    lines.push(truncateToWidth(fg(C.dimGray, "Namespace"), width));

    const snapshot = this.inspector.snapshot || {};
    const names = Object.keys(snapshot).sort();
    if (names.length === 0) lines.push(truncateToWidth(fg(C.dimGray, "(empty)"), width));
    else {
      for (const name of names.slice(-160)) {
        const summary = String(snapshot[name] ?? "");
        lines.push(...wrapTextWithAnsi(`  ${fg(C.mdHeading, name)}  ${fg(C.gray, summary)}`, width));
      }
    }

    lines.push(" ".repeat(width));
    lines.push(truncateToWidth(fg(C.dimGray, "Recent Activity"), width));
    const hist = Array.isArray(this.inspector.history) ? this.inspector.history : [];
    for (const h of hist.slice(-40)) {
      const tag = h?.tag || "?";
      const code = String(h?.code || "");
      lines.push(...wrapTextWithAnsi(`  [${fg(C.yellow, tag)}] ${fg(C.gray, code)}`, width));
    }
    return lines;
  }

  _renderDebug(width) {
    const lines = [];
    lines.push(truncateToWidth(fg(C.accent, bold("Debug")), width));
    for (const line of this.debugLines.slice(-500)) {
      lines.push(...wrapTextWithAnsi(fg(C.gray, line), width));
    }
    return lines;
  }

  _renderWire(width) {
    const lines = [];
    lines.push(truncateToWidth(fg(C.accent, bold("Wire")), width));
    for (const line of this.wireLines.slice(-700)) {
      lines.push(...wrapTextWithAnsi(line, width));
    }
    return lines;
  }

  _footer(width) {
    const cwd = shortenMiddle(String(this.status.cwd || ""), width);

    const leftParts = [];
    if (this.status.stats_compact) leftParts.push(this.status.stats_compact);
    if (this.queueDepth > 0) leftParts.push(`queue:${this.queueDepth}`);
    const left = fg(C.dimGray, leftParts.join(" · "));

    const rightParts = [];
    if (this.status.auth_label) {
      const auth = String(this.status.auth_label);
      const authColor = auth.toLowerCase().includes("oauth") || auth.toLowerCase().includes("browser") ? C.green : C.mdHeading;
      rightParts.push(fg(authColor, auth));
    }
    if (this.status.model) rightParts.push(fg(C.dimGray, String(this.status.model)));
    const right = rightParts.join(fg(C.dimGray, " · "));

    return [fg(C.dimGray, cwd), this._twoCol(left, right, width)];
  }

  _twoCol(left, right, width) {
    const leftW = visibleWidth(left);
    const rightW = visibleWidth(right);

    if (leftW + rightW + 1 <= width) {
      return left + " ".repeat(width - leftW - rightW) + right;
    }

    if (!right) return truncateToWidth(left, width);
    if (!left) return truncateToWidth(right, width);

    const gap = "  ";
    const half = Math.max(10, Math.floor((width - visibleWidth(gap)) / 2));
    return truncateToWidth(left, half) + gap + truncateToWidth(right, width - half - visibleWidth(gap));
  }
}

const rpc = new RpcClient(host, port);
const terminal = new ProcessTerminal();
const tui = new TUI(terminal, true);
const app = new ChatComponent(tui, rpc);

let shuttingDown = false;

function shutdown(exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  try {
    rpc.close();
  } catch {}
  try {
    tui.stop();
  } catch {}
  process.exit(exitCode);
}

rpc.onEvent = (event, data) => app.handleEvent(event, data);
rpc.onClose = () => shutdown(0);

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

(async () => {
  try {
    await rpc.connect();
    await rpc.request("hello", {});
  } catch (err) {
    process.stderr.write(`Failed to connect backend: ${String(err)}\n`);
    shutdown(2);
    return;
  }

  tui.addChild(app);
  tui.setFocus(app.input);

  tui.addInputListener((data) => {
    if (matchesKey(data, Key.ctrl("c"))) {
      shutdown(0);
      return { consume: true };
    }

    if (matchesKey(data, Key.ctrl("i"))) {
      app.toggleInspector();
      return { consume: true };
    }

    if (matchesKey(data, Key.ctrl("b"))) {
      app.toggleDebug();
      return { consume: true };
    }

    if (matchesKey(data, Key.ctrl("w"))) {
      app.toggleWire();
      return { consume: true };
    }

    return undefined;
  });

  tui.start();
})();
