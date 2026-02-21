import net from "node:net";
import process from "node:process";

import { Type } from "@sinclair/typebox";
import { calculateCost, createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  InteractiveMode,
  ModelRegistry,
} from "@mariozechner/pi-coding-agent";

const host = process.env.AIIPYTHON_RPC_HOST || "127.0.0.1";
const port = Number(process.env.AIIPYTHON_RPC_PORT || "0");
if (!port) {
  process.stderr.write("Missing AIIPYTHON_RPC_PORT\n");
  process.exit(2);
}

const SYNTHETIC_TOOLCALL_PREFIX = "aiipython-python-cell:";
const SYNTHETIC_BASH_TOOL_NAME = "bash";

class RpcClient {
  constructor(host, port) {
    this.host = host;
    this.port = port;
    this.socket = null;
    this.buffer = "";
    this.nextId = 1;
    this.pending = new Map();
    this.eventListeners = new Set();
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
          this._handle(msg);
        }
      });

      socket.on("error", (err) => {
        if (!this.socket) reject(err);
      });

      socket.on("close", () => {
        for (const [, pending] of this.pending) {
          pending.reject(new Error("Backend RPC connection closed"));
        }
        this.pending.clear();
      });
    });
  }

  close() {
    if (!this.socket) return;
    try {
      this.socket.end();
    } catch {}
    this.socket = null;
  }

  request(method, params = {}) {
    if (!this.socket) return Promise.reject(new Error("RPC not connected"));

    const id = this.nextId++;
    const payload = { type: "request", id, method, params };

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

  addEventListener(listener) {
    if (typeof listener !== "function") return () => {};
    this.eventListeners.add(listener);
    return () => {
      this.eventListeners.delete(listener);
    };
  }

  _handle(msg) {
    if (msg?.type === "event") {
      for (const listener of this.eventListeners) {
        try {
          listener(msg.event, msg.data || {});
        } catch {}
      }
      return;
    }

    if (msg?.type !== "response") return;
    const pending = this.pending.get(msg.id);
    if (!pending) return;
    this.pending.delete(msg.id);
    if (msg.ok) pending.resolve(msg.result ?? {});
    else pending.reject(new Error(msg?.error?.message || "RPC error"));
  }
}

function extractTextFromUserContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content ?? "");

  const parts = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      parts.push(String(block));
      continue;
    }
    if (block.type === "text") {
      parts.push(String(block.text || ""));
      continue;
    }
    if (block.type === "image") {
      parts.push("[image]");
      continue;
    }
  }
  return parts.join("\n");
}

function latestUserText(context) {
  const msgs = Array.isArray(context?.messages) ? context.messages : [];
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m?.role !== "user") continue;
    return extractTextFromUserContent(m.content);
  }
  return "";
}

function getMessages(context) {
  return Array.isArray(context?.messages) ? context.messages : [];
}

function getLastMessage(context) {
  const msgs = getMessages(context);
  return msgs.length > 0 ? msgs[msgs.length - 1] : undefined;
}

function getTrailingSyntheticToolResultIds(context) {
  const msgs = getMessages(context);
  const ids = [];
  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i];
    if (msg?.role !== "toolResult") break;
    const id = String(msg?.toolCallId || "");
    if (id.startsWith(SYNTHETIC_TOOLCALL_PREFIX)) {
      ids.push(id);
    }
  }
  return ids;
}

function makeUsage(promptTokens = 0, completionTokens = 0) {
  return {
    input: promptTokens,
    output: completionTokens,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: promptTokens + completionTokens,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

function makePythonCommand(code) {
  const body = String(code || "").trim();
  if (!body) return "python";
  return `python\n${body}`;
}

function normalizeReactCells(resp) {
  const cells = Array.isArray(resp?.react_cells) ? resp.react_cells : [];
  return cells
    .map((c) => {
      const code = String(c?.code || "").trim();
      const output = String(c?.output || "");
      if (!code && !output) return null;
      return { code, output };
    })
    .filter(Boolean);
}

function makeToolCallId(sessionKey, index) {
  const rand = Math.random().toString(36).slice(2, 10);
  return `${SYNTHETIC_TOOLCALL_PREFIX}${sessionKey}:${Date.now()}:${index}:${rand}`;
}

class PromptAbortError extends Error {
  constructor(message = "Request aborted") {
    super(message);
    this.name = "AbortError";
  }
}

function isAbortLike(err) {
  if (!err) return false;
  if (err instanceof PromptAbortError) return true;
  const name = String(err?.name || "").toLowerCase();
  const msg = String(err?.message || err || "").toLowerCase();
  return name.includes("abort") || msg.includes("aborted") || msg.includes("cancelled");
}

async function requestPromptStreamWithAbort(rpc, payload, signal) {
  if (!signal) {
    return rpc.request("prompt_stream", payload);
  }

  if (signal.aborted) {
    void rpc.request("abort_prompt", {}).catch(() => {});
    throw new PromptAbortError();
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      signal.removeEventListener("abort", onAbort);
    };

    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      void rpc.request("abort_prompt", {}).catch(() => {});
      reject(new PromptAbortError());
    };

    signal.addEventListener("abort", onAbort, { once: true });

    rpc.request("prompt_stream", payload).then(
      (result) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(result);
      },
      (err) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      },
    );
  });
}

function createSyntheticBashTool() {
  return {
    name: SYNTHETIC_BASH_TOOL_NAME,
    label: "python",
    description: "Internal aiipython renderer tool for executed Python cells.",
    parameters: Type.Object({
      command: Type.String(),
      timeout: Type.Optional(Type.Number()),
      __aiipython_output: Type.Optional(Type.String()),
    }),
    async execute(_toolCallId, params) {
      const text = typeof params?.__aiipython_output === "string" ? params.__aiipython_output : "";
      return {
        content: [{ type: "text", text }],
        details: {
          exitCode: 0,
          cancelled: false,
        },
      };
    },
  };
}

function createAiipythonStreamFactory(rpc) {
  const pendingFollowUps = new Map();

  function emitText(stream, output, text) {
    output.content.push({ type: "text", text });
    const index = output.content.length - 1;
    stream.push({ type: "text_start", contentIndex: index, partial: output });
    if (text) {
      stream.push({ type: "text_delta", contentIndex: index, delta: text, partial: output });
    }
    stream.push({ type: "text_end", contentIndex: index, content: text, partial: output });
  }

  function consumePendingFollowUp(context, sessionKey) {
    const pending = pendingFollowUps.get(sessionKey);
    if (!pending) return null;

    const trailingIds = new Set(getTrailingSyntheticToolResultIds(context));
    if (pending.toolCallIds.length > 0) {
      const allDone = pending.toolCallIds.every((id) => trailingIds.has(id));
      if (!allDone) return null;
    }

    pendingFollowUps.delete(sessionKey);
    return pending;
  }

  return function streamSimple(model, context, options = {}) {
    const stream = createAssistantMessageEventStream();

    (async () => {
      const output = {
        role: "assistant",
        content: [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: makeUsage(0, 0),
        stopReason: "stop",
        timestamp: Date.now(),
      };

      try {
        stream.push({ type: "start", partial: output });

        const sessionKey = String(options?.sessionId || "default");

        const pending = consumePendingFollowUp(context, sessionKey);
        if (pending) {
          const text = String(pending.text || "");
          emitText(stream, output, text);
          stream.push({ type: "done", reason: "stop", message: output });
          stream.end(output);
          return;
        }

        const lastMessage = getLastMessage(context);
        if (lastMessage?.role === "toolResult") {
          // Defensive fallback: don't re-run prompt_once on tool-result continuation.
          emitText(stream, output, "");
          stream.push({ type: "done", reason: "stop", message: output });
          stream.end(output);
          return;
        }

        const text = latestUserText(context);
        const modelStr = `${model.provider}/${model.id}`;
        const streamId = `prompt-${Date.now()}-${Math.random().toString(16).slice(2)}`;

        let streamedIndex = -1;
        let streamedStarted = false;
        let streamedText = "";
        let sawAnyDeltas = false;
        let sawStreamedSteps = false;
        let streamedStepBlockCounter = 0;
        let userAborted = false;

        const unsubscribeAbort = options?.signal
          ? (() => {
              const onAbort = () => {
                userAborted = true;
                void rpc.request("abort_prompt", {}).catch(() => {});
              };
              options.signal.addEventListener("abort", onAbort);
              return () => options.signal.removeEventListener("abort", onAbort);
            })()
          : () => {};

        const closeStreamedTextBlock = () => {
          if (!streamedStarted || streamedIndex < 0) return;
          stream.push({ type: "text_end", contentIndex: streamedIndex, content: streamedText, partial: output });
          streamedStarted = false;
          streamedIndex = -1;
          streamedText = "";
        };

        const emitStepTranscript = (index, code, outputText) => {
          const textBlock = [
            `▶ python #${index}`,
            "```python",
            String(code || ""),
            "```",
            "```text",
            String(outputText || ""),
            "```",
          ].join("\n");
          emitText(stream, output, textBlock);
        };

        const unsubscribe = rpc.addEventListener((event, data) => {
          const dataStreamId = String(data?.stream_id || "");
          if (dataStreamId !== streamId) return;

          if (event === "prompt_stream_delta") {
            const delta = String(data?.chunk || "");
            if (!delta) return;
            sawAnyDeltas = true;

            if (!streamedStarted) {
              output.content.push({ type: "text", text: "" });
              streamedIndex = output.content.length - 1;
              stream.push({ type: "text_start", contentIndex: streamedIndex, partial: output });
              streamedStarted = true;
            }

            streamedText += delta;
            const block = output.content[streamedIndex];
            if (block?.type === "text") {
              block.text = streamedText;
            }
            stream.push({ type: "text_delta", contentIndex: streamedIndex, delta, partial: output });
            return;
          }

          if (event === "prompt_stream_step") {
            const blocks = Array.isArray(data?.blocks) ? data.blocks : [];
            if (blocks.length === 0) return;

            sawStreamedSteps = true;
            closeStreamedTextBlock();

            for (const b of blocks) {
              streamedStepBlockCounter += 1;
              emitStepTranscript(
                streamedStepBlockCounter,
                String(b?.code || ""),
                String(b?.output || ""),
              );
            }
          }
        });

        let resp;
        try {
          resp = await requestPromptStreamWithAbort(
            rpc,
            {
              text,
              model: modelStr,
              stream_id: streamId,
            },
            options?.signal,
          );
        } finally {
          unsubscribe();
          unsubscribeAbort();
        }

        closeStreamedTextBlock();

        const promptTokens = Number(resp?.usage?.prompt_tokens || 0);
        const completionTokens = Number(resp?.usage?.completion_tokens || 0);
        output.usage = makeUsage(promptTokens, completionTokens);
        calculateCost(model, output.usage);

        if (resp?.aborted || userAborted || options?.signal?.aborted) {
          throw new PromptAbortError();
        }

        if (sawStreamedSteps) {
          const assistantText = String(resp?.display_markdown || resp?.assistant_markdown || "");
          if (!sawAnyDeltas && assistantText) {
            emitText(stream, output, assistantText);
          }
          stream.push({ type: "done", reason: "stop", message: output });
          stream.end(output);
          return;
        }

        const reactCells = normalizeReactCells(resp);
        if (reactCells.length > 0) {
          const toolCallIds = [];
          for (let i = 0; i < reactCells.length; i++) {
            const cell = reactCells[i];
            const toolCallId = makeToolCallId(sessionKey, i + 1);
            toolCallIds.push(toolCallId);

            const toolCall = {
              type: "toolCall",
              id: toolCallId,
              name: SYNTHETIC_BASH_TOOL_NAME,
              arguments: {},
            };
            output.content.push(toolCall);
            const contentIndex = output.content.length - 1;

            stream.push({ type: "toolcall_start", contentIndex, partial: output });

            const args = {
              command: makePythonCommand(cell.code),
              __aiipython_output: String(cell.output || ""),
            };
            toolCall.arguments = args;
            stream.push({
              type: "toolcall_delta",
              contentIndex,
              delta: JSON.stringify(args),
              partial: output,
            });
            stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
          }

          pendingFollowUps.set(sessionKey, {
            text: String(resp?.assistant_markdown || resp?.display_markdown || ""),
            toolCallIds,
          });

          stream.push({ type: "done", reason: "toolUse", message: output });
          stream.end(output);
          return;
        }

        const assistantText = String(resp?.display_markdown || resp?.assistant_markdown || "");
        if (!sawAnyDeltas) {
          emitText(stream, output, assistantText);
        }

        stream.push({ type: "done", reason: "stop", message: output });
        stream.end(output);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const aborted = isAbortLike(err) || Boolean(options?.signal?.aborted);
        output.stopReason = aborted ? "aborted" : "error";
        output.errorMessage = message;
        stream.push({ type: "error", reason: aborted ? "aborted" : "error", error: output });
        stream.end(output);
      }
    })();

    return stream;
  };
}

function parseModelString(modelStr) {
  if (!modelStr || !modelStr.includes("/")) return null;
  const idx = modelStr.indexOf("/");
  return { provider: modelStr.slice(0, idx), id: modelStr.slice(idx + 1) };
}

function createAiipythonUserBashExtension(rpc) {
  const shorten = (text, max = 140) => {
    const s = String(text || "");
    return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
  };

  return (pi) => {
    pi.on("input", async (event, ctx) => {
      if (event?.source === "extension") {
        return { action: "continue" };
      }

      const originalText = String(event?.text || "");
      if (!originalText.trim()) {
        return { action: "continue" };
      }

      try {
        const transformed = await rpc.request("transform_at_refs", { text: originalText });

        const messages = Array.isArray(transformed?.messages) ? transformed.messages : [];
        for (const msg of messages) {
          const line = String(msg || "").trim();
          if (!line) continue;
          ctx.ui.notify(line, line.startsWith("⚠") ? "warning" : "info");
        }

        if (Boolean(transformed?.handled)) {
          return { action: "handled" };
        }

        const newText = String(transformed?.text ?? originalText);
        if (newText !== originalText) {
          return { action: "transform", text: newText, images: event.images };
        }
      } catch {
        // Ignore transform errors and continue with original input.
      }

      return { action: "continue" };
    });

    pi.registerCommand("vars", {
      description: "Show IPython variables from the bound aiipython session",
      handler: async (args, ctx) => {
        const query = String(args || "").trim().toLowerCase();
        const inspector = await rpc.request("get_inspector", {});
        const snapshot = inspector?.snapshot && typeof inspector.snapshot === "object" ? inspector.snapshot : {};

        let entries = Object.entries(snapshot)
          .map(([name, summary]) => [String(name), String(summary)])
          .sort((a, b) => a[0].localeCompare(b[0]));

        if (query) {
          entries = entries.filter(([name, summary]) => {
            const n = name.toLowerCase();
            const s = summary.toLowerCase();
            return n.includes(query) || s.includes(query);
          });
        }

        if (entries.length === 0) {
          ctx.ui.notify(query ? `No variables matching '${query}'` : "No user variables found", "info");
          return;
        }

        const maxItems = 200;
        const options = entries.slice(0, maxItems).map(([name, summary]) => `${name} — ${shorten(summary)}`);
        if (entries.length > maxItems) {
          options.push(`… ${entries.length - maxItems} more (use /vars <filter>)`);
        }

        await ctx.ui.select(
          `IPython variables${query ? ` (filter: ${query})` : ""} • ${entries.length}`,
          options,
        );
      },
    });

    pi.on("user_bash", async (event) => {
      const command = String(event?.command || "").trim();
      if (!command) {
        return {
          result: {
            output: "",
            exitCode: 0,
            cancelled: false,
            truncated: false,
          },
        };
      }

      try {
        const resp = await rpc.request("run_ipython_shell", {
          command,
          excludeFromContext: Boolean(event?.excludeFromContext),
        });

        return {
          result: {
            output: String(resp?.output || ""),
            exitCode: Number(resp?.exit_code ?? 0),
            cancelled: Boolean(resp?.cancelled),
            truncated: Boolean(resp?.truncated),
            fullOutputPath: resp?.full_output_path ? String(resp.full_output_path) : undefined,
          },
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          result: {
            output: `IPython shell bridge error: ${message}`,
            exitCode: 1,
            cancelled: false,
            truncated: false,
          },
        };
      }
    });
  };
}

(async () => {
  const rpc = new RpcClient(host, port);
  await rpc.connect();

  const authStorage = new AuthStorage();
  const modelRegistry = new ModelRegistry(authStorage);

  const streamSimple = createAiipythonStreamFactory(rpc);
  const models = modelRegistry.getAll();
  const providerApis = new Map();
  for (const m of models) {
    if (!providerApis.has(m.provider)) providerApis.set(m.provider, new Set());
    providerApis.get(m.provider).add(m.api);
  }

  for (const [provider, apis] of providerApis.entries()) {
    for (const api of apis) {
      modelRegistry.registerProvider(provider, { api, streamSimple });
    }
  }

  let selectedModel;
  const desired = process.env.AIIPYTHON_MODEL || process.env.PYCODE_MODEL;
  const parsed = parseModelString(desired || "");
  if (parsed) {
    selectedModel = modelRegistry.find(parsed.provider, parsed.id);
  }

  const resourceLoader = new DefaultResourceLoader({
    extensionFactories: [createAiipythonUserBashExtension(rpc)],
  });
  await resourceLoader.reload();

  const { session, modelFallbackMessage } = await createAgentSession({
    authStorage,
    modelRegistry,
    model: selectedModel,
    tools: [],
    customTools: [createSyntheticBashTool()],
    resourceLoader,
  });

  const mode = new InteractiveMode(session, {
    modelFallbackMessage,
  });

  try {
    await mode.run();
  } finally {
    try {
      mode.stop();
    } catch {}
    try {
      session.dispose();
    } catch {}
    rpc.close();
  }
})();
