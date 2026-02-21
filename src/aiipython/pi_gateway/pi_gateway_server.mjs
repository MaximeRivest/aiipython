import http from "node:http";
import process from "node:process";

import { completeSimple } from "@mariozechner/pi-ai";
import { AuthStorage, ModelRegistry } from "@mariozechner/pi-coding-agent";

const host = process.env.AIIPYTHON_PI_GATEWAY_HOST || "127.0.0.1";
const port = Number(process.env.AIIPYTHON_PI_GATEWAY_PORT || "0");

const authStorage = new AuthStorage();
const modelRegistry = new ModelRegistry(authStorage);

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.statusCode = status;
  res.setHeader("content-type", "application/json; charset=utf-8");
  res.setHeader("content-length", Buffer.byteLength(body));
  res.end(body);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 2_000_000) {
        reject(new Error("Request body too large"));
      }
    });
    req.on("end", () => {
      if (!data) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(data));
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

function parseModelString(modelStr) {
  const raw = String(modelStr || "").trim();
  if (!raw) return { provider: "", id: "" };
  if (raw.includes("/")) {
    const idx = raw.indexOf("/");
    return { provider: raw.slice(0, idx), id: raw.slice(idx + 1) };
  }
  return { provider: "", id: raw };
}

function zeroUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

function extractText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content ?? "");

  const parts = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      parts.push(String(block));
      continue;
    }
    if (block.type === "text") parts.push(String(block.text || ""));
    else if (block.type === "image_url") parts.push("[image]");
    else parts.push(String(block.text || ""));
  }
  return parts.join("\n");
}

function convertUserContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content ?? "");

  const out = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      out.push({ type: "text", text: String(block) });
      continue;
    }

    if (block.type === "text") {
      out.push({ type: "text", text: String(block.text || "") });
      continue;
    }

    if (block.type === "image_url") {
      const url = block.image_url?.url;
      if (typeof url === "string" && url.startsWith("data:")) {
        const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
        if (m) {
          out.push({ type: "image", mimeType: m[1], data: m[2] });
          continue;
        }
      }
    }

    out.push({ type: "text", text: extractText([block]) });
  }

  return out.length === 1 && out[0].type === "text" ? out[0].text : out;
}

function convertOpenAIMessagesToContext(messages, model) {
  const systemParts = [];
  const contextMessages = [];

  for (const msg of Array.isArray(messages) ? messages : []) {
    const role = String(msg?.role || "user");
    const content = msg?.content;

    if (role === "system" || role === "developer") {
      const text = extractText(content).trim();
      if (text) systemParts.push(text);
      continue;
    }

    if (role === "assistant") {
      const text = extractText(content);
      contextMessages.push({
        role: "assistant",
        content: [{ type: "text", text }],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: zeroUsage(),
        stopReason: "stop",
        timestamp: Date.now(),
      });
      continue;
    }

    if (role === "tool") {
      continue;
    }

    contextMessages.push({
      role: "user",
      content: convertUserContent(content),
      timestamp: Date.now(),
    });
  }

  return {
    systemPrompt: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
    messages: contextMessages,
  };
}

function flattenAssistantText(message) {
  const blocks = Array.isArray(message?.content) ? message.content : [];
  const textParts = [];
  const thinkingParts = [];

  for (const block of blocks) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text" && block.text) textParts.push(String(block.text));
    if (block.type === "thinking" && block.thinking) thinkingParts.push(String(block.thinking));
  }

  return {
    text: textParts.join(""),
    thinking: thinkingParts.join("\n\n"),
  };
}

function getModelListPayload() {
  modelRegistry.refresh();
  const all = modelRegistry.getAll();
  const availableSet = new Set(modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`));

  return all.map((m) => ({
    provider: m.provider,
    id: m.id,
    name: m.name,
    api: m.api,
    reasoning: Boolean(m.reasoning),
    input: m.input,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    available: availableSet.has(`${m.provider}/${m.id}`),
    usingOAuth: modelRegistry.isUsingOAuth(m),
  }));
}

async function resolveModel(modelStr) {
  modelRegistry.refresh();

  const parsed = parseModelString(modelStr);
  let model;
  if (parsed.provider && parsed.id) {
    model = modelRegistry.find(parsed.provider, parsed.id);
  } else {
    const byId = modelRegistry.getAll().filter((m) => m.id === parsed.id || `${m.provider}/${m.id}` === parsed.id);
    model = byId[0];
  }

  if (!model) {
    throw new Error(`Unknown model: ${modelStr}`);
  }

  return model;
}

async function handleLmComplete(req, res) {
  const body = await readJson(req);
  const modelStr = String(body.model || "");
  if (!modelStr) {
    sendJson(res, 400, { error: "model is required" });
    return;
  }

  const model = await resolveModel(modelStr);
  const context = convertOpenAIMessagesToContext(body.messages || [], model);

  const apiKey =
    typeof body.apiKey === "string" && body.apiKey
      ? body.apiKey
      : await modelRegistry.getApiKey(model);

  const options = {};
  if (apiKey) options.apiKey = apiKey;

  const temperature = body?.options?.temperature;
  if (typeof temperature === "number") options.temperature = temperature;

  const maxTokens = body?.options?.max_tokens ?? body?.options?.max_output_tokens;
  if (typeof maxTokens === "number") options.maxTokens = maxTokens;

  const thinking = body?.options?.thinking;
  if (typeof thinking === "string" && thinking && thinking !== "off") {
    options.reasoning = thinking;
  }

  const started = Date.now();
  const message = await completeSimple(model, context, options);
  const durationMs = Date.now() - started;

  const flat = flattenAssistantText(message);
  const usage = message.usage || zeroUsage();

  sendJson(res, 200, {
    model: `${model.provider}/${model.id}`,
    stopReason: message.stopReason,
    text: flat.text,
    thinking: flat.thinking,
    usage: {
      input: usage.input || 0,
      output: usage.output || 0,
      cacheRead: usage.cacheRead || 0,
      cacheWrite: usage.cacheWrite || 0,
      totalTokens: usage.totalTokens || (usage.input || 0) + (usage.output || 0),
      cost: usage.cost || zeroUsage().cost,
    },
    durationMs,
    raw: message,
  });
}

async function handleAuthStatus(_req, res) {
  modelRegistry.refresh();

  const providers = new Set(modelRegistry.getAll().map((m) => m.provider));
  const rows = [];
  for (const provider of Array.from(providers).sort()) {
    const cred = authStorage.get(provider);
    rows.push({
      provider,
      hasAuth: authStorage.hasAuth(provider),
      source:
        cred?.type === "oauth"
          ? "oauth"
          : cred?.type === "api_key"
            ? "api_key"
            : process.env[`${provider.toUpperCase().replace(/[^A-Z0-9]/g, "_")}_API_KEY`]
              ? "env"
              : "none",
      type: cred?.type || "none",
    });
  }

  sendJson(res, 200, {
    providers: rows,
    oauthProviders: authStorage.getOAuthProviders().map((p) => p.id),
  });
}

const server = http.createServer(async (req, res) => {
  try {
    const method = req.method || "GET";
    const url = req.url || "/";

    if (method === "GET" && url === "/health") {
      sendJson(res, 200, { ok: true, pid: process.pid });
      return;
    }

    if (method === "GET" && url === "/models") {
      sendJson(res, 200, { models: getModelListPayload() });
      return;
    }

    if (method === "GET" && url === "/auth/status") {
      await handleAuthStatus(req, res);
      return;
    }

    if (method === "POST" && url === "/lm/complete") {
      await handleLmComplete(req, res);
      return;
    }

    sendJson(res, 404, { error: "Not found" });
  } catch (err) {
    sendJson(res, 500, { error: String(err?.message || err), type: err?.name || "Error" });
  }
});

server.listen(port, host, () => {
  const addr = server.address();
  const actualPort = typeof addr === "object" && addr ? addr.port : port;
  process.stdout.write(`AIIPYTHON_PI_GATEWAY_READY ${host}:${actualPort}\n`);
});

process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
});
