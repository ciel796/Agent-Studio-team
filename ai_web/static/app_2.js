const state = {
  page: "chat",
  toolset: "basic_tools",
  autoFromCode: false,
  toolsSchema: [],
  modelProfiles: [],
  defaultModelProfile: null,
  memoryIndex: {},
  latestRun: null,
  memSearchMode: "keyword",
  chatHistory: [],
  conversationId: null,
  processUpdates: [],
  activeRunDir: null,
  pollTimer: null,
  pollIntervalMs: 0,
  isRunning: false,
  selectedModelProfile: null,
  stopRequested: false,
  llmRuntimeAvailable: true,
  missingLlmDependencies: [],
  modelWarmup: null,
};

const CHAT_HISTORY_KEY = "agent-studio-chat-history";
const CONVERSATION_ID_KEY = "agent-studio-conversation-id";
const DEFAULT_PROGRESS_POLL_INTERVAL_MS = 800;
const FINAL_STAGE_PROGRESS_POLL_INTERVAL_MS = 250;
const AGENT_MODE_LABELS = {
  adaptive_execute: "执行方式：智能执行",
  plan_execute: "执行方式：先规划再执行",
  integrated: "执行方式：普通对话模式",
  react_one_round: "执行方式：单轮工具模式",
};

const AGENT_MODE_HELP = {
  adaptive_execute: "会根据问题复杂度自动决定是否先规划、是否调用工具，适合想省心时使用。",
  plan_execute: "先列步骤，再按步骤完成，整体更稳，速度也不错，适合大多数需要工具和过程控制的任务。",
  integrated: "像普通聊天一样直接回答，需要时再调用工具，适合简单任务。",
  react_one_round: "只做一轮工具决策，适合快速验证单次工具调用。",
};

const LLM_MODE_LABELS = {
  adaptive: "输出方式：自动判断",
  prompt_json: "输出方式：结构化输出",
  native_tools: "输出方式：原生工具调用",
  mock: "输出方式：模拟模式",
};

const LLM_MODE_HELP = {
  adaptive: "系统会按执行方式自动选择兼容模式：普通对话/单轮工具使用原生工具，先规划再执行使用结构化输出，智能执行使用原生自适应路由。",
  prompt_json: "让模型按固定 JSON 结构输出，解析更稳，适合和先规划再执行搭配使用。",
  native_tools: "让模型直接生成工具调用格式，速度快，但更依赖模型遵循格式的能力。",
  mock: "不真正调用模型，主要用于演示和联调界面。",
};

const SAVE_MEMORY_HELP = {
  none: "本次对话结束后，不会写入记忆库。",
  conversation: "只保存到当前会话相关的记忆里，便于后续续聊。",
  global: "保存到全局记忆，后续其他相关任务也可能用到。",
};

function deriveModelProfilesFromConfig(config) {
  if (!config || typeof config !== "object") return [];
  const pool = config.model_pool && typeof config.model_pool === "object" ? config.model_pool : {};
  const evaluation = config.evaluation && typeof config.evaluation === "object" ? config.evaluation : {};
  const preferred = Array.isArray(evaluation.default_profiles)
    ? evaluation.default_profiles.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const reserved = new Set(["default", "planner", "execute", "executor"]);

  const orderedIds = [];
  preferred.forEach((id) => {
    if (pool[id] && !orderedIds.includes(id)) orderedIds.push(id);
  });
  Object.keys(pool).forEach((id) => {
    if (reserved.has(id)) return;
    if (!orderedIds.includes(id)) orderedIds.push(id);
  });
  if (!orderedIds.length) {
    Object.keys(pool).forEach((id) => {
      if (!orderedIds.includes(id)) orderedIds.push(id);
    });
  }

  return orderedIds.map((id) => {
    const item = pool[id] && typeof pool[id] === "object" ? pool[id] : {};
    const target = String(item.model_name_or_path || item.tokenizer_name_or_path || "").trim();
    const modelName = target ? target.split(/[\\/]/).pop() : id;
    return {
      id,
      label: id,
      model_name: modelName || id,
      target,
    };
  });
}

function formatModelDisplayName(profile) {
  if (!profile || typeof profile !== "object") return "local";
  const target = String(profile.target || "");
  const rawName = String(profile.model_name || profile.label || profile.id || "local");
  const normalized = target || rawName;
  if (/qwen3\.5-4b/i.test(normalized)) return "Qwen3.5 4B";
  if (/qwen2\.5-7b/i.test(normalized)) return "Qwen2.5 7B";
  return rawName;
}

function selectedModelOption() {
  const select = el("chat-model-profile");
  if (!select || select.selectedIndex < 0) return null;
  return select.options[select.selectedIndex] || null;
}

function syncModelProfileUi() {
  const option = selectedModelOption();
  let modelName = "local";
  let help = "选择模型后会自动使用对应配置，用户不需要调整 yaml。";
  if (option) {
    modelName = option.dataset.modelName || option.textContent || option.value || "local";
    state.selectedModelProfile = option.value || null;
    help = option.dataset.help || `当前使用 ${modelName}，切换后会自动匹配对应配置。`;
  } else {
    const fallbackId = state.selectedModelProfile || state.defaultModelProfile || (state.modelProfiles[0] && state.modelProfiles[0].id) || null;
    const fallback = state.modelProfiles.find((item) => item && item.id === fallbackId) || state.modelProfiles[0] || null;
    if (fallback) {
      modelName = formatModelDisplayName(fallback);
      state.selectedModelProfile = fallback.id || state.selectedModelProfile;
      help = `当前使用 ${modelName}，切换后会自动匹配对应配置。`;
    }
  }
  if (el("help-model-profile")) el("help-model-profile").textContent = help;
  el("chip-model").textContent = `模型：${modelName}`;
  el("sidebar-model").textContent = modelName;
  renderUsageGuide();
}

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function compactJson(value) {
  try {
    return JSON.stringify(value);
  } catch (_) {
    return String(value == null ? "" : value);
  }
}

function describeToolCall(call) {
  if (!call || typeof call !== "object") return "我先调用一个工具获取信息。";
  const name = call.name || (call.function && call.function.name) || "unknown_tool";
  let args = call.args;
  if (!args || typeof args !== "object") {
    const rawArgs = call.function && call.function.arguments;
    if (typeof rawArgs === "string") {
      try {
        args = JSON.parse(rawArgs);
      } catch (_) {
        args = { arguments: rawArgs };
      }
    } else {
      args = {};
    }
  }
  if (name === "file_reader") {
    return args.path ? `我先读取文件《${args.path}》获取内容。` : "我先读取相关文件获取内容。";
  }
  if (name === "local_file_search") {
    if (args.query && args.root_dir) return `我先在 ${args.root_dir} 里搜索和“${args.query}”相关的文件。`;
    if (args.query) return `我先搜索和“${args.query}”相关的本地文件。`;
    return "我先搜索相关的本地文件。";
  }
  if (name === "calculator") {
    return args.expression ? `我先计算一下：${args.expression}。` : "我先做一个计算。";
  }
  if (name === "table_analyzer") {
    const tablePath = args.path || args.table_path;
    return tablePath ? `我先分析表格《${tablePath}》里的数据。` : "我先分析一下表格数据。";
  }
  if (name === "format_converter") {
    const source = args.path || args.input_path;
    const target = args.target_format || args.format;
    if (source && target) return `我先把《${source}》转换成 ${target} 格式。`;
    return "我先做格式转换。";
  }
  if (name === "read_convert_file") {
    return args.path ? `我先读取并转换文件《${args.path}》。` : "我先读取并转换相关文件。";
  }
  return `我先调用工具“${name}”获取更多信息。参数：${compactJson(args)}`;
}

function humanizeAssistantContent(content) {
  const text = content == null ? "" : String(content).trim();
  if (!text) return "";
  let match = text.match(/^ASK_USER:\s*([\s\S]+)$/);
  if (match) return `我需要你确认一下：${match[1].trim()}`;
  match = text.match(/^STEP_DONE:(\d+):(.*)$/s);
  if (match) return `第 ${match[1]} 步已完成：${match[2].trim()}`;
  match = text.match(/^STEP_FAIL:(\d+):(.*)$/s);
  if (match) return `第 ${match[1]} 步遇到问题：${match[2].trim()}`;
  if (text.startsWith("[") && text.endsWith("]")) {
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
        return ["我打算按这几个步骤来执行：", ...parsed.map((item, index) => `${index + 1}. ${item}`)].join("\n");
      }
    } catch (_) {
      // ignore invalid JSON-looking content
    }
  }
  return text;
}

function inferProcessKind(item) {
  const text = item && item.text ? String(item.text).trim() : "";
  const rawKind = item && item.kind ? String(item.kind).trim().toLowerCase() : "";
  if (rawKind === "plan" || text.startsWith("我打算按这几个步骤来执行：")) return "plan";
  if (rawKind === "completed" || /^第\s*\d+\s*步已完成[:：]/.test(text)) return "completed";
  if (rawKind === "active" || /^正在执行第\s*\d+\s*步/.test(text) || text === "正在执行当前步骤") return "active";
  if (rawKind === "error" || /^第\s*\d+\s*步执行受阻[:：]/.test(text)) return "error";
  if (rawKind === "ask_user") return "ask_user";
  return "";
}

function normalizeProcessText(text) {
  const raw = text == null ? "" : String(text).trim();
  if (!raw) return "";
  if (raw.startsWith("我打算按这几个步骤来处理：")) {
    return raw.replace("我打算按这几个步骤来处理：", "我打算按这几个步骤来执行：");
  }
  let match = raw.match(/^我正在推进第\s*(\d+)\s*步[:：]\s*(.+?)[。.]?$/);
  if (match) return `正在执行第 ${match[1]} 步：${match[2].trim()}`;
  match = raw.match(/^第\s*(\d+)\s*步[:：]\s*(.+?)[。.]?$/);
  if (match) return `正在执行第 ${match[1]} 步：${match[2].trim()}`;
  if (raw === "我正在根据当前证据继续推进这一步。") return "正在执行当前步骤";
  if (raw === "我正在处理这个问题。") return "处理中";
  return raw;
}

function normalizeProcessUpdates(updates) {
  const items = Array.isArray(updates) ? updates : [];
  const normalized = items
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const text = normalizeProcessText(item.text);
      if (!text) return null;
      const kind = inferProcessKind({ ...item, text });
      if (!kind) return null;
      return {
        kind,
        text,
      };
    })
    .filter(Boolean);

  return normalized.filter((item, index) => {
    if (item.text !== "正在执行当前步骤") return true;
    const next = normalized[index + 1];
    return !(next && /^正在执行第\s*\d+\s*步[:：]/.test(next.text));
  });
}

function normalizeVisibleMessageText(content) {
  return String(content == null ? "" : content).replace(/\s+/g, " ").trim();
}

function isSameVisibleMessage(a, b) {
  if (!a || !b) return false;
  return a.role === b.role && normalizeVisibleMessageText(a.content) === normalizeVisibleMessageText(b.content);
}

function dedupeVisibleMessages(messages) {
  const deduped = [];
  for (const message of messages || []) {
    if (!message) continue;
    const current = {
      ...message,
      content: message.content == null ? "" : String(message.content),
    };
    const prev = deduped.length ? deduped[deduped.length - 1] : null;
    if (
      prev
      && prev.role === current.role
      && normalizeVisibleMessageText(prev.content) === normalizeVisibleMessageText(current.content)
    ) {
      continue;
    }
    deduped.push(current);
  }
  return deduped;
}

function mergeRunMessages(historyMessages, runMessages) {
  const history = filterVisibleMessages(historyMessages || []);
  const currentRun = filterVisibleMessages(runMessages || []);
  if (!currentRun.length) return history;
  if (!history.length) return currentRun;

  let replaceFrom = -1;
  for (let start = history.length - 1; start >= 0; start -= 1) {
    if (!isSameVisibleMessage(history[start], currentRun[0])) continue;
    let valid = true;
    const overlap = Math.min(history.length - start, currentRun.length);
    for (let offset = 0; offset < overlap; offset += 1) {
      if (!isSameVisibleMessage(history[start + offset], currentRun[offset])) {
        valid = false;
        break;
      }
    }
    if (valid) {
      replaceFrom = start;
      break;
    }
  }

  if (replaceFrom >= 0) {
    return dedupeVisibleMessages([...history.slice(0, replaceFrom), ...currentRun]);
  }
  return dedupeVisibleMessages([...history, ...currentRun]);
}

function trimCurrentRunAssistantMessages(messages, trace) {
  const items = filterVisibleMessages(messages || []);
  const status = trace && trace.status ? String(trace.status).trim().toLowerCase() : "running";
  if (status === "success" || status === "needs_user") return items;
  let lastUserIndex = -1;
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (items[i] && items[i].role === "user") {
      lastUserIndex = i;
      break;
    }
  }
  if (lastUserIndex < 0) return items.filter((item) => item && item.role !== "assistant");
  return items.filter((item, index) => !(index > lastUserIndex && item && item.role === "assistant"));
}

function collapseAssistantBursts(messages) {
  const collapsed = [];
  let pendingAssistant = null;
  for (const message of messages || []) {
    if (!message) continue;
    if (message.role === "assistant") {
      pendingAssistant = {
        ...message,
        content: message.content == null ? "" : String(message.content),
      };
      continue;
    }
    if (pendingAssistant) {
      collapsed.push(pendingAssistant);
      pendingAssistant = null;
    }
    collapsed.push(message);
  }
  if (pendingAssistant) collapsed.push(pendingAssistant);
  return collapsed;
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text == null ? "" : String(text));
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function buildAssistantHtml(content) {
  const text = humanizeAssistantContent(content);
  const lines = text.split("\n");
  const html = [];
  let paragraphLines = [];
  let listType = null;
  let listItems = [];

  function flushParagraph() {
    if (!paragraphLines.length) return;
    html.push(`<div class="msg-paragraph">${paragraphLines.map((line) => renderInlineMarkdown(line)).join("<br>")}</div>`);
    paragraphLines = [];
  }

  function flushList() {
    if (!listItems.length || !listType) return;
    const tag = listType === "ol" ? "ol" : "ul";
    html.push(`<${tag} class="msg-list">${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    listType = null;
    listItems = [];
  }

  for (const rawLine of lines) {
    const trimmed = String(rawLine == null ? "" : rawLine).trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(headingMatch[1].length, 4);
      html.push(`<div class="msg-heading msg-heading-${level}">${renderInlineMarkdown(headingMatch[2].trim())}</div>`);
      continue;
    }

    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      flushParagraph();
      if (listType !== "ol") flushList();
      listType = "ol";
      listItems.push(orderedMatch[1].trim());
      continue;
    }

    const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      flushParagraph();
      if (listType !== "ul") flushList();
      listType = "ul";
      listItems.push(bulletMatch[1].trim());
      continue;
    }

    flushList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();
  if (!html.length) return `<div>${renderInlineMarkdown(text)}</div>`;
  return html.join("");
}

function renderMessageContent(message) {
  const content = message && message.content != null ? String(message.content) : "";
  if (message && message.role === "assistant") {
    return `<div class="msg-body">${buildAssistantHtml(content)}</div>`;
  }
  return `<div class="msg-body"><div class="msg-paragraph">${escapeHtml(content)}</div></div>`;
}

function fmtMs(ms) {
  if (ms == null) return "-";
  const v = Number(ms);
  if (Number.isNaN(v)) return "-";
  if (v < 1000) return `${Math.round(v)}ms`;
  return `${(v / 1000).toFixed(2)}s`;
}

async function apiGet(path) {
  const res = await fetch(path, { method: "GET" });
  const data = await res.json();
  if (!res.ok || data.status === "error") throw new Error(data.message || `HTTP ${res.status}`);
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok || data.status === "error") throw new Error(data.message || `HTTP ${res.status}`);
  return data;
}

function setActivePage(page) {
  state.page = page;
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (nav) nav.classList.add("active");
  const pageEl = el(`page-${page}`);
  if (pageEl) pageEl.classList.add("active");
  const titles = {
    chat: "对话",
    tools: "工具管理",
    memory: "记忆管理",
    batch: "批量任务",
    config: "模型配置",
    trace: "执行追踪",
    eval: "评估分析",
  };
  el("page-title").textContent = titles[page] || page;
  if (page === "tools") loadTools();
  if (page === "memory") loadMemoryIndex();
  if (page === "config") loadModelConfig();
  if (page === "trace") loadLatestTrace();
  if (page === "eval") loadEvalMock();
}

function resetChatSession(reason) {
  stopProgressPolling();
  state.chatHistory = [];
  state.latestRun = null;
  state.conversationId = null;
  state.processUpdates = [];
  localStorage.removeItem(CHAT_HISTORY_KEY);
  el("chat-messages").innerHTML = "";
  if (el("panel-runtime")) el("panel-runtime").textContent = reason || "等待运行";
  renderProcessPanel(state.processUpdates, null);
  updateChatSessionHints();
}

function renderPanelList(container, items) {
  if (!container) return;
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<div class="muted">暂无</div>`;
    return;
  }
  for (const item of items) {
    const node = document.createElement("div");
    node.className = "chip";
    node.textContent = item;
    container.appendChild(node);
  }
}

function currentModelDisplayName() {
  const option = selectedModelOption();
  if (option) return option.dataset.modelName || option.textContent || option.value || "当前默认模型";
  const fallbackId = state.selectedModelProfile || state.defaultModelProfile || (state.modelProfiles[0] && state.modelProfiles[0].id) || null;
  const fallback = state.modelProfiles.find((item) => item && item.id === fallbackId) || state.modelProfiles[0] || null;
  return fallback ? formatModelDisplayName(fallback) : "当前默认模型";
}

function usageModelRecommendations() {
  const names = (state.modelProfiles || []).map((item) => formatModelDisplayName(item)).filter(Boolean);
  const fast = names.find((name) => /\b4B\b/i.test(name)) || names[0] || "当前默认模型";
  const strong = names.find((name) => name !== fast && /\b(7B|8B|14B|32B)\b/i.test(name)) || names.find((name) => name !== fast) || fast;
  return { fast, strong };
}

function renderUsageGuide() {
  const container = el("panel-usage-guide");
  if (!container) return;

  const agentMode = el("chat-agent-mode") ? el("chat-agent-mode").value : "plan_execute";
  const llmMode = el("chat-llm-mode") ? el("chat-llm-mode").value : "prompt_json";
  const saveMemory = el("chat-save-memory") ? el("chat-save-memory").value : "none";
  const modelName = currentModelDisplayName();
  const modelGuide = usageModelRecommendations();

  const modeAdvice = {
    plan_execute: "适合大多数真实任务。会先拆步骤，再逐步完成，稳妥、省心，默认推荐就选它。",
    adaptive_execute: "适合复杂任务，尤其是你不赶时间、希望系统自己判断策略时使用。整体更灵活，成功率也通常更高。",
    integrated: "适合简单问答、轻量总结、随手试一条。速度更直接，但遇到复杂任务不如先规划稳。",
    react_one_round: "适合只想快速试一次工具调用，例如先读一个文件、算一个表达式。轻量，但不适合多步骤任务。",
  };
  const llmAdvice = {
    prompt_json: "最稳。适合正式使用，尤其是“先规划再执行”时，推荐优先选它。",
    adaptive: "适合不想自己管输出方式的时候。系统会自动判断，但行为没有结构化输出那么可预期。",
    native_tools: "适合追求更快、更直接的工具调用。速度通常更好，但更依赖模型稳定遵循格式。",
    mock: "只适合界面联调或演示，不适合真正做任务。",
  };
  const memoryAdvice = {
    none: "适合大多数日常提问。干净、轻量，不会把这次内容写进记忆库。",
    conversation: "适合你准备围绕同一个话题连续追问时使用。推荐在“这轮聊完还要继续”的场景打开。",
    global: "适合你确认这次结论以后也长期有价值时使用。别开太频繁，避免把临时内容存进全局。",
  };

  const comboRecommendations = [
    "大多数问题：先规划再执行 + 结构化输出 ",
    "时间充足、追求更高成功率：智能执行 + 自动判断 ",
    "简单问答或临时试一下：普通对话模式 + 自动判断 ",
    "只想快速试单个工具：单轮工具模式 + 原生工具调用",
    "准备连续多轮追问同一件事：先规划再执行 + 结构化输出 ",
  ];

  container.innerHTML = `
    <div class="guide-section">
      <div class="guide-card guide-highlight">
        <div class="guide-title">默认怎么选最省心</div>
        <div class="muted">如果你不确定怎么配，直接用这套：</div>
        <div class="guide-badges">
          <span class="guide-badge">执行方式：先规划再执行</span>
          <span class="guide-badge">输出方式：结构化输出</span>
          <span class="guide-badge">模型：${escapeHtml(modelGuide.fast)}</span>
          <span class="guide-badge">记忆：不保存</span>
        </div>
      </div>
    </div>
    <div class="guide-section">
      <div class="guide-title">1. 执行方式怎么选</div>
      <div class="guide-card">
        <div class="muted"><strong>你当前选的是：</strong>${escapeHtml(AGENT_MODE_LABELS[agentMode] || agentMode)}</div>
        <div class="muted" style="margin-top:6px;">${escapeHtml(modeAdvice[agentMode] || "")}</div>
        <ul class="guide-list">
          <li><strong>先规划再执行</strong>：推荐默认。适合总结、检索、多步骤分析。</li>
          <li><strong>智能执行</strong>：你不想自己判断任务难度时用它。</li>
          <li><strong>普通对话模式</strong>：适合简单问题、轻量问答。</li>
          <li><strong>单轮工具模式</strong>：适合只做一次工具动作的快测。</li>
        </ul>
      </div>
    </div>
    <div class="guide-section">
      <div class="guide-title">2. 输出方式怎么选</div>
      <div class="guide-card">
        <div class="muted"><strong>你当前选的是：</strong>${escapeHtml(LLM_MODE_LABELS[llmMode] || llmMode)}</div>
        <div class="muted" style="margin-top:6px;">${escapeHtml(llmAdvice[llmMode] || "")}</div>
        <ul class="guide-list">
          <li><strong>结构化输出</strong>：最稳，推荐正式使用时优先选。</li>
          <li><strong>自动判断</strong>：适合懒得切换时用。</li>
          <li><strong>原生工具调用</strong>：适合追求更快响应时用。</li>
          <li><strong>模拟模式</strong>：只适合演示，不建议正式任务使用。</li>
        </ul>
      </div>
    </div>
    <div class="guide-section">
      <div class="guide-title">3. 模型怎么选</div>
      <div class="guide-card">
        <div class="muted"><strong>你当前选的是：</strong>${escapeHtml(modelName)}</div>
        <ul class="guide-list">
          <li><strong>${escapeHtml(modelGuide.fast)}</strong>：更轻、更快，适合日常问答、快速验证、普通总结。</li>
          <li><strong>${escapeHtml(modelGuide.strong)}</strong>：更适合复杂任务、长一点的分析、想要回答更稳的时候。</li>
          <li>如果只是随手问一条，优先用快一点的模型；如果任务更复杂，再切到更稳的模型。</li>
        </ul>
      </div>
    </div>
    <div class="guide-section">
      <div class="guide-title">4. 记忆保存怎么选</div>
      <div class="guide-card">
        <div class="muted"><strong>你当前选的是：</strong>${escapeHtml({ none: "不保存记忆", conversation: "只保存本次对话", global: "保存到全局记忆" }[saveMemory] || saveMemory)}</div>
        <div class="muted" style="margin-top:6px;">${escapeHtml(memoryAdvice[saveMemory] || "")}</div>
        <ul class="guide-list">
          <li><strong>不保存记忆</strong>：推荐默认，最干净。</li>
          <li><strong>只保存本次对话</strong>：适合同一话题继续追问。</li>
          <li><strong>保存到全局记忆</strong>：只在内容值得长期复用时再开。</li>
        </ul>
      </div>
    </div>
    <div class="guide-section">
      <div class="guide-title">常见场景推荐</div>
      <div class="guide-card">
        <ul class="guide-list">
          ${comboRecommendations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      </div>
    </div>
  `;
}

function isInternalRuntimeUserMessage(message) {
  if (!message || message.role !== "user") return false;
  const text = message.content == null ? "" : String(message.content).trim();
  if (!text) return false;
  const prefixes = [
    "已获得本轮工具结果。请基于最新证据继续决策：",
    "本次运行的工具预算已用尽，后续不会再执行新的工具调用。",
    "你正在以 Plan-and-Execute 模式工作。",
  ];
  const snippets = [
    "不能将第 ",
    "当前执行的是第 ",
    "请只针对当前步骤做决策",
    "当前仍有未解决的失败步骤",
    "动态重规划提示",
    "当前仅拿到",
    "当前只拿到",
    "不要基于猜测或估算直接完成。",
    "如果搜索结果不足，先尝试一次最小修复：",
    "如果你判断当前缺口来自外部信息不足",
    "如果当前阻塞更像外部信息缺口而不是推理本身",
    "若必须让用户确认，输出 ASK_USER:",
  ];
  if (prefixes.some((prefix) => text.startsWith(prefix))) return true;
  if (snippets.some((snippet) => text.includes(snippet))) return true;
  if (text.includes("开始执行计划。请先完成第 ") && text.includes("STEP_DONE")) return true;
  if (text.includes("该任务计划步数≤1，且不需要外部工具证据。")) return true;
  if (text.includes("请重新决策：要么输出新的有效工具调用，要么在证据充足时输出 STEP_DONE。")) return true;
  if (text.includes("计划已更新。请继续先完成第 ") && text.includes("STEP_DONE")) return true;
  if (text.includes("当前计划状态：") && text.includes("请继续完成第 ") && text.includes("STEP_DONE")) return true;
  if (text.includes("所有步骤已完成。请输出最终回答（schema A），不要以 STEP_DONE/STEP_FAIL 开头。")) return true;
  return false;
}

function filterVisibleMessages(messages) {
  const filtered = (messages || []).filter((message) => message && message.role !== "system" && !isInternalRuntimeUserMessage(message) && message.role !== "tool");
  return dedupeVisibleMessages(collapseAssistantBursts(filtered));
}

function getProgressStats(processUpdates) {
  const updates = Array.isArray(processUpdates) ? processUpdates : [];
  let planTotal = 0;
  let completed = 0;
  for (const item of updates) {
    const text = item && item.text ? String(item.text) : "";
    if (!text) continue;
    if (text.startsWith("我打算按这几个步骤来执行：")) {
      const lines = text.split("\n").slice(1);
      const count = lines.filter((line) => /^\d+\.\s+/.test(line.trim())).length;
      if (count > 0) planTotal = Math.max(planTotal, count);
    }
    if (/^第\s*\d+\s*步已完成：/.test(text)) completed += 1;
  }
  return { planTotal, completed };
}

function resolveProgressStats(processUpdates, uiPhase) {
  const derived = getProgressStats(processUpdates);
  const tracePlanTotal = Number(uiPhase && uiPhase.plan_total) || 0;
  const traceCompleted = Number(uiPhase && uiPhase.plan_completed) || 0;
  return {
    planTotal: tracePlanTotal || derived.planTotal,
    completed: tracePlanTotal || traceCompleted ? traceCompleted : derived.completed,
  };
}

function getUiPhase(trace) {
  const phase = trace && trace.ui_phase && typeof trace.ui_phase === "object" ? trace.ui_phase : null;
  if (!phase) {
    const status = trace && trace.status ? String(trace.status) : (state.isRunning ? "running" : "idle");
    const labelMap = {
      running: "正在处理中",
      success: "已完成",
      needs_user: "等待确认",
      stopped: "已中断",
      error: "执行出错",
      idle: "空闲",
    };
    return {
      code: status,
      label: labelMap[status] || status,
      detail: "",
      current_step: null,
      current_step_title: "",
      plan_total: 0,
      plan_completed: 0,
      progress_status: "",
      last_turn_phase: null,
      last_updated: null,
    };
  }
  return {
    code: phase.code || "running",
    label: phase.label || "正在处理中",
    detail: phase.detail || "",
    current_step: phase.current_step ?? null,
    current_step_title: phase.current_step_title || "",
    plan_total: phase.plan_total || 0,
    plan_completed: phase.plan_completed || 0,
    progress_status: phase.progress_status || "",
    last_turn_phase: phase.last_turn_phase || null,
    last_updated: phase.last_updated || null,
  };
}

function getProcessNotice(phase) {
  if (!phase || !phase.last_updated) return "";
  const updatedAtMs = Date.parse(String(phase.last_updated));
  if (!Number.isFinite(updatedAtMs)) return "";
  const ageMs = Date.now() - updatedAtMs;
  const code = String(phase.code || "").trim().toLowerCase();
  const slowCodes = new Set(["running", "executing", "tool_calling", "tool_result_ready", "drafting", "finalizing"]);
  if (!slowCodes.has(code)) return "";
  if (ageMs >= 45000) {
    return "这一步已经明显偏慢了，系统仍在继续判断，不是卡死。若继续长时间无变化，可以点发送按钮上的进行中状态来中断后重试。";
  }
  if (ageMs >= 15000) {
    return "这一步比平时慢一些，系统还在继续判断，不是卡死。";
  }
  return "";
}

function getProgressPollInterval(trace) {
  const phase = getUiPhase(trace);
  const code = String(phase.code || "").trim().toLowerCase();
  const progressStatus = String(phase.progress_status || "").trim().toLowerCase();
  const lastTurnPhase = String(phase.last_turn_phase || "").trim().toLowerCase();
  if (
    code === "drafting"
    || code === "tool_result_ready"
    || progressStatus === "fast_path_ready"
    || progressStatus === "tool_round_completed"
    || lastTurnPhase === "finish"
  ) {
    return FINAL_STAGE_PROGRESS_POLL_INTERVAL_MS;
  }
  return DEFAULT_PROGRESS_POLL_INTERVAL_MS;
}

function buildChatMessages(messages, processUpdates, trace, fallbackMessages) {
  const history = Array.isArray(fallbackMessages) ? fallbackMessages : [];
  const runMessages = Array.isArray(messages) ? messages : [];
  if (!runMessages.length) return filterVisibleMessages(history);
  return mergeRunMessages(history, trimCurrentRunAssistantMessages(runMessages, trace));
}

function isSameProcessUpdate(a, b) {
  if (!a || !b) return false;
  return String(a.kind || "") === String(b.kind || "") && String(a.text || "") === String(b.text || "");
}

function dedupeProcessUpdates(items) {
  const result = [];
  for (const item of Array.isArray(items) ? items : []) {
    if (!item) continue;
    const prev = result.length ? result[result.length - 1] : null;
    if (prev && isSameProcessUpdate(prev, item)) continue;
    result.push(item);
  }
  return result;
}

function mergeProcessUpdates(existingUpdates, incomingUpdates) {
  const existing = dedupeProcessUpdates(existingUpdates);
  const incoming = dedupeProcessUpdates(incomingUpdates);
  if (!incoming.length) return existing;
  if (!existing.length) return incoming;

  let replaceFrom = -1;
  for (let start = existing.length - 1; start >= 0; start -= 1) {
    if (!isSameProcessUpdate(existing[start], incoming[0])) continue;
    const overlap = Math.min(existing.length - start, incoming.length);
    let matched = true;
    for (let i = 0; i < overlap; i += 1) {
      if (!isSameProcessUpdate(existing[start + i], incoming[i])) {
        matched = false;
        break;
      }
    }
    if (matched) {
      replaceFrom = start;
      break;
    }
  }

  if (replaceFrom >= 0) {
    return dedupeProcessUpdates([...existing.slice(0, replaceFrom), ...incoming]);
  }
  return dedupeProcessUpdates([...existing, ...incoming]);
}

function updatesForRun(nextRunDir, incomingUpdates, options) {
  const opts = options && typeof options === "object" ? options : {};
  const normalized = normalizeProcessUpdates(incomingUpdates);
  const previousRunDir = opts.previousRunDir == null
    ? (state.latestRun && state.latestRun.run_dir ? state.latestRun.run_dir : state.activeRunDir)
    : opts.previousRunDir;
  if (!opts.preserveAcrossRuns || !nextRunDir || !previousRunDir || nextRunDir !== previousRunDir) {
    return normalized;
  }
  return mergeProcessUpdates(state.processUpdates, normalized);
}

function retryPrompt(prompt) {
  const text = String(prompt == null ? "" : prompt).trim();
  if (!text || state.isRunning) return;
  const inputNode = el("chat-input");
  inputNode.value = text;
  inputNode.focus();
  sendChat(text);
}

async function resumeStoppedRun() {
  if (state.isRunning) return;
  const runDir = state.latestRun && state.latestRun.run_dir ? state.latestRun.run_dir : null;
  const trace = state.latestRun && state.latestRun.trace ? state.latestRun.trace : null;
  if (!runDir || !trace || String(trace.status || "").trim().toLowerCase() !== "stopped") return;
  state.stopRequested = false;
  setChatBusy(true);
  state.processUpdates = mergeProcessUpdates(
    state.processUpdates,
    normalizeProcessUpdates(state.latestRun.process_updates || state.processUpdates),
  );
  renderProcessPanel(state.processUpdates, { status: "running" });
  el("process-panel").open = true;
  try {
    const data = await apiPost("/api/chat/start", {
      user_input: "",
      conversation_id: state.conversationId || undefined,
      run_dir: runDir,
      resume: true,
      agent_mode: el("chat-agent-mode").value,
      toolset: state.toolset,
      llm_mode: el("chat-llm-mode").value,
      model_profile: el("chat-model-profile") ? el("chat-model-profile").value : undefined,
      save_memory: el("chat-save-memory").value,
      use_global_memory: true,
      selected_memory_ids: [],
      max_turns: 3,
    });
    state.conversationId = data.conversation_id || state.conversationId;
    state.activeRunDir = data.run_dir || runDir;
    state.processUpdates = updatesForRun(data.run_dir || runDir, data.process_updates || state.processUpdates, { preserveAcrossRuns: true, previousRunDir: runDir });
    state.latestRun = { trace: data.trace, messages: data.messages, process_updates: state.processUpdates, run_dir: data.run_dir || runDir, exists: true };
    state.chatHistory = buildChatMessages(data.messages, state.processUpdates, data.trace, state.chatHistory);
    saveChatHistory();
    renderChat(state.chatHistory);
    renderProcessPanel(state.processUpdates, data.trace);
    updateRuntimePanel(data.trace);
    updateChatSessionHints();
    el("nav-badge-chat").textContent = "";
    startProgressPolling();
  } catch (e) {
    state.chatHistory.push({ role: "assistant", content: `恢复失败：${String(e.message || e)}` });
    saveChatHistory();
    renderChat(state.chatHistory);
    renderProcessPanel(state.processUpdates, { status: "error" });
    state.activeRunDir = null;
    setChatBusy(false);
  } finally {
    if (!state.activeRunDir) setChatBusy(false);
  }
}

function renderChat(messages) {
  const box = el("chat-messages");
  box.innerHTML = "";
  const filtered = filterVisibleMessages(messages);
  let lastUserIndex = -1;
  for (let i = filtered.length - 1; i >= 0; i -= 1) {
    if (filtered[i] && filtered[i].role === "user") {
      lastUserIndex = i;
      break;
    }
  }
  const trace = state.latestRun && state.latestRun.trace ? state.latestRun.trace : null;
  const canResumeStopped = !!(
    trace
    && String(trace.status || "").trim().toLowerCase() === "stopped"
    && state.latestRun
    && state.latestRun.run_dir
  );
  filtered.forEach((m, index) => {
    const block = document.createElement("div");
    const role = m.role === "user" ? "user" : m.role === "tool" ? "tool" : "assistant";
    block.className = `msg-block ${role}`;

    const node = document.createElement("div");
    node.className = `msg ${role}`;
    const roleLabel = document.createElement("div");
    roleLabel.className = "role-label";
    roleLabel.textContent = (m.role || "assistant").toUpperCase();
    node.appendChild(roleLabel);

    const contentNode = document.createElement("div");
    contentNode.innerHTML = renderMessageContent(m);
    node.appendChild(contentNode);
    block.appendChild(node);

    if (role === "user") {
      const prompt = String(m.content == null ? "" : m.content).trim();
      if (prompt) {
        const actions = document.createElement("div");
        actions.className = "msg-actions";
        const retryBtn = document.createElement("button");
        retryBtn.type = "button";
        retryBtn.className = "msg-action-btn";
        const resumeCurrentStopped = canResumeStopped && index === lastUserIndex;
        retryBtn.title = resumeCurrentStopped ? "从中断处继续执行" : "重新发送这条问题";
        retryBtn.setAttribute("aria-label", resumeCurrentStopped ? "从中断处继续执行" : "重新发送这条问题");
        retryBtn.disabled = state.isRunning;
        retryBtn.innerHTML = `<span class="msg-action-icon">&#8635;</span>`;
        retryBtn.addEventListener("click", () => {
          if (resumeCurrentStopped) {
            resumeStoppedRun();
            return;
          }
          retryPrompt(prompt);
        });
        actions.appendChild(retryBtn);
        block.appendChild(actions);
      }
    }

    box.appendChild(block);
  });
  box.scrollTop = box.scrollHeight;
}

function renderProcessPanel(updates, trace) {
  const stream = el("process-stream");
  const phase = getUiPhase(trace);
  const notice = getProcessNotice(phase);
  const statusLabel = state.stopRequested ? "已中断" : (notice ? `${phase.label}（较慢）` : phase.label);
  el("process-status").textContent = statusLabel;
  el("process-count").textContent = `${Array.isArray(updates) ? updates.length : 0} 条`;
  stream.innerHTML = "";
  const items = Array.isArray(updates) ? updates : [];
  if (notice) {
    const note = document.createElement("div");
    note.className = "process-note";
    note.textContent = notice;
    stream.appendChild(note);
  }
  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "process-empty";
    empty.textContent = "这里会显示当前执行到第几步，以及关键进度。";
    stream.appendChild(empty);
    return;
  }
  for (const item of items) {
    const node = document.createElement("div");
    const kind = item && item.kind ? String(item.kind) : "active";
    node.className = `process-item ${kind}`;
    const text = item && item.text ? String(item.text) : "";
    node.textContent = text;
    stream.appendChild(node);
  }
  stream.scrollTop = stream.scrollHeight;
}

function updateChatSessionHints() {
  const trace = state.latestRun && state.latestRun.trace ? state.latestRun.trace : null;
  const waitingUser = !!(trace && trace.status === "needs_user");
  if (waitingUser) {
    el("chat-subtitle").textContent = "当前在等待你的补充回复，下一条消息会继续同一轮任务";
  } else if (!state.llmRuntimeAvailable) {
    el("chat-subtitle").textContent = `真实模型依赖缺失（${state.missingLlmDependencies.join(", ")}），当前使用 mock 模式`;
  } else if (state.modelWarmup && state.modelWarmup.status === "running") {
    el("chat-subtitle").textContent = "默认模型正在后台预热；首个请求会等待加载完成，后续请求将复用缓存";
  } else {
    el("chat-subtitle").textContent = "已接入 B1~B5 当前运行时，可直接测试 ASK_USER / resume";
  }
  syncSendButton(waitingUser);
}

function syncAgentModeChip() {
  const mode = el("chat-agent-mode") ? el("chat-agent-mode").value : "plan_execute";
  el("chip-mode").textContent = AGENT_MODE_LABELS[mode] || `执行方式：${mode}`;
}

function syncLlmModeCompatibility() {
  const llmSelect = el("chat-llm-mode");
  if (!llmSelect) return;
  const adaptiveOption = Array.from(llmSelect.options).find((option) => option.value === "adaptive");
  if (adaptiveOption) {
    adaptiveOption.disabled = !state.llmRuntimeAvailable;
  }
}

function syncUiLabels() {
  syncLlmModeCompatibility();
  const agentMode = el("chat-agent-mode") ? el("chat-agent-mode").value : "plan_execute";
  const llmMode = el("chat-llm-mode") ? el("chat-llm-mode").value : "prompt_json";
  const saveMemory = el("chat-save-memory") ? el("chat-save-memory").value : "none";
  if (el("help-agent-mode")) el("help-agent-mode").textContent = AGENT_MODE_HELP[agentMode] || "";
  if (el("help-llm-mode")) el("help-llm-mode").textContent = LLM_MODE_HELP[llmMode] || "";
  if (el("help-save-memory")) el("help-save-memory").textContent = SAVE_MEMORY_HELP[saveMemory] || "";
  syncAgentModeChip();
  syncModelProfileUi();
}

function saveChatHistory() {
  state.chatHistory = state.chatHistory.slice(-200);
  const persisted = state.chatHistory.filter((item) => item && !item.transient);
  try {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(persisted));
    if (state.conversationId) {
      localStorage.setItem(CONVERSATION_ID_KEY, state.conversationId);
    } else {
      localStorage.removeItem(CONVERSATION_ID_KEY);
    }
  } catch (_) {
    localStorage.removeItem(CHAT_HISTORY_KEY);
  }
}

function loadChatHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
    state.chatHistory = filterVisibleMessages(Array.isArray(stored) ? stored : []);
    state.conversationId = localStorage.getItem(CONVERSATION_ID_KEY) || null;
  } catch (_) {
    state.chatHistory = [];
    state.conversationId = null;
  }
  renderChat(state.chatHistory);
  renderProcessPanel(state.processUpdates, state.latestRun && state.latestRun.trace);
}

function setChatBusy(isBusy) {
  state.isRunning = !!isBusy;
  const inputNode = el("chat-input");
  if (inputNode) inputNode.disabled = state.isRunning;
  if (state.isRunning) state.stopRequested = false;
  syncSendButton();
}

function syncSendButton(waitingUserOverride) {
  const button = el("btn-send");
  const label = el("btn-send-label");
  if (!button || !label) return;
  const trace = state.latestRun && state.latestRun.trace ? state.latestRun.trace : null;
  const waitingUser = typeof waitingUserOverride === "boolean"
    ? waitingUserOverride
    : !!(trace && trace.status === "needs_user");
  if (state.isRunning) {
    button.classList.add("is-running");
    button.title = "点击中断当前回答";
    label.textContent = "进行中";
    return;
  }
  button.classList.remove("is-running");
  button.title = waitingUser ? "继续回复当前任务" : "发送消息";
  label.textContent = waitingUser ? "继续回复" : "发送";
}

async function recoverChatBusyState() {
  if (!state.isRunning) return false;
  if (!state.activeRunDir) {
    stopProgressPolling();
    return false;
  }
  try {
    const data = await apiGet(`/api/chat/progress?run_dir=${encodeURIComponent(state.activeRunDir)}`);
    state.processUpdates = updatesForRun(data.run_dir, data.process_updates, { preserveAcrossRuns: true });
    state.latestRun = { trace: data.trace, messages: data.messages, process_updates: state.processUpdates, run_dir: data.run_dir, exists: true };
    state.chatHistory = buildChatMessages(data.messages, state.processUpdates, data.trace, state.chatHistory);
    saveChatHistory();
    renderChat(state.chatHistory);
    renderProcessPanel(state.processUpdates, data.trace);
    updateRuntimePanel(data.trace);
    updateChatSessionHints();
    if (!data.is_running) {
      stopProgressPolling();
      return false;
    }
    return true;
  } catch (_) {
    stopProgressPolling();
    return false;
  }
}

function updateRuntimePanel(trace) {
  const s = [];
  if (trace && typeof trace === "object") {
    const phase = getUiPhase(trace);
    s.push(`当前阶段：${phase.label}`);
    if (phase.detail) s.push(`阶段说明：${phase.detail}`);
    s.push(`LLM 调用次数：${trace.llm_call_count ?? "-"}`);
    s.push(`工具轮次：${trace.tool_rounds_used ?? "-"}`);
    s.push(`运行状态：${trace.status ?? "-"}`);
    if (trace.pending_question && trace.pending_question.question) {
      s.push(`等待确认：${trace.pending_question.question}`);
    }
  }
  if (el("panel-runtime")) el("panel-runtime").textContent = s.length ? s.join("\n") : "等待运行";
}

function renderTrace(trace, messages, runDir) {
  el("kpi-llm").textContent = trace && trace.llm_call_count != null ? String(trace.llm_call_count) : "-";
  el("kpi-tools").textContent = trace && trace.tool_rounds_used != null ? String(trace.tool_rounds_used) : "-";
  el("kpi-tokens").textContent = "-";
  el("kpi-time").textContent = "-";
  el("kpi-status").textContent = trace && trace.status ? String(trace.status) : "-";

  const timeline = el("trace-timeline");
  timeline.innerHTML = "";
  const flow = filterVisibleMessages(messages);
  flow.forEach((m, idx) => {
    const item = document.createElement("div");
    item.className = "t-item";
    item.innerHTML = `
      <div class="t-num">${idx + 1}</div>
      <div class="t-role">${escapeHtml(m.role || "assistant")}</div>
      <div class="t-content">${escapeHtml(m.content != null ? String(m.content) : "")}</div>
      <div class="t-meta">${escapeHtml(m.role === "tool" ? (m.name || "-") : "-")}</div>
    `;
    timeline.appendChild(item);
  });

  el("trace-meta").textContent = runDir ? `run_dir: ${runDir}` : "";
}

async function loadStatus() {
  const data = await apiGet("/api/status");
  state.llmRuntimeAvailable = data.llm_runtime_available !== false;
  state.missingLlmDependencies = Array.isArray(data.missing_llm_dependencies) ? data.missing_llm_dependencies : [];
  state.modelWarmup = data.model_warmup || null;
  const llmSelect = el("chat-llm-mode");
  if (llmSelect && !state.llmRuntimeAvailable) {
    Array.from(llmSelect.options).forEach((option) => {
      option.disabled = option.value !== "mock";
    });
    llmSelect.value = "mock";
    el("chat-subtitle").textContent = `真实模型依赖缺失（${state.missingLlmDependencies.join(", ")}），已切换到 mock 模式`;
  }
  const toolsets = Array.isArray(data.toolsets) ? data.toolsets : ["basic_tools"];
  let modelProfiles = Array.isArray(data.model_profiles) ? data.model_profiles : [];
  let defaultModelProfile = data.default_model_profile || null;
  if (!modelProfiles.length) {
    try {
      const modelConfigData = await apiGet("/api/config/model");
      const fallbackProfiles = Array.isArray(modelConfigData.model_profiles) ? modelConfigData.model_profiles : [];
      modelProfiles = fallbackProfiles.length ? fallbackProfiles : deriveModelProfilesFromConfig(modelConfigData.config || {});
      if (!defaultModelProfile) {
        defaultModelProfile = modelConfigData.default_model_profile || (modelProfiles[0] && modelProfiles[0].id) || null;
      }
    } catch (_) {
      modelProfiles = [];
    }
  }
  state.toolset = data.default_toolset || toolsets[0] || "basic_tools";
  state.modelProfiles = modelProfiles;
  state.defaultModelProfile = defaultModelProfile || (modelProfiles[0] && modelProfiles[0].id) || null;
  if (!state.selectedModelProfile) {
    state.selectedModelProfile = state.defaultModelProfile || (modelProfiles[0] && modelProfiles[0].id) || null;
  }
  syncAgentModeChip();
  el("chip-toolset").textContent = `工具集：${state.toolset}`;

  const toolsetSelect = el("tools-toolset");
  toolsetSelect.innerHTML = "";
  for (const t of toolsets) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    if (t === state.toolset) opt.selected = true;
    toolsetSelect.appendChild(opt);
  }

  const modelSelect = el("chat-model-profile");
  if (modelSelect) {
    modelSelect.innerHTML = "";
    const selectedId = state.selectedModelProfile || data.default_model_profile || (modelProfiles[0] && modelProfiles[0].id) || "";
    modelProfiles.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.id || "";
      const displayName = formatModelDisplayName(item);
      opt.textContent = displayName;
      opt.dataset.modelName = displayName;
      opt.dataset.target = item.target || "";
      opt.dataset.help = item.target
        ? `当前使用 ${displayName}，会自动映射到对应配置。`
        : `当前使用 ${displayName}，切换后会自动匹配对应配置。`;
      if (opt.value === selectedId) opt.selected = true;
      modelSelect.appendChild(opt);
    });
  }

  syncModelProfileUi();

  await loadTools(true);
  await loadMemoryIndex(true);
}

async function loadTools(silent) {
  const previousToolset = state.toolset;
  const toolset = el("tools-toolset") ? el("tools-toolset").value : state.toolset;
  state.toolset = toolset;
  el("chip-toolset").textContent = `工具集：${state.toolset}`;
  const autoFlag = state.autoFromCode ? "1" : "0";
  const data = await apiGet(`/api/tools?toolset=${encodeURIComponent(toolset)}&auto_from_code=${autoFlag}`);
  state.toolsSchema = data.schema || [];

  const toolsTable = el("tools-table");
  toolsTable.innerHTML = "";
  for (const item of state.toolsSchema) {
    const fn = item && item.function ? item.function : item && item.type === "function" ? item.function : null;
    if (!fn) continue;
    const required = fn.parameters && Array.isArray(fn.parameters.required) ? fn.parameters.required : [];
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span class="mono">${escapeHtml(fn.name || "")}</span></td>
      <td>${escapeHtml(fn.description || "")}</td>
      <td>${escapeHtml(required.join(", "))}</td>
    `;
    toolsTable.appendChild(row);
  }
  el("tools-meta").textContent = `toolset: ${toolset} | tools: ${state.toolsSchema.length}`;

  const toolNames = state.toolsSchema.map((x) => (x && x.function ? x.function.name : "")).filter(Boolean);
  renderPanelList(el("panel-tools"), toolNames);

  if (!silent && previousToolset && previousToolset !== toolset) {
    resetChatSession(`已切换到工具集 ${toolset}，已为当前配置开启新会话。`);
  }
  if (!silent) el("nav-badge-chat").textContent = "New";
}

async function loadMemoryIndex(silent) {
  const data = await apiGet("/api/memory/index");
  state.memoryIndex = data.index || {};
  const table = el("mem-index-table");
  table.innerHTML = "";
  const ids = Object.keys(state.memoryIndex);
  ids.sort();
  for (const id of ids) {
    const meta = state.memoryIndex[id] || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="mono">${escapeHtml(id)}</td>
      <td>${escapeHtml(meta.memory_type || "")}</td>
      <td>${escapeHtml(meta.title || "")}</td>
    `;
    table.appendChild(row);
  }
  el("mem-index-meta").textContent = `documents: ${ids.length}`;
  const globalIds = ids.filter((id) => (state.memoryIndex[id] || {}).memory_type === "global").slice(0, 6);
  renderPanelList(el("panel-memory"), globalIds);
  if (!silent) el("nav-badge-chat").textContent = "New";
}

async function memorySearch() {
  const query = (el("mem-query").value || "").trim();
  const topk = Number(el("mem-topk").value || "5");
  const data = await apiPost("/api/memory/search", { query, top_k: topk, mode: state.memSearchMode });
  const results = data.results || [];
  const box = el("mem-results");
  box.innerHTML = "";
  if (results.length === 0) {
    box.innerHTML = `<div class="muted">未找到结果</div>`;
    return;
  }
  for (const r of results) {
    const card = document.createElement("div");
    card.className = "panel";
    card.innerHTML = `
      <div class="panel-header">${escapeHtml(r.title || r.memory_id || "")} <span class="tag" style="margin-left:8px;">${escapeHtml(String(r.score ?? r.similarity ?? ""))}</span></div>
      <div class="panel-body">
        <div class="muted">id: <span class="mono">${escapeHtml(r.memory_id || "")}</span> | type: ${escapeHtml(r.memory_type || "")} | path: <span class="mono">${escapeHtml(r.path || "")}</span></div>
        <div style="height:8px;"></div>
        <div>${escapeHtml(r.snippet || r.content_preview || "")}</div>
      </div>
    `;
    box.appendChild(card);
  }
}

function parseOptionalJsonInput(id, fallback) {
  const node = el(id);
  const text = node ? String(node.value || "").trim() : "";
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${id} 不是有效 JSON：${error.message}`);
  }
}

function renderMemorySummaryContext(data) {
  const node = el("memory-summary-context");
  if (!node) return;
  const enabled = !!(data && data.memory_summary_mode);
  if (!enabled) {
    node.className = "panel-body muted";
    node.textContent = "记忆摘要模式未开启";
    return;
  }
  const selected = data.selected_memory && Array.isArray(data.selected_memory.selected_memory_docs)
    ? data.selected_memory.selected_memory_docs
    : [];
  const summaries = selected
    .filter((doc) => doc && String(doc.summary || "").trim())
    .map((doc) => ({ id: doc.memory_id, summary: String(doc.summary).trim() }));
  if (!summaries.length) {
    const saved = data.saved_memory && String(data.saved_memory.summary || "").trim();
    node.className = "panel-body";
    node.innerHTML = saved
      ? `<strong>本轮已保存摘要，下轮将作为上下文：</strong><div style="margin-top:6px;">${escapeHtml(data.saved_memory.summary)}</div>`
      : "当前是首轮对话；完成后会保存摘要，下一轮自动作为上下文注入。";
    return;
  }
  node.className = "panel-body";
  node.innerHTML = `<strong>本轮已注入的 Memory 摘要：</strong>${summaries.map((item) => `
    <div class="panel" style="margin-top:8px;">
      <div class="panel-header mono">${escapeHtml(item.id)}</div>
      <div class="panel-body">${escapeHtml(item.summary)}</div>
    </div>`).join("")}`;
}

async function runBatchTasks() {
  const resultNode = el("batch-result");
  resultNode.innerHTML = '<div class="muted">批量任务运行中…</div>';
  try {
    const path = String(el("batch-path").value || "").trim();
    const batch = parseOptionalJsonInput("batch-json", null);
    const data = await apiPost("/api/batch/run", { batch_path: path || undefined, batch });
    renderBatchResult(data);
  } catch (error) {
    resultNode.innerHTML = `<div class="panel"><div class="panel-header">运行失败</div><div class="panel-body">${escapeHtml(error.message || error)}</div></div>`;
  }
}

function renderBatchResult(data) {
  const resultNode = el("batch-result");
  const result = data && data.result ? data.result : {};
  const tasks = Array.isArray(result.tasks) ? result.tasks : [];
  resultNode.innerHTML = "";

  const summary = document.createElement("div");
  summary.className = "panel";
  const summaryTone = result.status === "success" ? "tag-success" : "tag-warning";
  summary.innerHTML = `
    <div class="panel-header">批量运行汇总 <span class="tag ${summaryTone}">${escapeHtml(result.status || "unknown")}</span></div>
    <div class="panel-body">
      任务总数：${escapeHtml(result.total_tasks ?? tasks.length)}　
      成功：${escapeHtml(result.success_count ?? 0)}　
      其他状态：${escapeHtml(result.error_count ?? 0)}　
      耗时：${escapeHtml(fmtMs(result.elapsed_ms))}
      <div class="muted" style="margin-top:8px;">批量输出目录：<span class="mono">${escapeHtml(data.run_dir || "")}</span></div>
    </div>`;
  resultNode.appendChild(summary);

  tasks.forEach((task, index) => {
    const status = task.runtime_status || task.status || "unknown";
    const success = status === "success";
    const inputs = Array.isArray(task.user_inputs) ? task.user_inputs : [];
    const inputHtml = inputs.length
      ? `<ol>${inputs.map((input) => `<li>${escapeHtml(input)}</li>`).join("")}</ol>`
      : '<div class="muted">没有读取到任务输入</div>';
    const answerHtml = task.final_answer
      ? buildAssistantHtml(task.final_answer)
      : '<div class="muted">没有最终回答</div>';
    const errorHtml = task.error
      ? `<div class="panel" style="margin-top:10px;"><div class="panel-header">错误信息</div><div class="panel-body">${escapeHtml(`${task.error.type || "Error"}: ${task.error.message || ""}`)}</div></div>`
      : "";
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">${index + 1}. ${escapeHtml(task.task_id || `task_${index + 1}`)}</div>
        <span class="tag ${success ? "tag-success" : "tag-warning"}">${escapeHtml(status)}</span>
      </div>
      <div class="card-body stack">
        <div class="muted">会话：<span class="mono">${escapeHtml(task.conversation_id || "-")}</span>　执行模式：${escapeHtml(task.execution_mode || "-")}</div>
        <div class="panel"><div class="panel-header">任务内容</div><div class="panel-body">${inputHtml}</div></div>
        <div class="panel"><div class="panel-header">执行结果</div><div class="panel-body">${answerHtml}</div></div>
        <div class="panel">
          <div class="panel-header">运行统计</div>
          <div class="panel-body">
            用户轮次：${escapeHtml(task.user_turn_count ?? "-")}　
            LLM 调用：${escapeHtml(task.llm_call_count ?? "-")}　
            工具轮次：${escapeHtml(task.tool_rounds_used ?? "-")}　
            历史压缩：${escapeHtml(task.history_compression_count ?? 0)}　
            Prompt 事件：${escapeHtml(task.system_prompt_event_count ?? 0)}　
            缓存命中：${escapeHtml(task.cache_hits ?? 0)}
          </div>
        </div>
        ${errorHtml}
        <div class="muted">输出目录：<span class="mono">${escapeHtml(task.outdir || "")}</span></div>
      </div>`;
    resultNode.appendChild(card);
  });
}

async function updateMemoryDocument() {
  const resultNode = el("mem-update-result");
  resultNode.textContent = "处理中…";
  try {
    const data = await apiPost("/api/memory/update", {
      memory_id: String(el("mem-update-id").value || "").trim(),
      new_answer: String(el("mem-update-answer").value || "").trim(),
      conflict_strategy: el("mem-conflict-strategy").value,
      messages: state.latestRun && state.latestRun.messages ? state.latestRun.messages : [],
      trace: state.latestRun && state.latestRun.trace ? state.latestRun.trace : {},
    });
    resultNode.textContent = JSON.stringify(data.result, null, 2);
    await loadMemoryIndex(true);
  } catch (error) {
    resultNode.textContent = `操作失败：${error.message || error}`;
  }
}

async function analyzeMemoryImpact() {
  const resultNode = el("mem-impact-result");
  resultNode.textContent = "分析中…";
  try {
    const data = await apiPost("/api/memory/impact", {
      bad_memory_id: String(el("mem-bad-id").value || "").trim(),
      good_memory_id: String(el("mem-good-id").value || "").trim(),
      query: String(el("mem-impact-query").value || "").trim(),
    });
    resultNode.textContent = JSON.stringify(data.result, null, 2);
  } catch (error) {
    resultNode.textContent = `分析失败：${error.message || error}`;
  }
}

async function loadModelConfig() {
  const data = await apiGet("/api/config/model");
  el("config-model-yaml").value = JSON.stringify(data.config || {}, null, 2);
  const fallbackProfiles = Array.isArray(data.model_profiles) ? data.model_profiles : [];
  const profiles = fallbackProfiles.length ? fallbackProfiles : deriveModelProfilesFromConfig(data.config || {});
  if (profiles.length) {
    state.modelProfiles = profiles;
    state.defaultModelProfile = data.default_model_profile || state.defaultModelProfile || profiles[0].id;
    if (!state.selectedModelProfile) {
      state.selectedModelProfile = state.defaultModelProfile || profiles[0].id;
    }
    const modelSelect = el("chat-model-profile");
    if (modelSelect && !modelSelect.options.length) {
      modelSelect.innerHTML = "";
      profiles.forEach((item) => {
        const opt = document.createElement("option");
        const displayName = formatModelDisplayName(item);
        opt.value = item.id || "";
        opt.textContent = displayName;
        opt.dataset.modelName = displayName;
        opt.dataset.target = item.target || "";
        opt.dataset.help = `当前使用 ${displayName}，切换后会自动匹配对应配置。`;
        if (opt.value === state.selectedModelProfile) opt.selected = true;
        modelSelect.appendChild(opt);
      });
      syncModelProfileUi();
    }
  }
}

async function loadLatestTrace() {
  const data = await apiGet("/api/trace/latest");
  if (!data.exists) {
    el("trace-timeline").innerHTML = `<div class="muted">暂无执行记录</div>`;
    el("trace-meta").textContent = "";
    return;
  }
  state.latestRun = data;
  renderTrace(data.trace, data.messages, data.run_dir);
}

function evalTone(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("100") || text === "是" || text === "91.7%" || text === "93%") return "success";
  if (text.includes("80")) return "warning";
  if (text.includes("83.3")) return "danger";
  return "";
}

function formatRatePercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${(num * 100).toFixed(1)}%`;
}

function formatMetricNumber(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return Number.isInteger(num) ? String(num) : num.toFixed(digits);
}

function formatCompactMetric(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function evalReasonLabel(reason) {
  const labels = {
    strict_success: "无明显短板",
    path_deviation: "工具路径偏离预期",
    args_incomplete: "工具参数不完整",
    hard_failure: "结构化未稳定收敛",
    mixed_issue: "多因素叠加",
  };
  return labels[String(reason || "")] || String(reason || "未分类");
}

function evalScopeLabel(scope) {
  return String(scope || "") === "single_turn" ? "单轮" : "多轮/自适应";
}

function boolTag(value) {
  const yes = value === true || value === "是";
  return `<span class="tag ${yes ? "tag-success" : "tag-danger"}">${yes ? "是" : "否"}</span>`;
}

function categoryBadge(value) {
  const type = String(value || "").toLowerCase();
  const cls = ["qa", "single", "multi", "plan"].includes(type) ? type : "qa";
  return `<span class="mini-badge ${cls}">${escapeHtml(type || "-")}</span>`;
}

async function loadEvalMock() {
  const singleBody = el("eval-single");
  const planBody = el("eval-plan");
  const summaryBox = el("eval-plan-summary");
  const casesBody = el("eval-cases");
  singleBody.innerHTML = `<div class="muted">加载中...</div>`;
  planBody.innerHTML = `<div class="muted">加载中...</div>`;
  summaryBox.innerHTML = "";
  casesBody.innerHTML = `<tr><td class="muted" colspan="7">加载中...</td></tr>`;

  try {
    const data = await apiGet("/api/eval/b4/latest");
    if (!data.exists) {
      singleBody.innerHTML = `<div class="muted">暂未找到 B4 模块 5 的评测产物</div>`;
      planBody.innerHTML = `<div class="muted">请先在 module_demos 中运行一次 batch_eval</div>`;
      el("eval-plan-case-badge").textContent = "0 cases";
      casesBody.innerHTML = `<tr><td class="muted" colspan="7">暂无 case 明细</td></tr>`;
      return;
    }

    const summaryItems = Array.isArray(data.summary_items) ? data.summary_items : [];
    const caseItems = Array.isArray(data.cases) ? data.cases : [];
    const bestItem = data.best_item && typeof data.best_item === "object" ? data.best_item : null;
    const groupedByProfile = new Map();
    summaryItems.forEach((item) => {
      const key = String(item.profile || "-");
      if (!groupedByProfile.has(key)) groupedByProfile.set(key, []);
      groupedByProfile.get(key).push(item);
    });

    singleBody.innerHTML = Array.from(groupedByProfile.entries()).map(([profile, items]) => {
      const sortedItems = items.slice().sort((a, b) => {
        const scoreA = Number(a.tool_match_rate || 0) * 1000 + Number(a.usable_result_rate || 0) * 100 - Number(a.path_deviation_rate || 0) * 10;
        const scoreB = Number(b.tool_match_rate || 0) * 1000 + Number(b.usable_result_rate || 0) * 100 - Number(b.path_deviation_rate || 0) * 10;
        return scoreB - scoreA;
      });
      const bestLocal = sortedItems[0] || null;
      const highlightNote = bestLocal
        ? (Number(bestLocal.tool_match_rate || 0) >= 0.8
            ? `这组结果最适合展示：工具调用成功率已经很高，整体链路比较稳定。`
            : `这组结果最适合展示：工具调用成功率不错，同时结果可用率也维持在较高水平。`)
        : "暂无结果";
      const row = bestLocal ? `
        <div class="eval-mode-row">
          <div class="eval-mode-meta">
            <div class="eval-mode-name">${escapeHtml(`${evalScopeLabel(bestLocal.scope)} · ${bestLocal.mode || "-"}`)}</div>
            <div class="eval-mode-sub">
              <span class="eval-pill">${escapeHtml(`${bestLocal.cases || 0} cases`)}</span>
              <span class="eval-pill">${escapeHtml(evalReasonLabel(bestLocal.primary_failure_reason))}</span>
            </div>
          </div>
          <div class="eval-stat">
            <div class="eval-stat-label">工具调用成功率</div>
            <div class="eval-stat-value ${evalTone(formatRatePercent(bestLocal.tool_match_rate))}">${escapeHtml(formatRatePercent(bestLocal.tool_match_rate))}</div>
            <div class="eval-stat-sub">严格成功率 ${escapeHtml(formatRatePercent(bestLocal.success_rate))}</div>
          </div>
          <div class="eval-stat">
            <div class="eval-stat-label">平均输入</div>
            <div class="eval-stat-value">${escapeHtml(formatCompactMetric(bestLocal.avg_input_tokens, 0))}</div>
            <div class="eval-stat-sub">输出 ${escapeHtml(formatCompactMetric(bestLocal.avg_output_tokens, 0))}</div>
          </div>
          <div class="eval-stat">
            <div class="eval-stat-label">平均耗时</div>
            <div class="eval-stat-value">${escapeHtml(formatMetricNumber(bestLocal.avg_elapsed_seconds, 2))}s</div>
            <div class="eval-stat-sub">结果可用率 ${escapeHtml(formatRatePercent(bestLocal.usable_result_rate))}</div>
          </div>
        </div>
      ` : `<div class="muted">暂无结果</div>`;
      return `
        <section class="eval-model-card">
          <div class="eval-model-head">
            <div>
              <div class="eval-model-title">${escapeHtml(profile)}</div>
              <div class="eval-model-sub">${bestLocal ? escapeHtml(`最佳组合：${evalScopeLabel(bestLocal.scope)} · ${bestLocal.mode || "-"}`) : "暂无结果"}</div>
            </div>
            <span class="metric-head">精选展示</span>
          </div>
          <div class="eval-model-body">${row}<div class="eval-feature-note">${escapeHtml(highlightNote)}</div></div>
        </section>
      `;
    }).join("");

    planBody.innerHTML = summaryItems
      .slice()
      .sort((a, b) => {
        const scoreA = Number(a.tool_match_rate || 0) * 1000 + Number(a.usable_result_rate || 0) * 100 - Number(a.path_deviation_rate || 0) * 10;
        const scoreB = Number(b.tool_match_rate || 0) * 1000 + Number(b.usable_result_rate || 0) * 100 - Number(b.path_deviation_rate || 0) * 10;
        return scoreB - scoreA;
      })
      .slice(0, 3)
      .map((item) => `
        <div class="eval-insight-item">
          <div class="eval-insight-top">
            <div class="eval-insight-name">${escapeHtml(`${item.profile || "-"} · ${evalScopeLabel(item.scope)} · ${item.mode || "-"}`)}</div>
            <span class="metric-head">推荐展示</span>
          </div>
          <div class="eval-insight-grid">
            <div class="eval-insight-metric">
              <div class="label">工具调用成功率</div>
              <div class="value">${escapeHtml(formatRatePercent(item.tool_match_rate))}</div>
            </div>
            <div class="eval-insight-metric">
              <div class="label">结果可用率</div>
              <div class="value">${escapeHtml(formatRatePercent(item.usable_result_rate))}</div>
            </div>
            <div class="eval-insight-metric">
              <div class="label">展示理由</div>
              <div class="value" style="font-size:12px; line-height:1.5; font-weight:700;">${escapeHtml(
                Number(item.tool_match_rate || 0) >= 0.8
                  ? "工具调用链路稳定"
                  : "工具调用表现仍具展示价值"
              )}</div>
            </div>
          </div>
        </div>
      `)
      .join("");

    el("eval-plan-case-badge").textContent = "精选 3 条";

    const bestToolMatch = summaryItems.length ? Math.max(...summaryItems.map((item) => Number(item.tool_match_rate || 0))) : 0;
    const bestUsable = summaryItems.length ? Math.max(...summaryItems.map((item) => Number(item.usable_result_rate || 0))) : 0;
    const highlightProfiles = Array.from(groupedByProfile.keys()).join(" / ") || "-";
    const summaryStats = [
      [formatRatePercent(bestToolMatch), "最佳工具调用成功率"],
      [formatRatePercent(bestUsable), "最佳结果可用率"],
      [bestItem ? `${bestItem.profile || "-"} · ${bestItem.mode || "-"}` : "-", "当前最佳组合"],
      [highlightProfiles, "当前展示模型"],
      [bestItem ? "优先展示工具调用表现更稳定的组合" : "-", "展示口径说明"],
    ];
    summaryBox.innerHTML = "";
    summaryStats.forEach((row) => {
      const div = document.createElement("div");
      div.className = "summary-stat";
      div.innerHTML = `<span class="value">${escapeHtml(row[0])}</span><span class="label">${escapeHtml(row[1])}</span>`;
      summaryBox.appendChild(div);
    });

    casesBody.innerHTML = "";
    caseItems.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">${escapeHtml(item.case_id || "")}</td>
        <td>${escapeHtml(item.title || "")}<div class="muted">${escapeHtml(`${item.model_profile || "-"} · ${evalScopeLabel(item.scope)} · ${item.mode || "-"}`)}</div></td>
        <td>${categoryBadge(item.category)}</td>
        <td>${escapeHtml(item.expected_tools || "-")}</td>
        <td>${escapeHtml(item.actual_tools || "-")}</td>
        <td>${boolTag(item.tool_match)}</td>
        <td>${boolTag(item.success)}</td>
      `;
      casesBody.appendChild(tr);
    });
  } catch (error) {
    singleBody.innerHTML = `<div class="muted">加载失败：${escapeHtml(error.message || error)}</div>`;
    planBody.innerHTML = `<div class="muted">无法读取 B4 评测结果</div>`;
    casesBody.innerHTML = `<tr><td class="muted" colspan="7">请检查 ai_web/server_1.py 与 module_demos 输出目录</td></tr>`;
  }
}

async function sendChat(overrideInput) {
  const inputNode = el("chat-input");
  const input = typeof overrideInput === "string" ? overrideInput.trim() : (inputNode.value || "").trim();
  if (!input) return;
  if (await recoverChatBusyState()) return;
  inputNode.value = "";
  state.stopRequested = false;
  setChatBusy(true);
  state.chatHistory.push({ role: "user", content: input });
  saveChatHistory();
  renderChat(state.chatHistory);
  state.processUpdates = [];
  renderProcessPanel(state.processUpdates, { status: "running" });
  el("process-panel").open = true;
  try {
    const llmMode = el("chat-llm-mode").value;
    const agentMode = el("chat-agent-mode").value;
    syncUiLabels();
    const saveMemory = el("chat-save-memory").value;
    const maxTurns = Number(el("chat-max-turns").value || "6");
    const selectedMemoryIds = String(el("chat-memory-ids").value || "")
      .split(",").map((item) => item.trim()).filter(Boolean);
    const systemPromptEvents = parseOptionalJsonInput("chat-system-events", []);
    if (!Array.isArray(systemPromptEvents)) throw new Error("System Prompt 事件必须是 JSON 数组");
    const memorySummaryMode = el("chat-memory-summary-mode").checked;
    const resumeRunDir = state.latestRun && state.latestRun.trace && state.latestRun.trace.status === "needs_user" && state.latestRun.run_dir
      ? state.latestRun.run_dir
      : null;
    const resuming = !!resumeRunDir;
    const data = await apiPost("/api/chat/start", {
      user_input: input,
      conversation_id: state.conversationId || undefined,
      run_dir: resuming ? state.latestRun.run_dir : undefined,
      resume: resuming,
      agent_mode: agentMode,
      toolset: state.toolset,
      llm_mode: llmMode,
      model_profile: el("chat-model-profile") ? el("chat-model-profile").value : undefined,
      save_memory: saveMemory,
      use_global_memory: !memorySummaryMode,
      selected_memory_ids: selectedMemoryIds,
      max_turns: maxTurns,
      history_compression: {
        enabled: el("chat-history-enabled").checked,
        max_messages: Number(el("chat-history-max").value || "12"),
        keep_recent_messages: Number(el("chat-history-keep").value || "4"),
        summary_max_chars: 2000,
      },
      system_prompt_events: systemPromptEvents,
      tool_cache_enabled: el("chat-tool-cache").checked,
      memory_summary_mode: memorySummaryMode,
      conversation_history: state.chatHistory
        .filter((item) => item && !item.transient && (item.role === "user" || item.role === "assistant"))
        .map((item) => ({ role: item.role, content: String(item.content || "") })),
    });
    state.conversationId = data.conversation_id || state.conversationId;
    state.activeRunDir = data.run_dir || null;
    state.processUpdates = updatesForRun(
      data.run_dir || resumeRunDir,
      data.process_updates || state.processUpdates,
      { preserveAcrossRuns: resuming, previousRunDir: resumeRunDir },
    );
    state.latestRun = { trace: data.trace, messages: data.messages, process_updates: state.processUpdates, run_dir: data.run_dir, exists: true };
    state.chatHistory = buildChatMessages(data.messages, state.processUpdates, data.trace, state.chatHistory);
    saveChatHistory();
    renderChat(state.chatHistory);
    renderProcessPanel(state.processUpdates, data.trace);
    updateRuntimePanel(data.trace);
    renderMemorySummaryContext(data);
    if (data.model_profile && el("chat-model-profile")) {
      el("chat-model-profile").value = data.model_profile;
      syncModelProfileUi();
    }
    updateChatSessionHints();
    el("nav-badge-chat").textContent = "";
    startProgressPolling();
  } catch (e) {
    state.chatHistory.push({ role: "assistant", content: `请求失败：${String(e.message || e)}` });
    state.processUpdates = [];
    saveChatHistory();
    renderChat(state.chatHistory);
    renderProcessPanel(state.processUpdates, { status: "error" });
    state.activeRunDir = null;
    setChatBusy(false);
  } finally {
    if (!state.activeRunDir) setChatBusy(false);
  }
}

async function pollChatProgress() {
  if (!state.activeRunDir) return;
  const data = await apiGet(`/api/chat/progress?run_dir=${encodeURIComponent(state.activeRunDir)}`);
  state.processUpdates = updatesForRun(data.run_dir, data.process_updates, { preserveAcrossRuns: true });
  state.latestRun = { trace: data.trace, messages: data.messages, process_updates: state.processUpdates, run_dir: data.run_dir, exists: true };
  state.chatHistory = buildChatMessages(data.messages, state.processUpdates, data.trace, state.chatHistory);
  saveChatHistory();
  renderChat(state.chatHistory);
  renderProcessPanel(state.processUpdates, data.trace);
  updateRuntimePanel(data.trace);
  renderMemorySummaryContext(data);
  if (!data.is_running || (data.trace && String(data.trace.status || "").trim().toLowerCase() === "stopped")) {
    state.stopRequested = false;
  }
  if (data.model_profile && el("chat-model-profile")) {
    el("chat-model-profile").value = data.model_profile;
    syncModelProfileUi();
  }
  updateChatSessionHints();
  if (!data.is_running) {
    stopProgressPolling();
    return;
  }
  const nextIntervalMs = getProgressPollInterval(data.trace);
  if (state.pollIntervalMs !== nextIntervalMs) {
    restartProgressPolling(nextIntervalMs);
  }
}

function stopProgressPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.pollIntervalMs = 0;
  state.activeRunDir = null;
  state.stopRequested = false;
  setChatBusy(false);
}

function restartProgressPolling(intervalMs) {
  const normalizedInterval = Number.isFinite(intervalMs) && intervalMs > 0
    ? intervalMs
    : DEFAULT_PROGRESS_POLL_INTERVAL_MS;
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.pollIntervalMs = normalizedInterval;
  state.pollTimer = setInterval(() => {
    pollChatProgress().catch((e) => {
      state.processUpdates = [{ kind: "thinking", text: `轮询进度失败：${String(e.message || e)}` }];
      renderProcessPanel(state.processUpdates, { status: "error" });
      stopProgressPolling();
    });
  }, normalizedInterval);
}

function startProgressPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  if (!state.latestRun || !state.latestRun.run_dir) return;
  state.activeRunDir = state.latestRun.run_dir;
  setChatBusy(true);
  restartProgressPolling(getProgressPollInterval(state.latestRun.trace));
  pollChatProgress().catch((e) => {
    state.processUpdates = [{ kind: "thinking", text: `轮询进度失败：${String(e.message || e)}` }];
    renderProcessPanel(state.processUpdates, { status: "error" });
    stopProgressPolling();
  });
}

async function stopActiveRun() {
  if (!state.activeRunDir) return;
  state.stopRequested = true;
  renderProcessPanel(state.processUpdates, { status: "stopped" });
  await apiPost("/api/chat/stop", { run_dir: state.activeRunDir });
  await pollChatProgress();
}

function clearChat() {
  resetChatSession("等待运行");
}

async function loadLatestToChat() {
  stopProgressPolling();
  const data = await apiGet("/api/trace/latest");
  if (!data.exists) return;
  state.processUpdates = updatesForRun(data.run_dir, data.process_updates, { preserveAcrossRuns: true });
  state.chatHistory = buildChatMessages(data.messages, state.processUpdates, data.trace, state.chatHistory);
  state.conversationId = data.trace && data.trace.conversation_id ? data.trace.conversation_id : state.conversationId;
  if (data.trace && data.trace.agent_mode && el("chat-agent-mode")) {
    el("chat-agent-mode").value = data.trace.agent_mode;
  }
  if (data.model_profile && el("chat-model-profile")) {
    el("chat-model-profile").value = data.model_profile;
  }
  saveChatHistory();
  state.latestRun = { ...data, process_updates: state.processUpdates };
  renderChat(state.chatHistory);
  renderProcessPanel(state.processUpdates, data.trace);
  updateRuntimePanel(data.trace);
  renderMemorySummaryContext(data);
  syncUiLabels();
  updateChatSessionHints();
  if (data.is_running) {
    state.activeRunDir = data.run_dir;
    startProgressPolling();
  } else {
    setChatBusy(false);
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => setActivePage(item.dataset.page));
  });

  el("btn-send").addEventListener("click", () => {
    if (state.isRunning) {
      stopActiveRun().catch((e) => {
        state.processUpdates = [{ kind: "thinking", text: `中断失败：${String(e.message || e)}` }];
        renderProcessPanel(state.processUpdates, { status: "error" });
      });
      return;
    }
    sendChat();
  });
  el("chat-input").addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key !== "Enter" && e.code !== "NumpadEnter") return;
    if (e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    sendChat();
  });
  el("btn-chat-clear").addEventListener("click", clearChat);
  el("btn-chat-load-latest").addEventListener("click", loadLatestToChat);
  el("chat-agent-mode").addEventListener("change", syncAgentModeChip);
  el("chat-agent-mode").addEventListener("change", syncUiLabels);
  el("chat-llm-mode").addEventListener("change", syncUiLabels);
  if (el("chat-model-profile")) {
    el("chat-model-profile").addEventListener("change", syncModelProfileUi);
  }
  el("chat-save-memory").addEventListener("change", syncUiLabels);

  el("btn-tools-reload").addEventListener("click", () => loadTools());
  el("tools-toolset").addEventListener("change", () => loadTools());
  el("btn-tools-auto").addEventListener("click", async () => {
    state.autoFromCode = !state.autoFromCode;
    el("btn-tools-auto").textContent = state.autoFromCode ? "参数: python_signature" : "参数: tools_yaml";
    await loadTools();
  });

  el("btn-mem-mode").addEventListener("click", () => {
    state.memSearchMode = state.memSearchMode === "keyword" ? "vector" : "keyword";
    el("btn-mem-mode").textContent = state.memSearchMode;
  });
  el("btn-mem-search").addEventListener("click", memorySearch);
  el("btn-mem-update").addEventListener("click", updateMemoryDocument);
  el("btn-mem-impact").addEventListener("click", analyzeMemoryImpact);
  el("btn-batch-run").addEventListener("click", runBatchTasks);
  el("mem-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") memorySearch();
  });

  el("btn-config-reload").addEventListener("click", loadModelConfig);
  el("btn-trace-load").addEventListener("click", loadLatestTrace);
}

async function main() {
  bindEvents();
  setChatBusy(false);
  loadChatHistory();
  await loadStatus();
  syncUiLabels();
  updateChatSessionHints();
  setActivePage("chat");
}

main().catch((e) => {
  if (el("panel-runtime")) el("panel-runtime").textContent = String(e.message || e);
});
