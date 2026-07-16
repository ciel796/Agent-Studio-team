const state = {
  page: "chat",
  toolset: "basic_tools",
  autoFromCode: false,
  toolsSchema: [],
  memoryIndex: {},
  latestRun: null,
  memSearchMode: "keyword",
  chatHistory: [],
};

const CHAT_HISTORY_KEY = "agent-studio-chat-history";

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
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

function renderPanelList(container, items) {
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

function renderChat(messages) {
  const box = el("chat-messages");
  box.innerHTML = "";
  const filtered = (messages || []).filter((m) => m && m.role !== "system");
  for (const m of filtered) {
    const node = document.createElement("div");
    node.className = `msg ${m.role === "user" ? "user" : m.role === "tool" ? "tool" : "assistant"}`;
    const roleLabel = document.createElement("div");
    roleLabel.className = "role-label";
    roleLabel.textContent = (m.role || "assistant").toUpperCase();
    node.appendChild(roleLabel);

    const contentNode = document.createElement("div");
    if (m.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
      const lines = [];
      for (const call of m.tool_calls) {
        const args = call && call.args ? JSON.stringify(call.args) : call && call.function && call.function.arguments ? call.function.arguments : "";
        const name = call && (call.name || (call.function && call.function.name)) ? (call.name || call.function.name) : "unknown_tool";
        lines.push(`tool_call: ${name} ${args}`);
      }
      if (m.content) lines.push(m.content);
      contentNode.innerHTML = `<div class="mono">${escapeHtml(lines.join("\n"))}</div>`;
    } else {
      const content = m && m.content != null ? String(m.content) : "";
      const name = m.role === "tool" && m.name ? `[${m.name}] ` : "";
      contentNode.innerHTML = `<div>${escapeHtml(name + content)}</div>`;
    }
    node.appendChild(contentNode);
    box.appendChild(node);
  }
  box.scrollTop = box.scrollHeight;
}

function saveChatHistory() {
  state.chatHistory = state.chatHistory.slice(-200);
  try {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(state.chatHistory));
  } catch (_) {
    localStorage.removeItem(CHAT_HISTORY_KEY);
  }
}

function loadChatHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
    state.chatHistory = Array.isArray(stored) ? stored : [];
  } catch (_) {
    state.chatHistory = [];
  }
  renderChat(state.chatHistory);
}

function updateRuntimePanel(trace) {
  const s = [];
  if (trace && typeof trace === "object") {
    s.push(`LLM Calls: ${trace.llm_call_count ?? "-"}`);
    s.push(`Tool Rounds: ${trace.tool_rounds_used ?? "-"}`);
    s.push(`Status: ${trace.status ?? "-"}`);
  }
  el("panel-runtime").textContent = s.length ? s.join("\n") : "等待运行";
}

function renderTrace(trace, messages, runDir) {
  el("kpi-llm").textContent = trace && trace.llm_call_count != null ? String(trace.llm_call_count) : "-";
  el("kpi-tools").textContent = trace && trace.tool_rounds_used != null ? String(trace.tool_rounds_used) : "-";
  el("kpi-tokens").textContent = "-";
  el("kpi-time").textContent = "-";
  el("kpi-status").textContent = trace && trace.status ? String(trace.status) : "-";

  const timeline = el("trace-timeline");
  timeline.innerHTML = "";
  const flow = (messages || []).filter((m) => m && m.role !== "system");
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
  const toolsets = Array.isArray(data.toolsets) ? data.toolsets : ["basic_tools"];
  state.toolset = data.default_toolset || toolsets[0] || "basic_tools";
  el("chip-mode").textContent = `mode: integrated`;
  el("chip-toolset").textContent = `toolset: ${state.toolset}`;
  el("chip-model").textContent = `model: local`;
  el("sidebar-model").textContent = "local";

  const toolsetSelect = el("tools-toolset");
  toolsetSelect.innerHTML = "";
  for (const t of toolsets) {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    if (t === state.toolset) opt.selected = true;
    toolsetSelect.appendChild(opt);
  }

  await loadTools(true);
  await loadMemoryIndex(true);
}

async function loadTools(silent) {
  const toolset = el("tools-toolset") ? el("tools-toolset").value : state.toolset;
  state.toolset = toolset;
  el("chip-toolset").textContent = `toolset: ${state.toolset}`;
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
      <div class="panel-header">${escapeHtml(r.title || r.memory_id || "")} <span class="tag" style="margin-left:8px;">${escapeHtml(String(r.score ?? ""))}</span></div>
      <div class="panel-body">
        <div class="muted">id: <span class="mono">${escapeHtml(r.memory_id || "")}</span> | type: ${escapeHtml(r.memory_type || "")} | path: <span class="mono">${escapeHtml(r.path || "")}</span></div>
        <div style="height:8px;"></div>
        <div>${escapeHtml(r.snippet || r.content_preview || "")}</div>
      </div>
    `;
    box.appendChild(card);
  }
}

async function loadModelConfig() {
  const data = await apiGet("/api/config/model");
  el("config-model-yaml").value = JSON.stringify(data.config || {}, null, 2);
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

function boolTag(value) {
  const yes = value === true || value === "是";
  return `<span class="tag ${yes ? "tag-success" : "tag-danger"}">${yes ? "是" : "否"}</span>`;
}

function categoryBadge(value) {
  const type = String(value || "").toLowerCase();
  const cls = ["qa", "single", "multi", "plan"].includes(type) ? type : "qa";
  return `<span class="mini-badge ${cls}">${escapeHtml(type || "-")}</span>`;
}

function loadEvalMock() {
  const single = [
    ["测试用例数", "12", "12", "prompt_json", "native_tools"],
    ["成功率", "83.3%", "91.7%", "prompt_json", "native_tools"],
    ["工具匹配率", "91.7%", "100%", "prompt_json", "native_tools"],
    ["平均输入 Token", "1,245", "1,102", "prompt_json", "native_tools"],
    ["平均耗时", "2.34s", "1.89s", "prompt_json", "native_tools"],
  ];
  const plan = [
    ["找房+预算分析", "4", "100%", "1"],
    ["文件搜索汇总", "3", "100%", "0"],
    ["多文件对比", "5", "80%", "2"],
  ];
  const planSummary = [
    ["93%", "步骤完成率"],
    ["1.0", "平均重规划"],
    ["5.2", "平均 LLM Calls"],
  ];
  const cases = [
    ["case_001", "简单问答", "qa", "—", "—", true, true],
    ["case_002", "读取文件", "single", "file_reader", "file_reader", true, true],
    ["case_003", "多文件读取", "multi", "reader x2", "reader x2", true, true],
    ["case_004", "搜索+读取", "multi", "search, reader", "search, reader", true, true],
    ["case_005", "预算计算", "plan", "analyzer, calc", "analyzer, calc", true, true],
  ];
  const singleBody = el("eval-single");
  singleBody.innerHTML = "";
  single.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="metric-name">${escapeHtml(row[0])}</td>
      <td class="metric-col">
        <span class="metric-head">${escapeHtml(row[3])}</span>
        <span class="metric-num ${evalTone(row[1])}">${escapeHtml(row[1])}</span>
      </td>
      <td class="metric-col">
        <span class="metric-head">${escapeHtml(row[4])}</span>
        <span class="metric-num ${evalTone(row[2])}">${escapeHtml(row[2])}</span>
      </td>
    `;
    singleBody.appendChild(tr);
  });

  const planBody = el("eval-plan");
  planBody.innerHTML = "";
  plan.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row[0])}</td>
      <td>${escapeHtml(row[1])}</td>
      <td><span class="tag ${row[2] === "100%" ? "tag-success" : "tag-warning"}">${escapeHtml(row[2])}</span></td>
      <td>${escapeHtml(row[3])}</td>
    `;
    planBody.appendChild(tr);
  });

  el("eval-plan-case-badge").textContent = `${plan.length} cases`;

  const summaryBox = el("eval-plan-summary");
  summaryBox.innerHTML = "";
  planSummary.forEach((row) => {
    const div = document.createElement("div");
    div.className = "summary-stat";
    div.innerHTML = `<span class="value">${escapeHtml(row[0])}</span><span class="label">${escapeHtml(row[1])}</span>`;
    summaryBox.appendChild(div);
  });

  const casesBody = el("eval-cases");
  casesBody.innerHTML = "";
  cases.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${escapeHtml(row[0])}</td>
      <td>${escapeHtml(row[1])}</td>
      <td>${categoryBadge(row[2])}</td>
      <td>${escapeHtml(row[3])}</td>
      <td>${escapeHtml(row[4])}</td>
      <td>${boolTag(row[5])}</td>
      <td>${boolTag(row[6])}</td>
    `;
    casesBody.appendChild(tr);
  });
}

async function sendChat() {
  const inputNode = el("chat-input");
  const input = (inputNode.value || "").trim();
  if (!input) return;
  inputNode.value = "";
  el("btn-send").disabled = true;
  state.chatHistory.push({ role: "user", content: input });
  saveChatHistory();
  renderChat(state.chatHistory);
  const box = el("chat-messages");
  const thinkingNode = document.createElement("div");
  thinkingNode.className = "msg assistant thinking";
  thinkingNode.innerHTML = `<div class="role-label">ASSISTANT</div><div class="thinking-text">Thinking<span class="thinking-dots" aria-hidden="true"></span></div>`;
  box.appendChild(thinkingNode);
  box.scrollTop = box.scrollHeight;
  try {
    const llmMode = el("chat-llm-mode").value;
    const saveMemory = el("chat-save-memory").value;
    const data = await apiPost("/api/chat", {
      user_input: input,
      toolset: state.toolset,
      llm_mode: llmMode,
      save_memory: saveMemory,
      use_global_memory: true,
      selected_memory_ids: [],
      max_turns: 3,
    });
    thinkingNode.remove();
    const responseMessages = (data.messages || []).filter((message) => message && message.role !== "system" && message.role !== "user");
    state.chatHistory.push(...responseMessages);
    saveChatHistory();
    renderChat(state.chatHistory);
    updateRuntimePanel(data.trace);
    state.latestRun = { trace: data.trace, messages: data.messages, run_dir: data.run_dir, exists: true };
    el("nav-badge-chat").textContent = "";
  } catch (e) {
    thinkingNode.remove();
    state.chatHistory.push({ role: "tool", name: "ERROR", content: String(e.message || e) });
    saveChatHistory();
    renderChat(state.chatHistory);
  } finally {
    el("btn-send").disabled = false;
  }
}

function clearChat() {
  state.chatHistory = [];
  localStorage.removeItem(CHAT_HISTORY_KEY);
  el("chat-messages").innerHTML = "";
  el("panel-runtime").textContent = "等待运行";
}

async function loadLatestToChat() {
  const data = await apiGet("/api/trace/latest");
  if (!data.exists) return;
  state.chatHistory = (data.messages || []).filter((message) => message && message.role !== "system");
  saveChatHistory();
  renderChat(state.chatHistory);
  updateRuntimePanel(data.trace);
  state.latestRun = data;
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => setActivePage(item.dataset.page));
  });

  el("btn-send").addEventListener("click", sendChat);
  el("chat-input").addEventListener("keydown", (e) => {
    if (e.isComposing) return;
    if (e.key !== "Enter" && e.code !== "NumpadEnter") return;
    if (e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    sendChat();
  });
  el("btn-chat-clear").addEventListener("click", clearChat);
  el("btn-chat-load-latest").addEventListener("click", loadLatestToChat);

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
  el("mem-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") memorySearch();
  });

  el("btn-config-reload").addEventListener("click", loadModelConfig);
  el("btn-trace-load").addEventListener("click", loadLatestTrace);
}

async function main() {
  bindEvents();
  loadChatHistory();
  await loadStatus();
  setActivePage("chat");
}

main().catch((e) => {
  el("panel-runtime").textContent = String(e.message || e);
});
