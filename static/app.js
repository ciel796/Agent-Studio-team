const state = { module: "b1", modules: {}, clientId: "", displayName: "", presenceTimer: null };
const el = (id) => document.getElementById(id);
const CLIENT_ID_KEY = "agent-module-studio-client-id";
const DISPLAY_NAME_KEY = "agent-module-studio-display-name";

function initializeIdentity() {
  let clientId = localStorage.getItem(CLIENT_ID_KEY);
  if (!clientId) {
    clientId = `client_${typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`}`;
    localStorage.setItem(CLIENT_ID_KEY, clientId);
  }
  state.clientId = clientId;
  state.displayName = localStorage.getItem(DISPLAY_NAME_KEY) || `演示者-${clientId.slice(-4)}`;
  el("participant-name").value = state.displayName;
}

function renderPresence(data) {
  const names = (data.users || []).map((item) => item.display_name).join("、");
  el("online-count").textContent = `${data.online_count || 0} 人在线`;
  el("online-count").title = names || "暂无在线用户";
}

async function heartbeat() {
  const response = await fetch("/api/presence", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ client_id: state.clientId, display_name: state.displayName }) });
  const data = await response.json();
  if (response.ok) renderPresence(data);
}

function startPresence() {
  heartbeat().catch(() => { el("online-count").textContent = "连接失败"; });
  state.presenceTimer = window.setInterval(() => heartbeat().catch(() => {}), 10000);
  el("participant-name").addEventListener("change", () => {
    state.displayName = el("participant-name").value.trim() || `演示者-${state.clientId.slice(-4)}`;
    el("participant-name").value = state.displayName;
    localStorage.setItem(DISPLAY_NAME_KEY, state.displayName);
    heartbeat().catch(() => {});
  });
}

const examples = {
  b1: {
    task: "读取 docs/agent_intro.txt，并总结三条中文要点。",
    userInputs: ["读取 Agent 简介并总结。", "继续说明工具调用循环。", "最后总结记忆模块的作用。"],
    batchTasks: ["读取 Agent 简介并总结。", "说明 Tool Calling 消息闭环。"],
  },
  b2: {
    calculator: { expression: "23 * 17 + 9" },
    file_reader: { path: "docs/agent_intro.txt", max_chars: 2000 },
    local_file_search: { query: "Agent 工具调用", root_dir: "docs", file_types: ["txt", "md"], top_k: 5 },
    table_analyzer: { path: "tables/results.csv", max_rows_preview: 5, describe: true },
    format_converter: { text: "a: 1\nb: 2", target_format: "markdown", output_filename: "converted_sample.md" },
    read_convert_file: { path: "docs/agent_intro.txt", max_chars: 1000, target_format: "markdown", output_filename: "agent_intro_bullets.md" },
    code_executor: { code: "import math\nvalues = [1, 2, 3, 4]\nprint('count', len(values))\nsum(values) + math.sqrt(16)", timeout_seconds: 3, allowed_imports: ["math"], work_dir: "sandbox" },
  },
  b3: {
    basic_tools: [{ id: "demo_call_001", name: "calculator", args: { expression: "8 * 12 + 4" } }],
    advanced_tools: [{ id: "demo_call_advanced", name: "code_executor", args: { code: "sum(range(1, 101))", timeout_seconds: 3, allowed_imports: [], work_dir: "sandbox" } }],
    cache_hit: [
      { id: "cache_call_001", name: "calculator", args: { expression: "10 + 5 * 2" } },
      { id: "cache_call_002", name: "calculator", args: { expression: "10 + 5 * 2" } },
    ],
    timeout: [{ id: "timeout_call_001", name: "code_executor", args: { code: "while True:\n    pass", timeout_seconds: 1, allowed_imports: [], work_dir: "sandbox" } }],
  },
  b4: {
    messages: [{ role: "system", content: "You are a local tool-using agent." }, { role: "user", content: "读取 docs/agent_intro.txt 并总结三点。" }],
    notes: {
      mock: "确定性模拟推理，无需加载模型，适合验收现场快速展示。",
      prompt_json: "把 Tools Schema 写入提示词，真实本地模型输出 JSON 工具调用。",
      native_tools: "通过模型原生 tools 参数传入 Schema，需要模型后端支持原生工具调用。",
    },
  },
  b5: {
    keyword: { query: "Agent 系统如何调用工具？", ids: "" },
    vector: { query: "模型怎样使用外部能力完成任务？", ids: "" },
    auto: { query: "Agent 的模型、工具和记忆如何协作？", ids: "" },
    id: { query: "", ids: "mem_course_001" },
  },
};

const field = (label, html) => `<label>${label}</label>${html}`;
const jsonText = (value) => JSON.stringify(value, null, 2);
const b2Labels = {
  calculator: "calculator",
  file_reader: "file_reader",
  local_file_search: "local_file_search",
  table_analyzer: "table_analyzer",
  format_converter: "format_converter",
  read_convert_file: "read_convert_file（复合 Skill）",
  code_executor: "code_executor",
};

function renderForm(module) {
  if (module === "b1") {
    return field("功能演示模式", `<select id="b1-mode"><option value="single">基础 Agent 循环</option><option value="multi_tool_loop">多次 ToolCall 循环</option><option value="multi_user">多轮用户输入</option><option value="batch">批量任务运行</option><option value="resume">断点恢复</option><option value="history_compression">历史消息压缩</option><option value="prompt_switch">System Prompt 添加/切换</option></select>`) +
      `<div id="b1-task-group">${field("首轮任务描述", `<textarea id="task">${examples.b1.task}</textarea>`)}</div>` +
      `<div id="b1-user-group" hidden>${field("多轮用户输入（每行一轮）", `<textarea id="b1-user-inputs">${examples.b1.userInputs.join("\n")}</textarea>`)}</div>` +
      `<div id="b1-batch-group" hidden>${field("批量任务（每行一个 Agent 任务）", `<textarea id="b1-batch-tasks">${examples.b1.batchTasks.join("\n")}</textarea>`)}</div>` +
      `<p class="example-caption" id="b1-mode-note">运行一次基础消息和工具闭环。</p>`;
  }
  if (module === "b2") {
    return field("Skill 选择", `<select id="skill">${Object.keys(examples.b2).map((name) => `<option value="${name}">${b2Labels[name]}</option>`).join("")}</select>`) +
      field("Skill JSON 参数", `<textarea id="json">${jsonText(examples.b2.calculator)}</textarea>`) +
      `<p class="example-caption">切换 Skill 会自动载入对应的可执行参数示例。</p><div class="composite-note" id="skill-note">当前为单一 Skill，直接执行一个能力函数。</div>`;
  }
  if (module === "b3") {
    return field("执行样例", `<select id="b3-example"><option value="normal">正常工具调用</option><option value="cache_hit">相同参数重复调用（缓存命中）</option><option value="timeout">代码执行超时（可恢复错误与重试）</option></select>`) +
      `<div class="inline-options">${field("动作", `<select id="action"><option value="execute">执行 ToolCall</option><option value="schema">仅生成 Schema</option></select>`)}${field("工具集", `<select id="toolset"><option>basic_tools</option><option>advanced_tools</option></select>`)}</div>` +
      `<div class="inline-options">${field("重试上限", `<input id="retry-limit" type="number" min="0" max="5" value="1">`)}<label class="check-row"><input id="cache-enabled" type="checkbox" checked>启用结果缓存</label></div>` +
      `<label class="check-row"><input id="auto-schema" type="checkbox">从 Python 函数签名自动生成 Schema</label>` +
      field("ToolCall 数组", `<textarea id="json">${jsonText(examples.b3.basic_tools)}</textarea>`) +
      `<p class="example-caption">advanced_tools 比 basic_tools 多出受限代码执行工具 code_executor。</p>`;
  }
  if (module === "b4") {
    return field("功能演示模式", `<select id="b4-action"><option value="generation">单轮 AIMessage</option><option value="multi_tool_roundtrip">多 ToolCall / 多 ToolMessage 往返</option><option value="plan_execute">Plan-and-Execute</option><option value="injection_compare">tools_schema 注入方式对比</option><option value="batch_eval">双模型批量成功率与 Token 对比</option></select>`) +
      `<div class="inline-options">${field("推理模式", `<select id="mode"><option>mock</option><option>prompt_json</option><option>native_tools</option></select>`)}${field("模型 Profile", `<select id="model-profile"><option value="qwen_4b">Qwen3.5-4B</option><option value="qwen_7b">Qwen2.5-7B</option></select>`)}</div>` +
      field("最大计划步骤", `<input id="max-plan-steps" type="number" min="1" max="10" value="4">`) +
      `<p class="example-caption" id="mode-note">${examples.b4.notes.mock}</p>` +
      field("Messages 数组", `<textarea id="json">${jsonText(examples.b4.messages)}</textarea>`);
  }
  return `<div class="inline-options">${field("检索模式", `<select id="mode"><option>keyword</option><option>vector</option><option>auto</option><option>id</option></select>`)}${field("Top K", `<input id="topk" type="number" min="1" max="20" value="5">`)}</div>` +
    field("检索问题", `<textarea id="task">${examples.b5.keyword.query}</textarea>`) +
    field("指定记忆 ID（逗号分隔，仅 ID 模式需要）", `<input id="memory-ids" value="">`) +
    `<label class="check-row"><input id="global-memory" type="checkbox" checked>同时载入全局记忆</label>`;
}

function bindFormEvents(module) {
  if (module === "b1") {
    const notes = {
      single: "运行一次基础消息和工具闭环。",
      multi_tool_loop: "单次对话连续执行多轮 assistant.tool_calls → ToolMessage → assistant。",
      multi_user: "依次追加三轮 user 输入，并在同一 checkpoint 中维护历史。",
      batch: "读取页面中的批量任务列表，通过 run_batch_tasks 分别执行并汇总。",
      resume: "先生成 checkpoint，再使用 resume=True 从同一目录恢复状态。",
      history_compression: "超过消息阈值时生成 history_summary，并保留最近消息继续对话。",
      prompt_switch: "第一轮添加 brief Prompt，第二轮切换 strict Prompt。",
    };
    el("b1-mode").addEventListener("change", (event) => {
      const mode = event.target.value;
      el("b1-mode-note").textContent = notes[mode];
      el("b1-task-group").hidden = mode === "batch";
      el("b1-user-group").hidden = !["multi_user", "history_compression", "prompt_switch"].includes(mode);
      el("b1-batch-group").hidden = mode !== "batch";
    });
  }
  if (module === "b2") {
    el("skill").addEventListener("change", (event) => {
      const skill = event.target.value;
      el("json").value = jsonText(examples.b2[skill]);
      el("skill-note").textContent = skill === "read_convert_file"
        ? "复合调用链：file_reader（读取文件） → format_converter（转换格式） → 汇总 read/conversion 两阶段结果。"
        : "当前为单一 Skill，直接执行一个能力函数。";
    });
  }
  if (module === "b3") {
    el("toolset").addEventListener("change", (event) => { el("json").value = jsonText(examples.b3[event.target.value]); });
    el("b3-example").addEventListener("change", (event) => {
      const preset = event.target.value;
      if (preset === "timeout") {
        el("toolset").value = "advanced_tools";
        el("retry-limit").value = "1";
        el("cache-enabled").checked = false;
      } else {
        el("toolset").value = "basic_tools";
        el("cache-enabled").checked = true;
      }
      el("json").value = jsonText(preset === "normal" ? examples.b3.basic_tools : examples.b3[preset]);
    });
  }
  if (module === "b4") {
    el("mode").addEventListener("change", (event) => { el("mode-note").textContent = examples.b4.notes[event.target.value]; });
    el("b4-action").addEventListener("change", (event) => {
      const action = event.target.value;
      if (["multi_tool_roundtrip", "plan_execute"].includes(action)) el("mode").value = "mock";
      if (["injection_compare", "batch_eval"].includes(action)) el("mode-note").textContent = "展示服务器真实模型运行后保存的验证产物，避免本机重复加载大模型。";
    });
  }
  if (module === "b5") {
    el("mode").addEventListener("change", (event) => {
      const sample = examples.b5[event.target.value];
      el("task").value = sample.query;
      el("memory-ids").value = sample.ids;
    });
  }
}

function payload() {
  if (state.module === "b1") return { demo_mode: el("b1-mode").value, task: el("task").value, user_inputs: el("b1-user-inputs").value.split("\n").map((item) => item.trim()).filter(Boolean), batch_tasks: el("b1-batch-tasks").value.split("\n").map((item) => item.trim()).filter(Boolean) };
  if (state.module === "b2") return { skill: el("skill").value, args: JSON.parse(el("json").value) };
  if (state.module === "b3") return { action: el("action").value, toolset: el("toolset").value, auto_from_code: el("auto-schema").checked, retry_limit: Number(el("retry-limit").value), cache_enabled: el("cache-enabled").checked, tool_calls: JSON.parse(el("json").value) };
  if (state.module === "b4") return { action: el("b4-action").value, mode: el("mode").value, profile: el("model-profile").value, max_plan_steps: Number(el("max-plan-steps").value), messages: JSON.parse(el("json").value) };
  return {
    query: el("task").value,
    search_mode: el("mode").value,
    top_k: Number(el("topk").value),
    selected_memory_ids: el("memory-ids").value.split(",").map((item) => item.trim()).filter(Boolean),
    use_global_memory: el("global-memory").checked,
  };
}

function renderNavigation() {
  el("module-nav").innerHTML = Object.entries(state.modules).map(([key, meta]) =>
    `<button class="nav-button${key === state.module ? " active" : ""}" data-module="${key}" type="button"><span class="nav-code">${key.toUpperCase()}</span><span class="nav-name">${meta.title.replace(/^B\d\s*/, "")}</span></button>`
  ).join("");
  document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => selectModule(button.dataset.module)));
}

function selectModule(module) {
  if (!state.modules[module]) return;
  state.module = module;
  window.location.hash = module;
  const meta = state.modules[module];
  document.title = `${module.toUpperCase()} · Agent Module Studio`;
  el("title").textContent = meta.title;
  el("summary").textContent = meta.summary;
  el("owner").textContent = `OWNER ${meta.owner} · ${module.toUpperCase()} INDEPENDENT DEMO`;
  el("input-contract").textContent = meta.input;
  el("output-contract").textContent = meta.output;
  el("endpoint").textContent = meta.output_interface;
  el("input-from").textContent = meta.input_from;
  el("generated-output").textContent = meta.output;
  el("receiver").textContent = meta.receiver;
  el("integration").textContent = meta.integration;
  el("module-chip").textContent = module.toUpperCase();
  el("form").innerHTML = renderForm(module);
  el("status").textContent = "等待运行";
  el("metrics").innerHTML = "";
  el("output").textContent = "运行后显示模块标准输出。";
  el("result-insights").hidden = true;
  bindFormEvents(module);
  renderNavigation();
}

async function run() {
  const button = el("run");
  button.disabled = true;
  el("status").textContent = "运行中";
  el("output").textContent = `正在调用 /api/modules/${state.module}/run ...`;
  try {
    const requestPayload = { ...payload(), _client_id: state.clientId, _display_name: state.displayName };
    const response = await fetch(`/api/modules/${state.module}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestPayload) });
    const data = await response.json();
    if (!response.ok || data.status === "error") throw new Error(data.error?.message || `HTTP ${response.status}`);
    el("status").textContent = "成功";
    el("output").textContent = JSON.stringify(data, null, 2);
    el("metrics").innerHTML = `<span class="metric">模块 ${data.module}</span><span class="metric">接口调用成功</span><span class="metric">运行产物已保存</span>`;
    renderResultParameters(data);
  } catch (error) {
    el("status").textContent = "失败";
    el("output").textContent = String(error.message || error);
    el("metrics").innerHTML = `<span class="metric">结构化异常</span>`;
  } finally {
    button.disabled = false;
  }
}

function parameterItem(label, value, note = "") {
  return `<div class="parameter-item"><span>${label}</span><strong>${value}</strong>${note ? `<em>${note}</em>` : ""}</div>`;
}

function renderResultParameters(data) {
  const section = el("result-insights");
  if (data.module === "B1") {
    const params = data.output.result_parameters;
    if (params.demo_mode === "batch") {
      el("parameter-row").innerHTML =
        parameterItem("演示模式", "批量任务") + parameterItem("任务总数", params.batch_total) +
        parameterItem("成功 / 失败", `${params.batch_success} / ${params.batch_errors}`) + parameterItem("批量耗时", `${params.elapsed_ms} ms`);
    } else {
      el("parameter-row").innerHTML =
        parameterItem("演示模式", params.demo_mode) + parameterItem("用户轮数", params.user_turn_count) +
        parameterItem("工具循环", params.tool_rounds_used) + parameterItem("LLM 调用", params.llm_call_count) +
        parameterItem("断点状态", params.resumed_from_checkpoint ? "已恢复" : (params.checkpoint_exists ? "已保存" : "无")) +
        parameterItem("历史压缩", `${params.history_compression_count} 次`) +
        parameterItem("Prompt 事件", `${params.system_prompt_event_count} 次`, `当前 ${params.active_system_prompt}`);
    }
  } else if (data.module === "B3") {
    const params = data.output.result_parameters;
    const accuracy = params.schema_accuracy || {};
    el("parameter-row").innerHTML =
      parameterItem("缓存命中", params.cache_hit ? "是" : "否", params.cache_enabled ? "缓存已启用" : "缓存未启用") +
      parameterItem("调用次数", params.total_tool_calls) +
      parameterItem("失败率", `${(params.failure_rate * 100).toFixed(2)}%`) +
      parameterItem("重试次数", params.retry_count) +
      parameterItem("平均耗时", `${params.avg_latency_ms} ms`) +
      parameterItem("Schema 描述准确率", `详细 ${(Number(accuracy.detailed_schema || 0) * 100).toFixed(0)}%`, `简略 ${(Number(accuracy.brief_schema || 0) * 100).toFixed(0)}% · 固定评测集`);
  } else if (data.module === "B5") {
    const params = data.output.result_parameters;
    const scores = params.top_k_scores_desc.length ? params.top_k_scores_desc.map((score) => Number(score).toFixed(4)).join(" > ") : "无检索分数";
    const vector = params.vector_evaluation;
    el("parameter-row").innerHTML =
      parameterItem("Top-K 返回", `${params.returned_count} / ${params.requested_top_k}`) +
      parameterItem("Top-K 得分（降序）", scores, params.scores_descending ? "已按相关度从高到低排列" : "请检查排序") +
      parameterItem("长度管理", `${params.max_memory_chars} max chars`, params.length_management_applied ? `本次纳入 ${params.total_included_chars} 字符 · ${params.content_truncated ? "发生截断" : "未截断"}` : "当前为检索排序阶段，全文装载时应用上限") +
      parameterItem("检索评测参数", vector ? `${vector.dimensions} 维 · cosine` : "关键词/ID 模式", vector ? `${vector.method} · min score ${vector.minimum_score}` : "按当前检索模式返回");
  } else if (data.module === "B4") {
    const params = data.output.result_parameters;
    if (params.action === "batch_eval") {
      el("parameter-row").innerHTML = Object.entries(params.model_comparison).map(([name, item]) =>
        parameterItem(name, `成功率 ${(Number(item.success_rate || 0) * 100).toFixed(1)}%`, `工具匹配 ${(Number(item.tool_match_rate || 0) * 100).toFixed(1)}% · input ${item.avg_input_tokens ?? "N/A"} · output ${item.avg_output_tokens ?? "N/A"}`)
      ).join("");
    } else if (params.action === "injection_compare") {
      el("parameter-row").innerHTML =
        parameterItem("Prompt 注入", params.prompt_json_success ? "成功" : "失败", `input ${params.prompt_json_input_tokens} · output ${params.prompt_json_output_tokens}`) +
        parameterItem("原生 tools_schema", params.native_tools_success ? "成功" : "失败", `input ${params.native_tools_input_tokens} · output ${params.native_tools_output_tokens}`) +
        parameterItem("数据来源", params.data_source);
    } else if (params.action === "plan_execute") {
      el("parameter-row").innerHTML = parameterItem("计划步骤", params.plan_steps) + parameterItem("计划状态", params.plan_status) + parameterItem("LLM 调用", params.llm_call_count) + parameterItem("工具轮次", params.tool_rounds_used) + parameterItem("模型 Profile", params.selected_profile) + parameterItem("数据来源", params.data_source);
    } else {
      el("parameter-row").innerHTML = parameterItem("生成 ToolCall", params.generated_tool_calls ?? 0) + parameterItem("接收 ToolMessage", params.received_tool_messages ?? 0) + parameterItem("后续 ToolCall", params.followup_tool_calls ?? 0) + parameterItem("模型 Profile", params.selected_profile) + parameterItem("Schema 模式", params.mode) + parameterItem("Token", `${params.input_tokens ?? "N/A"} / ${params.output_tokens ?? "N/A"}`, "input / output");
    }
  } else {
    section.hidden = true;
    return;
  }
  el("result-module").textContent = data.module;
  section.hidden = false;
}

async function main() {
  initializeIdentity();
  startPresence();
  const response = await fetch("/api/modules");
  const data = await response.json();
  state.modules = data.modules;
  const requested = window.location.hash.slice(1).toLowerCase();
  selectModule(state.modules[requested] ? requested : "b1");
  el("run").addEventListener("click", run);
  window.addEventListener("hashchange", () => {
    const module = window.location.hash.slice(1).toLowerCase();
    if (module !== state.module && state.modules[module]) selectModule(module);
  });
}

main().catch((error) => { el("output").textContent = String(error); });
