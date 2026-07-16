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
    messageSets: {
      generation_multi_tool_calls: [
        { role: "system", content: "You are a local tool-using agent. If tools are needed, you may request one or multiple tool calls in the same response when they are independent." },
        { role: "user", content: "请同时阅读 docs/agent_intro.txt 和 docs/tool_calling.md，比较这两个文档的关注点差异。" },
      ],
      generation_multi_tool_messages: [
        { role: "system", content: "You are a local tool-using agent. If tools are needed, you may request one or multiple tool calls in the same response when they are independent. After receiving ToolMessage results, decide whether to request more tools or provide the final answer." },
        { role: "user", content: "请同时阅读 docs/agent_intro.txt 和 docs/tool_calling.md，比较这两个文档的关注点差异。" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            { id: "call_001", name: "file_reader", args: { path: "docs/agent_intro.txt", max_chars: 2000 } },
            { id: "call_002", name: "file_reader", args: { path: "docs/tool_calling.md", max_chars: 2000 } },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_001",
          name: "file_reader",
          content: "{\"skill_name\":\"file_reader\",\"status\":\"success\",\"input\":{\"path\":\"docs/agent_intro.txt\",\"max_chars\":2000},\"output\":{\"content\":\"文档A聚焦 Agent 的组成结构。\",\"num_chars\":18,\"source\":\"docs/agent_intro.txt\",\"truncated\":false},\"error\":null,\"latency_ms\":1.0}",
          status: "success",
        },
        {
          role: "tool",
          tool_call_id: "call_002",
          name: "file_reader",
          content: "{\"skill_name\":\"file_reader\",\"status\":\"success\",\"input\":{\"path\":\"docs/tool_calling.md\",\"max_chars\":2000},\"output\":{\"content\":\"文档B聚焦 Tool Calling 的闭环机制。\",\"num_chars\":27,\"source\":\"docs/tool_calling.md\",\"truncated\":false},\"error\":null,\"latency_ms\":1.0}",
          status: "success",
        },
      ],
      plan_execute: [
        { role: "system", content: "You are a local tool-using agent." },
        {
          role: "user",
          content: "请以 Plan-and-Execute 的方式完成一个稳定的三步任务：1) 读取 docs/agent_intro.txt，总结 Agent 系统的 3 条核心组成；2) 读取 docs/tool_calling.md，总结 Tool Calling 闭环的 3 条关键机制，并指出其中哪 1 条与第1步最相关；3) 读取 tables/results.csv，说明这个表里最值得关注的 2 个字段，并判断它更适合用于展示“结果统计”还是“过程追踪”。最后给出一段整合结论。"
        },
      ],
      react_one_round: [
        { role: "system", content: "You are a local tool-using agent." },
        {
          role: "user",
          content: "请读取 docs/agent_intro.txt，并用中文总结 3 条要点。尽量用最少轮次完成。"
        },
      ],
      adaptive: [
        { role: "system", content: "You are a local tool-using agent." },
        {
          role: "user",
          content: "请判断这个任务更适合 ReAct 还是 Plan-and-Execute，并完成它：先比较 docs/agent_intro.txt 与 docs/tool_calling.md 的关注点差异，再看 tables/results.csv 是否能补充一个“评测结果统计”的例子，最后给出你的结论。"
        },
      ],
      model_routing: [
        { role: "system", content: "You are a local tool-using agent. If tools are needed, you may request one or multiple tool calls in the same response when they are independent." },
        { role: "user", content: "请比较 docs/agent_intro.txt 与 docs/tool_calling.md 的关注点差异，并说明这个任务在 direct / plan / execute 三个阶段里更适合怎样的模型配置。" },
      ],
      injection_compare: [
        { role: "system", content: "You are a local tool-using agent." },
        { role: "user", content: "请同时阅读 docs/agent_intro.txt 和 docs/tool_calling.md，比较这两个文档的关注点差异。" },
      ],
      batch_eval: [
        { role: "system", content: "You are a local tool-using agent." },
        { role: "user", content: "读取 docs/agent_intro.txt 并总结三点。" },
      ],
      execution_engine: [
        { role: "system", content: "You are a local tool-using agent." },
        {
          role: "user",
          content: "请演示计划状态管理与证据传播：1) 读取 docs/agent_intro.txt，总结 Agent 系统的 3 个核心组成；2) 再读取 docs/tool_calling.md，总结 Tool Calling 闭环的 3 个关键机制，并指出其中哪 1 条与第1步最相关；3) 最后读取 tables/results.csv，指出最值得关注的 2 个字段，并说明它更适合用于展示“结果统计”还是“过程追踪”。要求按步骤推进；每一步只基于当前真实工具证据完成，若证据有限请直接说明限制，不要额外搜索无关文件。"
        },
      ],
    },
    notes: {
      mock: "确定性模拟推理，无需加载模型，适合验收现场快速展示。",
      prompt_json: "把 Tools Schema 写入提示词，真实本地模型输出 JSON 工具调用。",
      native_tools: "通过模型原生 tools 参数传入 Schema，需要模型后端支持原生工具调用。",
      adaptive: "自适应模式会根据任务复杂度在不同工具调用/执行策略之间选择更合适的方案；第 5 模块下用于批量评测自适应策略表现。",
      live_eval: "实时批量评测会自动依次运行多个模型 Profile，并汇总不同模式下的工具调用成功率与 Token 使用量。",
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
    const b4ModeOptions = `<option>mock</option><option>prompt_json</option><option>native_tools</option>`;
    const b4CompareModeOptions = `<option value="mock">mock</option><option value="live_compare">实时对比</option>`;
    const b4BatchEvalModeOptions = `<option value="prompt_json">prompt_json</option><option value="native_tools">native_tools</option><option value="adaptive">adaptive</option>`;
    return field("功能演示模块", `<select id="b4-action"><option value="generation">第一模块：单轮 AI 消息生成</option><option value="plan_execute">第二模块：多轮执行策略（Plan-and-Execute / ReAct / 自适应）</option><option value="model_routing">第三模块：模型配置与动态切换</option><option value="injection_compare">第四模块：工具调用模式对比（Prompt 注入 vs. 原生工具 Schema vs. 自适应路由）</option><option value="batch_eval">第五模块：批量评估与性能统计</option><option value="execution_engine">第六模块：计划状态管理与证据传播</option></select>`) +
      `<div id="b4-message-variant-group">${field("Messages 场景", `<select id="b4-message-variant"><option value="generation_multi_tool_calls">测试多个 tool_calls</option><option value="generation_multi_tool_messages">测试接收多个 ToolMessage</option></select>`)}</div>` +
      `<div id="b4-strategy-group" hidden>${field("执行策略", `<select id="b4-strategy"><option value="plan_execute">Plan-and-Execute</option><option value="react_one_round">ReAct</option><option value="adaptive">自适应</option></select>`)}</div>` +
      `<div class="inline-options" id="b4-options-row">${field("推理模式", `<select id="mode" data-default-options="${encodeURIComponent(b4ModeOptions)}" data-compare-options="${encodeURIComponent(b4CompareModeOptions)}" data-batch-options="${encodeURIComponent(b4BatchEvalModeOptions)}">${b4ModeOptions}</select>`)}<div id="b4-profile-group">${field("模型 Profile", `<select id="model-profile"><option value="qwen_4b">Qwen3.5-4B</option><option value="qwen_7b">Qwen2.5-7B</option></select>`)}</div></div>` +
      `<div id="b4-max-plan-group">${field("最大计划步骤", `<input id="max-plan-steps" type="number" min="1" max="10" value="4">`)}</div>` +
      `<p class="example-caption" id="mode-note">${examples.b4.notes.mock}</p>` +
      `<div class="composite-note" id="b4-feature-card"><strong>当前验证目标：</strong>第一模块 ` +
      `以 <code>generate_ai_message</code> 为核心，验证 <code>mock</code>、<code>prompt_json</code>、<code>native_tools</code> 三种模式下的单轮 AI 消息生成能力，重点看是否支持单轮返回多个 <code>tool_calls</code>、接收多个 <code>ToolMessage</code>、解析失败自动重试，以及 <code>compress_tool_messages</code> 的上下文压缩能力。</div>` +
      field("Messages 数组", `<textarea id="json">${jsonText(examples.b4.messageSets.generation_multi_tool_calls)}</textarea>`);
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
    const actionNotes = {
      generation: "第一模块先验证 generate_ai_message：可在下方切换 Messages 场景，分别观察多个 tool_calls 和接收多个 ToolMessage 两类输入。",
      plan_execute: "第二模块验证多轮执行策略：可切换 Plan-and-Execute、ReAct、自适应三种方式。",
      model_routing: "第三模块验证 model_pool 与 routing 配置，并展示自动路由和强制 Profile 的差异。",
      injection_compare: "第四模块对比 prompt_json、native_tools 与 adaptive 三种工具调用/路由方案的结构化输出效果。",
      batch_eval: "第五模块批量运行多个 Profile 与 Mode 组合，并统计工具调用成功率、参数完整性、结构化输出成功率与 Token/耗时。",
      execution_engine: "第六模块验证计划状态管理、步骤证据传播与执行产物落盘。"
    };
    const variantNotes = {
      generation_multi_tool_calls: "当前 Messages 用于测试模型是否能在单轮中返回多个 tool_calls。",
      generation_multi_tool_messages: "当前 Messages 用于测试模型接收多个 ToolMessage 后如何继续生成。",
    };
    const strategyNotes = {
      plan_execute: "当前策略为 Plan-and-Execute：先生成计划，再按步骤执行。",
      react_one_round: "当前策略为 ReAct：采用单轮工具决策与执行。",
      adaptive: "当前策略为自适应：自动判断任务复杂度并选择 ReAct 或 Plan-and-Execute。",
    };
    const featureCards = {
      generation: {
        generation_multi_tool_calls: "第一模块以 <code>generate_ai_message</code> 为核心，验证 <code>mock</code>、<code>prompt_json</code>、<code>native_tools</code> 三种模式下的单轮 AI 消息生成能力，重点看是否支持单轮返回多个 <code>tool_calls</code>、解析失败自动重试，以及 <code>compress_tool_messages</code> 的上下文压缩能力。",
        generation_multi_tool_messages: "第一模块以 <code>generate_ai_message</code> 为核心，验证模型在接收多个 <code>ToolMessage</code> 后是否能够继续发起工具调用或直接收敛答案，同时观察重试与工具结果压缩能力。 ",
      },
      plan_execute: "第二模块围绕 <code>run_plan_execute</code>、<code>run_react_one_round_execute</code>、<code>run_adaptive_execute</code> 三种多轮执行策略展开：既验证 Plan-and-Execute 的显式计划-执行循环、<code>plan_state</code> 推进、<code>STEP_DONE</code>/<code>STEP_FAIL</code>/<code>ASK_USER</code> 等状态管理能力，也验证 ReAct 在轻量任务下用更少轮次快速收敛的能力，以及自适应路由在 ReAct 与 Plan-and-Execute 之间按任务复杂度自动选择策略的能力。",
      model_routing: "第三模块围绕 <code>model_pool</code> 与 <code>routing</code> 配置展开，验证 <code>_resolve_model_profile</code>、<code>_merged_model_settings</code> 与 <code>forced_profile</code> 的行为，展示当前任务在 <code>direct</code> / <code>plan</code> / <code>execute</code> 阶段下的自动 Profile 解析结果，以及手动指定 Profile 后的覆盖效果。",
      injection_compare: "第四模块以 <code>compare_tools_injection_modes</code> 为核心，对比 <code>prompt_json</code>、<code>native_tools</code> 与 <code>adaptive</code> 三种方案：前两者分别代表固定的 Prompt 注入和原生工具 Schema 接线，自适应模式则根据任务复杂度在二者之间动态切换。重点观察结构化输出成功率、工具调用正确性、参数完整性、Token 开销以及自适应路由命中的策略。",
      batch_eval: "第五模块围绕 <code>run_batch_evaluation</code> 与 <code>run_batch_plan_execute_evaluation</code> 展开：读取预定义评测用例，自动运行多个模型 Profile 与 Mode 组合，并统计 <code>tool_match</code>、<code>args_complete</code>、结构化输出成功率、输入/输出 Token 与耗时。该模块的多轮评测部分已包含 <code>adaptive</code>，用于对比 <code>prompt_json</code>、<code>native_tools</code> 与自适应策略在批量任务上的表现。",
      execution_engine: "第六模块围绕 <code>plan_state</code>、步骤证据与执行轨迹展开，复用 <code>run_plan_execute</code> 验证计划步骤的 <code>pending / completed / failed</code> 状态推进、<code>ToolMessage</code> 证据向后续步骤传播、证据不足时的守卫，以及 <code>trace.json</code>、<code>plan_preview.md</code>、<code>progress.md</code> 三类产物是否稳定生成。",
    };
    const resolveFeatureCard = (action, variant, strategy) => {
      if (action === "generation") return featureCards.generation[variant] || featureCards.generation.generation_multi_tool_calls;
      if (action === "plan_execute") return featureCards.plan_execute;
      return featureCards[action] || "";
    };
    const syncB4State = () => {
      const action = el("b4-action").value;
      const modeSelect = el("mode");
      const previousMode = modeSelect.value;
      const defaultModeOptions = decodeURIComponent(modeSelect.dataset.defaultOptions || "");
      const compareModeOptions = decodeURIComponent(modeSelect.dataset.compareOptions || "");
      const batchModeOptions = decodeURIComponent(modeSelect.dataset.batchOptions || "");
      const expectedModeOptions = action === "batch_eval" ? batchModeOptions : (action === "injection_compare" ? compareModeOptions : defaultModeOptions);
      if (modeSelect.innerHTML !== expectedModeOptions) {
        modeSelect.innerHTML = expectedModeOptions;
        const availableModes = Array.from(modeSelect.options).map((item) => item.value);
        modeSelect.value = availableModes.includes(previousMode) ? previousMode : (action === "batch_eval" ? "prompt_json" : "mock");
      }
      const mode = modeSelect.value;
      const variantGroup = el("b4-message-variant-group");
      const strategyGroup = el("b4-strategy-group");
      const maxPlanGroup = el("b4-max-plan-group");
      const profileGroup = el("b4-profile-group");
      const optionsRow = el("b4-options-row");
      const jsonInput = el("json");
      let batchModelsNote = el("b4-batch-models-note");
      let batchCasesNote = el("b4-batch-cases-note");
      if (!batchModelsNote && optionsRow) {
        batchModelsNote = document.createElement("div");
        batchModelsNote.id = "b4-batch-models-note";
        batchModelsNote.innerHTML = field("评测模型", `<input type="text" value="Qwen3.5-4B + Qwen2.5-7B（自动依次运行）" readonly>`);
        optionsRow.appendChild(batchModelsNote);
      }
      if (!batchCasesNote && jsonInput) {
        batchCasesNote = document.createElement("p");
        batchCasesNote.id = "b4-batch-cases-note";
        batchCasesNote.className = "example-caption";
        jsonInput.insertAdjacentElement("afterend", batchCasesNote);
      }
      const featureCard = el("b4-feature-card");
      const variant = el("b4-message-variant").value;
      const strategy = el("b4-strategy").value;
      const isGeneration = action === "generation";
      const isStrategyModule = action === "plan_execute";
      const isBatchEval = action === "batch_eval";
      variantGroup.hidden = !isGeneration;
      strategyGroup.hidden = !isStrategyModule;
      maxPlanGroup.hidden = !["plan_execute", "batch_eval", "execution_engine"].includes(action);
      if (profileGroup) profileGroup.style.display = isBatchEval ? "none" : "";
      if (batchModelsNote) batchModelsNote.style.display = isBatchEval ? "" : "none";
      const batchEvalPreview = {
        selected_mode: mode,
        single_turn_cases_path: mode === "adaptive" ? null : "agent/data/messages/eval_cases_feature5.json",
        multi_turn_cases_path: "agent/data/messages/eval_cases_feature5_extended.json",
        profiles: ["qwen_4b", "qwen_7b"],
        single_turn_modes: mode === "adaptive" ? [] : [mode],
        multi_turn_modes: [mode],
        note: "第五模块实际读取预定义评测文件；这里展示的是批量评测入口，不是手工 Messages 数组。",
      };
      if (isBatchEval) {
        batchEvalPreview.note = mode === "adaptive"
          ? "第五模块当前按 adaptive 模式对两个模型做批量对比；adaptive 只评测多轮扩展用例。"
          : "第五模块当前按所选推理模式对两个模型做批量对比；single_turn 与 multi_turn 都会使用同一 mode。";
      }
      const messageSet = isBatchEval
        ? batchEvalPreview
        : (isGeneration
        ? (examples.b4.messageSets[variant] || examples.b4.messageSets.generation_multi_tool_calls)
          : (isStrategyModule
          ? (examples.b4.messageSets[strategy] || examples.b4.messageSets.plan_execute)
          : (examples.b4.messageSets[action] || examples.b4.messageSets.injection_compare)));
      jsonInput.value = jsonText(messageSet);
      jsonInput.readOnly = isBatchEval;
      if (batchCasesNote) {
        batchCasesNote.textContent = isBatchEval
          ? "批量评测实际会读取两个文件：single_turn 使用 eval_cases_feature5.json，multi_turn / adaptive 使用 eval_cases_feature5_extended.json。"
          : "";
        batchCasesNote.hidden = !isBatchEval;
        if (isBatchEval) {
          batchCasesNote.textContent = mode === "adaptive"
            ? "当前模式为 adaptive：只读取 eval_cases_feature5_extended.json，对比不同模型在多轮/自适应任务上的结果。"
            : `当前模式为 ${mode}：读取 eval_cases_feature5.json 与 eval_cases_feature5_extended.json，对比不同模型在同一推理模式下的批量结果。`;
        }
      }
      featureCard.innerHTML = `<strong>当前验证目标：</strong>${resolveFeatureCard(action, variant, strategy)}`;
      if (action === "model_routing") {
        el("mode-note").textContent = `${actionNotes[action]} 当前会读取 model.yaml 中的 routing / model_pool 配置，并在非 mock 模式下额外做一次实时验证。 ${examples.b4.notes[mode]}`;
      } else {
        const variantNote = isGeneration ? ` ${variantNotes[variant]}` : "";
        const strategyNote = isStrategyModule ? ` ${strategyNotes[strategy]}` : "";
        el("mode-note").textContent = `${actionNotes[action]}${variantNote}${strategyNote} ${examples.b4.notes[mode]}`;
      }
    };
    el("mode").addEventListener("change", syncB4State);
    el("b4-action").addEventListener("change", (event) => {
      const action = event.target.value;
      if (["plan_execute", "execution_engine"].includes(action)) el("mode").value = "mock";
      syncB4State();
    });
    el("b4-message-variant").addEventListener("change", syncB4State);
    el("b4-strategy").addEventListener("change", syncB4State);
    syncB4State();
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
  if (state.module === "b4") {
    const moduleAction = el("b4-action").value;
    const finalAction = moduleAction === "plan_execute" ? el("b4-strategy").value : moduleAction;
    if (moduleAction === "batch_eval") {
      return {
        action: finalAction,
        mode: el("mode").value,
        profiles: ["qwen_4b", "qwen_7b"],
        max_plan_steps: Number(el("max-plan-steps").value),
      };
    }
    return { action: finalAction, mode: el("mode").value, profile: el("model-profile").value, max_plan_steps: Number(el("max-plan-steps").value), messages: JSON.parse(el("json").value) };
  }
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

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMetricValue(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "N/A";
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  const parsed = Number(value);
  if (Number.isFinite(parsed)) {
    return Number.isInteger(parsed) ? String(parsed) : parsed.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function renderB4ComparisonTableLegacy(comparison, dataSource = "") {
  const modeLabels = { prompt_json: "Prompt 注入", native_tools: "原生工具 Schema", adaptive: "自适应路由" };
  const modeOrder = ["prompt_json", "native_tools", "adaptive"];
  const rows = modeOrder
    .filter((key) => comparison && typeof comparison[key] === "object" && comparison[key] !== null)
    .map((key) => {
      const item = comparison[key] || {};
      const strategy = item.strategy === "react_one_round" ? "ReAct" : (item.strategy === "plan_execute" ? "Plan-and-Execute" : "固定模式");
      const adaptiveNote = key === "adaptive"
        ? `路由 ${strategy} · 命中 ${item.selected_mode || "N/A"} · 复杂度 ${item.complexity || "N/A"} · conf ${formatMetricValue(item.confidence, 2)} · LLM ${formatMetricValue(item.llm_call_count, 0)} · 工具轮次 ${formatMetricValue(item.tool_rounds_used, 0)}`
        : "固定方案";
      return `<tr>
        <td>${escapeHtml(modeLabels[key] || key)}</td>
        <td>${escapeHtml(item.status || (item.structured_output_success ? "success" : "failed"))}</td>
        <td>${item.structured_output_success ? "成功" : "失败"}</td>
        <td>${item.tool_name_correct ? "是" : "否"}</td>
        <td>${item.args_complete ? "是" : "否"}</td>
        <td>${escapeHtml(formatMetricValue(item.tool_call_count, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.input_tokens, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.output_tokens, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.elapsed_seconds, 4))}</td>
        <td>${escapeHtml(adaptiveNote)}</td>
      </tr>`;
    })
    .join("");
  if (!rows) return "";
  return `<div class="detail-card">
    <div class="detail-card-head">
      <strong>工具调用模式对比表</strong>
      <span>${escapeHtml(dataSource || "comparison.json")}</span>
    </div>
    <div class="detail-table-wrap">
      <table class="comparison-table">
        <thead>
          <tr>
            <th>方案</th>
            <th>状态</th>
            <th>结构化输出</th>
            <th>工具正确</th>
            <th>参数完整</th>
            <th>工具调用数</th>
            <th>输入 Token</th>
            <th>输出 Token</th>
            <th>耗时(s)</th>
            <th>补充说明</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

function renderB4ComparisonTable(comparison, dataSource = "", params = {}) {
  const safeComparison = comparison && typeof comparison === "object" ? { ...comparison } : {};
  if (!safeComparison.adaptive || typeof safeComparison.adaptive !== "object") {
    safeComparison.adaptive = {
      status: params.adaptive_success ? "success" : "unavailable",
      structured_output_success: !!params.adaptive_success,
      tool_name_correct: !!params.adaptive_tool_name_correct,
      args_complete: !!params.adaptive_args_complete,
      tool_call_count: params.adaptive_tool_call_count,
      input_tokens: params.adaptive_input_tokens,
      output_tokens: params.adaptive_output_tokens,
      elapsed_seconds: params.adaptive_elapsed_seconds,
      selected_mode: params.adaptive_selected_mode,
      strategy: params.adaptive_strategy,
      complexity: params.adaptive_complexity,
      confidence: params.adaptive_confidence,
      llm_call_count: params.adaptive_llm_call_count,
      tool_rounds_used: params.adaptive_tool_rounds_used,
    };
  }

  const modeLabels = { prompt_json: "Prompt 注入", native_tools: "原生工具 Schema", adaptive: "自适应路由" };
  const modeOrder = ["prompt_json", "native_tools", "adaptive"];
  const rows = modeOrder
    .filter((key) => safeComparison && typeof safeComparison[key] === "object" && safeComparison[key] !== null)
    .map((key) => {
      const item = safeComparison[key] || {};
      const strategy = item.strategy === "react_one_round" ? "ReAct" : (item.strategy === "plan_execute" ? "Plan-and-Execute" : "固定模式");
      const adaptiveNote = key === "adaptive"
        ? `路由 ${strategy} · 命中 ${item.selected_mode || "N/A"} · 复杂度 ${item.complexity || "N/A"} · conf ${formatMetricValue(item.confidence, 2)} · LLM ${formatMetricValue(item.llm_call_count, 0)} · 工具轮次 ${formatMetricValue(item.tool_rounds_used, 0)}`
        : "固定方案";
      return `<tr>
        <td>${escapeHtml(modeLabels[key] || key)}</td>
        <td>${escapeHtml(item.status || (item.structured_output_success ? "success" : "failed"))}</td>
        <td>${item.structured_output_success ? "成功" : "失败"}</td>
        <td>${item.tool_name_correct ? "是" : "否"}</td>
        <td>${item.args_complete ? "是" : "否"}</td>
        <td>${escapeHtml(formatMetricValue(item.tool_call_count, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.input_tokens, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.output_tokens, 0))}</td>
        <td>${escapeHtml(formatMetricValue(item.elapsed_seconds, 4))}</td>
        <td>${escapeHtml(adaptiveNote)}</td>
      </tr>`;
    })
    .join("");
  if (!rows) return "";
  return `<div class="detail-card">
    <div class="detail-card-head">
      <strong>工具调用模式对比表</strong>
      <span>${escapeHtml(dataSource || "comparison.json")}</span>
    </div>
    <div class="detail-table-wrap">
      <table class="comparison-table">
        <thead>
          <tr>
            <th>方案</th>
            <th>状态</th>
            <th>结构化输出</th>
            <th>工具正确</th>
            <th>参数完整</th>
            <th>工具调用数</th>
            <th>输入 Token</th>
            <th>输出 Token</th>
            <th>耗时(s)</th>
            <th>补充说明</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

function renderB4BatchEvalTablesLegacy1(evaluation, dataSource = "") {
  const summary = evaluation && typeof evaluation === "object" && evaluation.combined_summary && typeof evaluation.combined_summary === "object"
    ? evaluation.combined_summary
    : {};
  const groups = {};
  Object.entries(summary).forEach(([label, item]) => {
    if (!item || typeof item !== "object") return;
    const parts = String(label).split(" 路 ");
    const scope = item.scope || parts[0] || "未分类";
    const profileLabel = parts[1] || item.profile || "default";
    const mode = item.mode || parts[2] || "unknown";
    if (!groups[profileLabel]) groups[profileLabel] = [];
    groups[profileLabel].push({ scope, mode, ...item });
  });

  const scopeOrder = { "单轮": 0, "多轮/自适应": 1 };
  const modeOrder = { prompt_json: 0, native_tools: 1, adaptive: 2 };
  const cards = Object.entries(groups).map(([profileLabel, rows]) => {
    const sortedRows = rows.slice().sort((left, right) => {
      const scopeDelta = (scopeOrder[left.scope] ?? 9) - (scopeOrder[right.scope] ?? 9);
      if (scopeDelta !== 0) return scopeDelta;
      return (modeOrder[left.mode] ?? 9) - (modeOrder[right.mode] ?? 9);
    });
    const tableRows = sortedRows.map((item) => `
      <tr>
        <td>${escapeHtml(item.scope || "未分类")}</td>
        <td>${escapeHtml(item.mode || "unknown")}</td>
        <td>${escapeHtml(formatMetricValue(item.cases, 0))}</td>
        <td>${escapeHtml(`${(Number(item.tool_match_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(`${(Number(item.structured_output_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_input_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_output_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_elapsed_seconds, 4))}</td>
      </tr>
    `).join("");
    return `<div class="detail-card eval-model-card">
      <div class="detail-card-head">
        <strong>${escapeHtml(profileLabel)}</strong>
        <span>${escapeHtml(dataSource || "eval_summary.json")}</span>
      </div>
      <div class="detail-table-wrap">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>评测范围</th>
              <th>模式</th>
              <th>样例数</th>
              <th>工具调用成功率</th>
              <th>结构化输出率</th>
              <th>平均输入 Token</th>
              <th>平均输出 Token</th>
              <th>平均耗时(s)</th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>`;
  }).join("");

  if (!cards) return "";
  return `<div class="eval-grid">${cards}</div>`;
}

function renderB4BatchEvalTablesLegacy2(evaluation, dataSource = "") {
  const summary = evaluation && typeof evaluation === "object" && evaluation.combined_summary && typeof evaluation.combined_summary === "object"
    ? evaluation.combined_summary
    : {};
  const groups = {};
  Object.entries(summary).forEach(([label, item]) => {
    if (!item || typeof item !== "object") return;
    const normalizedLabel = String(label).replace(/鍗曡疆/g, "单轮").replace(/澶氳疆\/鑷€傚簲/g, "多轮/自适应");
    const parts = normalizedLabel.split(" 路 ");
    const scope = item.scope || parts[0] || "未分类";
    const profileLabel = parts[1] || item.profile || "default";
    const mode = item.mode || parts[2] || "unknown";
    if (!groups[profileLabel]) groups[profileLabel] = [];
    groups[profileLabel].push({ scope, mode, ...item });
  });

  const scopeOrder = { "单轮": 0, "多轮/自适应": 1 };
  const modeOrder = { prompt_json: 0, native_tools: 1, adaptive: 2 };
  const cards = Object.entries(groups).map(([profileLabel, rows]) => {
    const sortedRows = rows.slice().sort((left, right) => {
      const scopeDelta = (scopeOrder[left.scope] ?? 9) - (scopeOrder[right.scope] ?? 9);
      if (scopeDelta !== 0) return scopeDelta;
      return (modeOrder[left.mode] ?? 9) - (modeOrder[right.mode] ?? 9);
    });
    const tableRows = sortedRows.map((item) => `
      <tr>
        <td>${escapeHtml(item.scope || "未分类")}</td>
        <td>${escapeHtml(item.mode || "unknown")}</td>
        <td>${escapeHtml(formatMetricValue(item.cases, 0))}</td>
        <td>${escapeHtml(`${(Number(item.tool_match_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(`${(Number(item.structured_output_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_input_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_output_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_elapsed_seconds, 4))}</td>
      </tr>
    `).join("");
    return `<div class="detail-card eval-model-card">
      <div class="detail-card-head">
        <strong>${escapeHtml(profileLabel)}</strong>
        <span>${escapeHtml(dataSource || "eval_summary.json")}</span>
      </div>
      <div class="detail-table-wrap">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>评测范围</th>
              <th>模式</th>
              <th>样例数</th>
              <th>工具调用成功率</th>
              <th>结构化输出率</th>
              <th>平均输入 Token</th>
              <th>平均输出 Token</th>
              <th>平均耗时(s)</th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>`;
  }).join("");

  if (!cards) return "";
  return `<div class="eval-grid">${cards}</div>`;
}

function renderB4BatchEvalTables(evaluation, dataSource = "", judgementNote = "") {
  const summary = evaluation && typeof evaluation === "object" && evaluation.combined_summary && typeof evaluation.combined_summary === "object"
    ? evaluation.combined_summary
    : {};
  const groups = {};
  const knownModes = ["prompt_json", "native_tools", "adaptive"];

  Object.entries(summary).forEach(([label, item]) => {
    if (!item || typeof item !== "object") return;
    const resolvedMode = item.mode || knownModes.find((mode) => String(label).endsWith(mode)) || "unknown";
    const trimmedLabel = resolvedMode === "unknown"
      ? String(label)
      : String(label).slice(0, String(label).lastIndexOf(resolvedMode)).trim();
    const segments = trimmedLabel.split(/\s+\u8DEF\s+/);
    const scope = item.scope || segments[0] || "未分类";
    const profileLabel = segments[1] || item.profile || "default";
    if (!groups[profileLabel]) groups[profileLabel] = [];
    groups[profileLabel].push({ scope, mode: resolvedMode, ...item });
  });

  const scopeOrder = { "单轮": 0, "多轮/自适应": 1 };
  const modeOrder = { prompt_json: 0, native_tools: 1, adaptive: 2 };
  const cards = Object.entries(groups).map(([profileLabel, rows]) => {
    const sortedRows = rows.slice().sort((left, right) => {
      const scopeDelta = (scopeOrder[left.scope] ?? 9) - (scopeOrder[right.scope] ?? 9);
      if (scopeDelta !== 0) return scopeDelta;
      return (modeOrder[left.mode] ?? 9) - (modeOrder[right.mode] ?? 9);
    });
    const tableRows = sortedRows.map((item) => `
      <tr>
        <td>${escapeHtml(item.scope || "未分类")}</td>
        <td>${escapeHtml(item.mode || "unknown")}</td>
        <td>${escapeHtml(formatMetricValue(item.cases, 0))}</td>
        <td>${escapeHtml(`${(Number(item.tool_match_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(`${(Number(item.structured_output_rate || 0) * 100).toFixed(1)}%`)}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_input_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_output_tokens, 1))}</td>
        <td>${escapeHtml(formatMetricValue(item.avg_elapsed_seconds, 4))}</td>
      </tr>
    `).join("");

    return `<div class="detail-card eval-model-card">
      <div class="detail-card-head">
        <strong>${escapeHtml(profileLabel)}</strong>
        <span>${escapeHtml(dataSource || "eval_summary.json")}</span>
      </div>
      <div class="detail-table-wrap">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>评测范围</th>
              <th>模式</th>
              <th>样例数</th>
              <th>工具调用成功率</th>
              <th>结构化输出率</th>
              <th>平均输入 Token</th>
              <th>平均输出 Token</th>
              <th>平均耗时(s)</th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>`;
  }).join("");

  if (!cards) return "";
  const noteHtml = judgementNote ? `<p class="example-caption">${escapeHtml(judgementNote)}</p>` : "";
  return `${noteHtml}<div class="eval-grid">${cards}</div>`;
}

function renderResultParameters(data) {
  const section = el("result-insights");
  const detail = el("result-detail");
  detail.hidden = true;
  detail.innerHTML = "";
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
      const evaluation = data.output.evaluation || {};
      const profileCount = Array.isArray(params.profiles) ? params.profiles.length : 0;
      const selectedMode = params.selected_mode || (Array.isArray(params.modes) && params.modes[0]) || "unknown";
      const modeCount = 1;
      const summaryRows = params.model_comparison && typeof params.model_comparison === "object" ? Object.values(params.model_comparison) : [];
      const bestToolRate = summaryRows.length ? Math.max(...summaryRows.map((item) => Number(item.tool_match_rate || 0))) : 0;
      const inputCandidates = summaryRows.map((item) => Number(item.avg_input_tokens)).filter((value) => Number.isFinite(value) && value > 0);
      const lowestInput = inputCandidates.length ? Math.min(...inputCandidates) : null;
      el("parameter-row").innerHTML =
        parameterItem("对比模型", `${profileCount} 个`, "自动依次运行多个模型 Profile") +
        parameterItem("评测模式", `${modeCount} 种`, (params.modes || []).join(" / ")) +
        parameterItem("最佳工具成功率", `${(bestToolRate * 100).toFixed(1)}%`, "按 tool_match_rate 汇总") +
        parameterItem("最低平均输入 Token", lowestInput == null ? "N/A" : lowestInput.toFixed(1), params.data_source || "批量评测");
      el("parameter-row").innerHTML += parameterItem("当前模式", selectedMode, "同一推理模式下比较不同模型的批量结果");
      detail.innerHTML = renderB4BatchEvalTables(evaluation, params.data_source, params.judgement_note || "");
      detail.hidden = !detail.innerHTML;
    } else if (params.action === "model_routing") {
      el("parameter-row").innerHTML =
        parameterItem("当前阶段", params.current_phase || "direct", `任务分类 ${params.task_profile || "default"}`) +
        parameterItem("当前模型", params.current_model_target || "未解析到模型路径", params.live_model_target ? "本次实时验证实际命中模型" : "当前按自动/强制 Profile 推导") +
        parameterItem("自动 Profile", params.auto_profile || "default", params.auto_model_target || "未解析到模型路径") +
        parameterItem("强制 Profile", params.forced_profile || "未指定", params.forced_model_target || "当前以下拉选择为准") +
        parameterItem("路由配置", `${params.default_profile || "default"} / ${params.plan_profile || "plan"} / ${params.execute_profile || "execute"}`, "default / plan / execute") +
        parameterItem("实时验证", params.live_selected_profile || "未执行", params.live_model_target || params.data_source);
    } else if (params.action === "injection_compare") {
      const comparison = data.output.comparison || {};
      const adaptiveRouteNote = params.adaptive_strategy
        ? `${params.adaptive_strategy === "react_one_round" ? "ReAct" : "Plan-and-Execute"} · 命中 ${params.adaptive_selected_mode || "N/A"}`
        : "等待自适应路由结果";
      el("parameter-row").innerHTML =
        parameterItem("Prompt 注入", params.prompt_json_success ? "成功" : "失败", `工具正确 ${params.prompt_json_tool_name_correct ? "是" : "否"} · 参数完整 ${params.prompt_json_args_complete ? "是" : "否"}`) +
        parameterItem("Prompt 指标", `${params.prompt_json_tool_call_count ?? 0} 次调用`, `input ${params.prompt_json_input_tokens ?? "N/A"} · output ${params.prompt_json_output_tokens ?? "N/A"} · ${params.prompt_json_elapsed_seconds ?? "N/A"} s`) +
        parameterItem("原生 Schema", params.native_tools_success ? "成功" : "失败", `工具正确 ${params.native_tools_tool_name_correct ? "是" : "否"} · 参数完整 ${params.native_tools_args_complete ? "是" : "否"}`) +
        parameterItem("Schema 指标", `${params.native_tools_tool_call_count ?? 0} 次调用`, `input ${params.native_tools_input_tokens ?? "N/A"} · output ${params.native_tools_output_tokens ?? "N/A"} · ${params.native_tools_elapsed_seconds ?? "N/A"} s`) +
        parameterItem("自适应路由", params.adaptive_success ? "成功" : "失败", adaptiveRouteNote) +
        parameterItem("Adaptive 指标", `${params.adaptive_tool_call_count ?? 0} 次调用`, `input ${params.adaptive_input_tokens ?? "N/A"} · output ${params.adaptive_output_tokens ?? "N/A"} · ${params.adaptive_elapsed_seconds ?? "N/A"} s`) +
        parameterItem("数据来源", params.data_source);
      detail.innerHTML = renderB4ComparisonTable(comparison, params.data_source, params);
      detail.hidden = !detail.innerHTML;
    } else if (params.action === "execution_engine") {
      el("parameter-row").innerHTML =
        parameterItem("计划步骤", `${params.total_steps} 步`, `${params.completed_steps} 已完成 / ${params.failed_steps} 失败 / ${params.pending_steps} 待执行`) +
        parameterItem("证据策略", params.evidence_policy || "lite", params.evidence_guard_enabled ? "已启用证据充足性检查" : "未启用") +
        parameterItem("执行产物", `${params.trace_written ? "trace" : "无"} / ${params.plan_preview_written ? "plan" : "无"} / ${params.progress_written ? "progress" : "无"}`, "trace.json / plan_preview.md / progress.md") +
        parameterItem("工具轮次", params.tool_rounds_used ?? 0) +
        parameterItem("LLM 调用", params.llm_call_count ?? 0) +
        parameterItem("模型 Profile", params.selected_profile) +
        parameterItem("数据来源", params.data_source);
    } else if (params.action === "plan_execute") {
      el("parameter-row").innerHTML = parameterItem("计划步骤", params.plan_steps) + parameterItem("计划状态", params.plan_status) + parameterItem("LLM 调用", params.llm_call_count) + parameterItem("工具轮次", params.tool_rounds_used) + parameterItem("模型 Profile", params.selected_profile) + parameterItem("数据来源", params.data_source);
    } else {
      el("parameter-row").innerHTML =
        parameterItem("验证模块", params.demo_module || "第一模块") +
        parameterItem("核心函数", params.core_function || "generate_ai_message") +
        parameterItem("生成 ToolCall", params.generated_tool_calls ?? 0) +
        parameterItem("接收 ToolMessage", params.received_tool_messages ?? 0) +
        parameterItem("后续 ToolCall", params.followup_tool_calls ?? 0) +
        parameterItem("重试机制", params.repair_attempted ? `已触发，累计 ${params.attempt_count ?? 0} 次` : `可用，本次 ${params.attempt_count ?? 0} 次生成`) +
        parameterItem("结果压缩", params.compression_feature || "compress_tool_messages", "用于控制 ToolMessage 上下文长度") +
        parameterItem("模型 Profile", params.selected_profile) +
        parameterItem("Schema 模式", params.mode) +
        parameterItem("Token", `${params.input_tokens ?? "N/A"} / ${params.output_tokens ?? "N/A"}`, "input / output");
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
