# B1-B5 Agent Module Studio

本目录提供一个统一 Web 演示系统，网页内包含 B1-B5 五个可切换子页面。每个子页面直接调用对应模块公开函数，具有明确输入、标准输出、接口去向和独立运行产物。

## 统一启动

在项目根目录执行：

```bash
python module_demos/run_all_demo.py
```

浏览器打开 `http://127.0.0.1:8100/`，也可用以下地址直接进入指定子页面：

统一入口默认监听 `0.0.0.0:8100`。同一局域网内其他成员可通过 `http://<运行电脑IP>:8100/` 同时访问；服务器部署时使用服务器 IP 或 VS Code 转发后的共享地址。

每个浏览器会生成独立会话 ID，页面左下角显示在线人数和当前身份。运行产物按访问者隔离保存：

```text
module_demos/outputs/<b1-b5>/<client_id>/<timestamp>/
```

```text
http://127.0.0.1:8100/#b1
http://127.0.0.1:8100/#b2
http://127.0.0.1:8100/#b3
http://127.0.0.1:8100/#b4
http://127.0.0.1:8100/#b5
```

通过 VS Code Remote SSH 在服务器运行时，在“端口”面板转发 `8100`。旧的 `run_b1_demo.py` 至 `run_b5_demo.py` 仍可用于单模块独立启动。

## 统一 API

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `GET /api/modules` | 无 | B1-B5 元信息、负责人和接口流向 |
| `POST /api/modules/b1/run` | 任务描述 | `messages`、`trace`、`final_answer` |
| `POST /api/modules/b2/run` | Skill 名称、JSON 参数 | `SkillResult` |
| `POST /api/modules/b3/run` | action、toolset、ToolCall | Tools Schema、`ToolMessage` |
| `POST /api/modules/b4/run` | messages、mode、Tools Schema | `AIMessage`、raw record |
| `POST /api/modules/b5/run` | query、search_mode、top_k | `MemoryResult` |

通用调用示例：

```bash
curl -X POST http://127.0.0.1:8100/api/modules/b2/run \
  -H "Content-Type: application/json" \
  -d '{"skill":"calculator","args":{"expression":"23 * 17 + 9"}}'
```

## 模块协作关系

| 模块 | 输入来自 | 标准输出 | 输出交给谁 |
| --- | --- | --- | --- |
| B1 Agent Runtime | 用户、B5 的记忆、B4 的模型消息、B3 的工具消息 | `messages`、`trace`、`final_answer` | 最终回答交给前端；运行记录交给 B5 保存 |
| B2 Skill | B3 校验后的工具名和参数 | `SkillResult` | B3 封装为带 `tool_call_id` 的 `ToolMessage` |
| B3 Tool Layer | B4 产生的 `AIMessage.tool_calls` | Tools Schema、`ToolMessage` | Schema 交给 B4；工具消息交给 B1 |
| B4 Local LLM | B1 的 messages、B3 的 Tools Schema | `AIMessage`、raw record | B1 判断直接回答或把 tool_calls 转交 B3 |
| B5 Memory | B1 的用户问题；保存时接收 B1 运行结果 | `MemoryResult` | B1 将命中记忆注入模型上下文 |

完整集成链路为：`用户 → B1 → B5 → B1 → B4 → B1 → B3 → B2 → B3 → B1 → B4 → B1 → 用户/B5`。

## B1 执行示例

负责人 A。页面输入：

```text
读取 docs/agent_intro.txt，并总结三条中文要点。
```

B1 独立页使用 fixture 隔离 B2-B5，但仍完整生成 `system → user → assistant(tool_calls) → tool → assistant(final)` 消息闭环。重点展示 `trace` 中的路由、轮次和工具调用关联。

前端还提供多次 ToolCall 循环、多轮用户输入、批量任务、断点恢复、历史压缩和 System Prompt 添加/切换六种高级演示。参数行分别展示用户轮数、工具轮次、LLM 调用数、checkpoint 状态、压缩次数、Prompt 事件和批量成功数。

## B2 所有 Skill 示例

负责人 A。页面切换 Skill 时会自动载入以下可执行 JSON。

### `calculator`

```json
{"expression":"23 * 17 + 9"}
```

预期：返回计算结果 `400`、`status=success` 和 `latency_ms`。

### `file_reader`

```json
{"path":"docs/agent_intro.txt","max_chars":2000}
```

预期：返回文件内容、字符数、规范化路径和截断状态。

### `local_file_search`

```json
{"query":"Agent 工具调用","root_dir":"docs","file_types":["txt","md"],"top_k":5}
```

预期：返回匹配文件、行号、片段、得分和扫描统计。

### `table_analyzer`

```json
{"path":"tables/results.csv","max_rows_preview":5,"describe":true}
```

预期：返回行列数、列名、数据预览和数值统计。

### `format_converter`

```json
{"text":"a: 1\nb: 2","target_format":"markdown","output_filename":"converted_sample.md"}
```

预期：返回 Markdown 内容和生成文件路径。JSON 分支示例：`{"text":"{\"a\":1}","target_format":"json","output_filename":"converted_sample.json"}`。

### `read_convert_file`

```json
{"path":"docs/agent_intro.txt","max_chars":1000,"target_format":"markdown","output_filename":"agent_intro_bullets.md"}
```

预期：先读取文件，再返回读取元数据和格式转换结果。将 `target_format` 改为 `json` 可演示 JSON 分支。

### `code_executor`

```json
{
  "code":"import math\nvalues = [1, 2, 3, 4]\nprint('count', len(values))\nsum(values) + math.sqrt(16)",
  "timeout_seconds":3,
  "allowed_imports":["math"],
  "work_dir":"sandbox"
}
```

预期：在受限子进程中返回 `stdout`、最终表达式结果和沙箱策略。该 Skill 仅属于 `advanced_tools`。

## B3 所有选择示例

负责人 B。

前端“执行样例”额外提供“相同参数重复调用（缓存命中）”和“代码执行超时（可恢复错误与重试）”。执行后页面底部展示缓存命中、调用次数、失败率、重试次数、平均耗时，以及固定评测集上的 brief/detailed Schema 描述准确率。

### 动作

| 页面选择 | 示例输入 | 预期输出 |
| --- | --- | --- |
| `execute` | `tool_calls=[{"id":"call_1","name":"calculator","args":{"expression":"8*12+4"}}]` | Tools Schema 和关联 `call_1` 的 ToolMessage |
| `schema` | `tool_calls=[]` | 仅生成 Tools Schema 和 Schema 报告 |

### 工具集

| 页面选择 | 示例 | 差异 |
| --- | --- | --- |
| `basic_tools` | 执行 `calculator` ToolCall | 包含六个基础 Skill，不允许 `code_executor` |
| `advanced_tools` | `[{"id":"call_code","name":"code_executor","args":{"code":"sum(range(1,101))","timeout_seconds":3,"allowed_imports":[],"work_dir":"sandbox"}}]` | 在基础工具集上增加 `code_executor` |

### Schema 来源

| 页面选择 | 请求字段 | 预期结果 |
| --- | --- | --- |
| 不勾选“从 Python 函数签名生成” | `"auto_from_code":false` | 根据 `configs/tools.yaml` 生成 Schema |
| 勾选“从 Python 函数签名生成” | `"auto_from_code":true` | 根据 Skill 函数签名生成并输出来源报告 |

## B4 所有推理模式示例

负责人 C。三种模式均可使用以下 messages：

```json
[
  {"role":"system","content":"You are a local tool-using agent."},
  {"role":"user","content":"读取 docs/agent_intro.txt 并总结三点。"}
]
```

| 页面选择 | 执行方式 | 预期输出 |
| --- | --- | --- |
| `mock` | 不加载模型，确定性模拟 | 立即返回标准 `AIMessage`，适合现场演示 |
| `prompt_json` | Tools Schema 写入提示词后调用真实本地模型 | 模型以 JSON 形式返回 content/tool_calls |
| `native_tools` | 通过模型原生 tools 参数传入 Schema | 后端原生工具调用结果转换为标准 `AIMessage` |

`prompt_json` 和 `native_tools` 需要服务器存在 `configs/model.yaml` 指向的本地模型与 GPU/运行依赖；离线验收优先展示 `mock`。

B4 前端还提供多 ToolCall/多 ToolMessage 往返、Plan-and-Execute、tools_schema 注入方式对比、Qwen 4B/7B profile 切换和六条批量任务评测。注入方式与双模型指标默认读取服务器真实运行产物，并展示工具调用成功率、工具匹配率、输入/输出 Token 和数据来源。

## B5 所有检索选择示例

负责人 D。

执行后页面底部展示实际返回数量、前 K 个得分的降序序列、`max_memory_chars`、截断状态；向量模式额外展示向量方法、维度、相似度指标和最低分数。

| 页面选择 | query / IDs 示例 | 预期输出 |
| --- | --- | --- |
| `keyword` | query=`Agent 系统如何调用工具？` | 按关键词命中，返回匹配项和分数 |
| `vector` | query=`模型怎样使用外部能力完成任务？` | 使用 hashed bag-of-terms cosine 返回语义近似结果 |
| `auto` | query=`Agent 的模型、工具和记忆如何协作？`，IDs 为空 | 自动选择 keyword；若提供 IDs 则转为 ID 检索 |
| `id` | IDs=`mem_course_001` | 精确载入指定记忆 |

`top_k` 可选范围为 1-20，默认执行示例使用 `5`。“同时载入全局记忆”勾选时请求字段为 `use_global_memory=true`，用于加入全局记忆；取消勾选时为 `false`，只处理指定/检索结果。

## 输出目录

每次运行产物保存在：

```text
module_demos/outputs/<b1-b5>/<timestamp>/
```

验收时依次展示：模块输入、运行操作、标准 JSON 输出、接口接收方、运行产物目录和本模块在完整链路中的位置。
