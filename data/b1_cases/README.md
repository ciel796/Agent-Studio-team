# B1 测试用例运行说明

以下命令均在项目根目录执行。

## 在 PowerShell 中执行 B1 测试用例

### 执行综合批量测试

```powershell
python .\code\b1_agent_runtime_1.py `
  --batch_input .\data\b1_cases\b1_all_features_batch.json `
  --tools_config .\configs\tools.yaml `
  --memory_config .\configs\memory.yaml `
  --model_config .\configs\model.yaml `
  --outdir .\outputs\B1_test_cases
```

### 执行断点续跑测试

先执行第一阶段，使任务进入等待用户补充信息的状态：

```powershell
python .\code\b1_agent_runtime_1.py `
  --input .\data\b1_cases\b1_resume_initial.json `
  --outdir .\outputs\B1_resume_case
```

再执行第二阶段，从 `checkpoint.json` 恢复：

```powershell
python .\code\b1_agent_runtime_1.py `
  --input .\data\b1_cases\b1_resume_continue.json `
  --outdir .\outputs\B1_resume_case `
  --resume
```

## 在 ai_web 中执行测试用例

### 启动 ai_web

在服务器上的项目根目录执行：

```powershell
python .\ai_web\server.py --host 0.0.0.0 --port 8010
```

在浏览器中打开服务器地址：

```text
http://<服务器IP>:8010/
```

### 运行服务器上的批量 JSON

1. 打开左侧的“批量任务”页面。
2. 在“批量任务 JSON 文件路径”输入框中填写服务器项目内的相对路径：

```text
data/b1_cases/b1_all_features_batch.json
```

3. 保持下方“直接填写 JSON”文本框为空。
4. 点击“运行批量任务”。
5. 在页面结果区域直接查看每个任务的任务内容、运行状态、最终回答、调用统计和输出目录。

这里填写的是服务器项目中的 JSON 路径，不要填写本机桌面路径或 Windows 绝对路径。

## 专项功能测试用例

### 用例一：多轮用户输入与多次 tool_calls 循环

综合批量文件中包含两个相关任务：

- `multi_tool_loop`：同一个用户问题依次触发两轮工具调用，先调用 `file_reader`，再调用 `calculator`，最后由 LLM 生成回答。
- `multi_turn_prompt_compression`：通过 `user_inputs` 连续执行三轮用户输入，验证同一会话中的多轮处理。

使用 PowerShell 执行：

```powershell
python .\code\b1_agent_runtime_1.py `
  --batch_input .\data\b1_cases\b1_all_features_batch.json `
  --tools_config .\configs\tools.yaml `
  --memory_config .\configs\memory.yaml `
  --model_config .\configs\model.yaml `
  --outdir .\outputs\B1_test_cases
```

在 ai_web 中执行时，使用服务器 JSON：

```text
data/b1_cases/b1_all_features_batch.json
```

预期结果：

- `multi_tool_loop` 状态为 `success`。
- 页面显示任务内容“先读取 Agent 文档，再计算 6×7”。
- 页面统计显示 LLM 调用 `3` 次、工具轮次 `2` 次。
- 最终回答包含“已完成两轮工具调用”和计算结果 `42`。
- `multi_turn_prompt_compression` 显示三个用户输入，用户轮次为 `3`。

详细运行产物位于：

```text
outputs/B1_test_cases/multi_tool_loop/trace.json
outputs/B1_test_cases/multi_tool_loop/messages.json
outputs/B1_test_cases/multi_turn_prompt_compression/trace.json
```

### 用例二：断点续跑与状态恢复

第一阶段使用 `b1_resume_initial.json`。fixture 中的 LLM 会返回 `ASK_USER`，Runtime 保存 checkpoint 后进入 `needs_user`：

```powershell
python .\code\b1_agent_runtime_1.py `
  --input .\data\b1_cases\b1_resume_initial.json `
  --outdir .\outputs\B1_resume_case
```

第一阶段预期生成：

```text
outputs/B1_resume_case/checkpoint.json
outputs/B1_resume_case/pending_question.md
outputs/B1_resume_case/trace.json
```

此时 `trace.json` 中的 `status` 应为 `needs_user`。

第二阶段提供补充输入并从 checkpoint 恢复：

```powershell
python .\code\b1_agent_runtime_1.py `
  --input .\data\b1_cases\b1_resume_continue.json `
  --outdir .\outputs\B1_resume_case `
  --resume
```

预期恢复已有消息、LLM 调用位置和待处理状态，继续执行后续 LLM 步骤；最终 `status` 为 `success`，用户轮次为 `2`，回答为“两个整数相加等于 12”。

在 ai_web 对话页中，如果任务返回 `needs_user`，直接输入补充信息并再次发送。页面会携带原 `run_dir` 调用恢复接口，不会创建一条无关的新任务。如果手动中断任务，可点击该问题旁的重试按钮，从原 checkpoint 继续执行。

### 用例三：在记忆模式下将历史压缩为摘要并作为后续上下文

这个用例测试的是 ai_web 的“记忆摘要模式”，不是单次 Runtime 内的 `<compressed_history>` 压缩。

工作流程如下：

1. 第一轮回答完成后，将本轮消息和回答保存为 conversation Memory。
2. Memory 索引保存一份精简摘要；后续保存时会把旧摘要和新回答合并成累积摘要。
3. 下一轮仍使用同一个 `conversation_id` 时，ai_web 自动选择 `mem_conversation_<conversation_id>`。
4. Runtime 使用 `memory_context_mode=summary`，只把 Memory 摘要注入 System 上下文，不注入完整 Messages、Trace 或整篇 Memory 文档。
5. 页面“高级运行设置”区域显示本轮实际注入的 Memory 摘要。

#### ai_web 操作步骤

1. 打开“对话”页面并点击“清空”，创建一条新会话。
2. 展开“高级运行设置”。
3. 勾选“记忆摘要模式”。开启后后端会自动使用 `save_memory=conversation`。
4. 第一轮输入：

```text
请记住：我的项目是一个本地 Agent 系统，我偏好简洁的中文回答，并且后续优先说明工具调用过程。
```

5. 等待第一轮完成。摘要区域应显示“本轮已保存摘要，下轮将作为上下文”。
6. 不要点击“清空”，直接进行第二轮输入：

```text
请根据你记住的项目背景和回答偏好，说明这个 Agent 应该如何处理文件读取任务。
```

7. 第二轮开始后，摘要区域应显示“本轮已注入的 Memory 摘要”，其中包含第一轮保存的项目背景或回答偏好。
8. 第二轮回答应能够使用摘要中的信息，例如使用简洁中文并说明工具调用过程。

ai_web 会在浏览器中保存当前 `conversation_id`，刷新页面后仍会继续使用同一会话摘要。点击“清空”会清除该 ID 并开始新会话。

记忆摘要模式默认不加载 global Memory，只注入同一 `conversation_id` 对应的 conversation Memory；因此不会混入与当前对话无关的全局摘要。

#### 预期检查结果

- 第一轮完成后，`memory/memory_index.json` 中新增：

```text
mem_conversation_<conversation_id>
```

- 对应索引记录包含非空的 `summary`。
- 第二轮运行目录的 `runtime_input.json` 包含：

```json
{
  "save_memory": "conversation",
  "memory_summary_mode": true,
  "memory_context_mode": "summary",
  "selected_memory_ids": ["mem_conversation_<conversation_id>"]
}
```

- 第二轮运行目录的 `selected_memory.json` 中，目标文档包含 `summary` 字段。
- 第二轮 checkpoint 的首条 System 消息包含 `context_mode="summary"`。
- 注入内容不应包含 `## Messages` 或 `## Trace`，说明使用的是摘要而不是完整 Memory 文档。
- 第二轮结束后，同一 Memory 的摘要会累积合并新回答，供第三轮继续使用。

### 用例四：一次对话中添加或切换 System Prompt

本用例使用独立的服务器批处理文件：

```text
data/b1_cases/system_prompt_switch_batch.json
```

该任务包含三个连续用户轮次：

1. 第一轮使用初始模板 `prompts/local_tool_agent.txt`。
2. 第二轮通过 `mode=add` 追加“回答必须简洁”。原 System Prompt 仍然保留。
3. 第三轮通过 `mode=switch` 切换到 `prompts/strict_tool_prompt.txt`。

该文件使用真实 `integrated + prompt_json` 模型调用，不再使用 fixture 固定回答。服务器必须已经安装 `requirements-llm.txt` 并能加载 `configs/model.yaml` 指定的本地模型。

#### 在 ai_web 中执行

1. 在服务器项目根目录启动 ai_web：

```powershell
python .\ai_web\server.py --host 0.0.0.0 --port 8010
```

2. 浏览器打开：

```text
http://<服务器IP>:8010/
```

3. 点击左侧“批量任务”。
4. 在“服务器项目内 JSON 路径”输入框中填写：

```text
data/b1_cases/system_prompt_switch_batch.json
```

5. 保持下方“直接填写 JSON”文本框为空。
6. 点击“运行批量任务”。
7. 等待任务卡片 `system_prompt_add_and_switch` 出现。
8. 检查页面结果：
   - 状态应为 `success`。
   - 用户轮次应为 `3`。
   - Prompt 事件应为 `2`。
   - 页面应显示三个用户输入及模型针对三个问题生成的实际回答，而不是“第一轮完成”之类的固定文本。
   - 第三轮应先调用 `file_reader` 读取 `life_rent/requirements.md`，再给出两条硬性条件。

页面显示的输出目录类似：

```text
outputs/web_ui/batch/<运行时间>/system_prompt_add_and_switch
```

打开该目录的 `trace.json`，检查 `system_prompt_events_applied`。预期包含：

```json
[
  {
    "user_turn_index": 2,
    "mode": "add",
    "label": "brief"
  },
  {
    "user_turn_index": 3,
    "mode": "switch",
    "label": "strict"
  }
]
```

#### 直接在前端粘贴 JSON

如果不填写服务器文件路径，也可以把以下代码粘贴到“直接填写 JSON”文本框：

```json
{
  "tasks": [
    {
      "task_id": "system_prompt_add_and_switch",
      "llm_mode": "prompt_json",
      "runtime_input": {
        "conversation_id": "b1_system_prompt_switch_001",
        "execution_mode": "integrated",
        "agent_mode": "integrated",
        "agent_options": {
          "llm_mode": "prompt_json"
        },
        "user_inputs": [
          "第一轮：请用4-6句话介绍 Agent 的基本组成，并说明各部分之间的关系。",
          "第二轮：继续说明工具调用的作用，并给出一个计算器工具的简单例子。",
          "第三轮：请读取 life_rent/requirements.md，只列出其中两条基础硬性条件，并标注来源文件。"
        ],
        "system_prompt_path": "../../prompts/local_tool_agent.txt",
        "system_prompt_events": [
          {
            "user_turn_index": 2,
            "mode": "add",
            "label": "brief",
            "content": "回答必须简洁。"
          },
          {
            "user_turn_index": 3,
            "mode": "switch",
            "label": "strict",
            "system_prompt_path": "../../prompts/strict_tool_prompt.txt"
          }
        ],
        "selected_memory_ids": [],
        "use_global_memory": false,
        "toolset": "basic_tools",
        "max_turns": 3,
        "save_memory": "none"
      }
    }
  ]
}
```

直接粘贴 JSON 时，必须清空上方服务器文件路径。页面会优先使用直接填写的 JSON。
