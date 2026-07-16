from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
CODE_DIR = PROJECT_ROOT / "code"
STATIC_DIR = DEMO_DIR / "static"
OUTPUT_ROOT = DEMO_DIR / "outputs"
sys.path.insert(0, str(CODE_DIR))

MODULE_META = {
    "b1": {
        "title": "B1 Agent Runtime Lab", "owner": "A", "summary": "输入任务描述，独立展示消息编排、工具分支和最终 Agent 轨迹。",
        "input": "任务描述", "output": "messages、trace、final_answer", "port": 8101,
        "input_from": "用户；集成时同时接收 B5 MemoryResult、B4 AIMessage 和 B3 ToolMessage",
        "output_interface": "POST /api/modules/b1/run",
        "receiver": "前端展示 final_answer；B5 接收 messages、trace、final_answer 用于记忆保存",
        "integration": "B1 是总编排层：维护 system → user → assistant → tool → assistant 消息闭环。",
    },
    "b2": {
        "title": "B2 Skill Sandbox", "owner": "A", "summary": "输入 Skill 与 JSON 参数，独立展示 SkillResult、结构化错误和耗时。",
        "input": "skill_name + JSON 参数", "output": "SkillResult", "port": 8102,
        "input_from": "B3 下发通过 Schema 校验的工具名称与 args；独立演示时由页面输入",
        "output_interface": "POST /api/modules/b2/run",
        "receiver": "B3 接收 SkillResult，并封装为带 tool_call_id 的 ToolMessage",
        "integration": "B2 只实现具体能力与安全限制，不参与模型决策和 Agent 循环。",
    },
    "b3": {
        "title": "B3 Tool Contract Studio", "owner": "B", "summary": "独立生成 Tools Schema，或校验并执行 ToolCall。",
        "input": "toolset + ToolCall JSON", "output": "Tools Schema / ToolMessage", "port": 8103,
        "input_from": "B4 生成的 AIMessage.tool_calls；独立演示时由页面输入 ToolCall",
        "output_interface": "POST /api/modules/b3/run",
        "receiver": "Tools Schema 传给 B4；ToolMessage 传给 B1；内部调用 B2 获取 SkillResult",
        "integration": "B3 是 B4 的模型协议与 B2 的执行函数之间的适配层。",
    },
    "b4": {
        "title": "B4 Local LLM Console", "owner": "C", "summary": "先验证 generate_ai_message 的单轮 AI 消息生成与多工具调用，再展示多轮策略、模型切换、工具调用模式对比、批量评估与计划状态管理。",
        "input": "messages + tools_schema + mode + profile", "output": "AIMessage / trace / comparison / evaluation", "port": 8104,
        "input_from": "B1 传入 messages，B3 传入 Tools Schema",
        "output_interface": "POST /api/modules/b4/run",
        "receiver": "B1 接收 AIMessage；若包含 tool_calls，B1 将调用请求转交 B3",
        "integration": "B4 先用 generate_ai_message 产出标准 AIMessage，再扩展到多轮规划、模型路由、工具调用模式对比、批量评估与计划状态管理；不直接读取文件或执行 Skill。",
    },
    "b5": {
        "title": "B5 Memory Explorer", "owner": "D", "summary": "输入检索问题和模式，独立展示记忆命中、得分和截断信息。",
        "input": "query + search_mode + top_k", "output": "MemoryResult", "port": 8105,
        "input_from": "B1 的用户问题；保存阶段接收 B1 的 messages、trace 和 final_answer",
        "output_interface": "POST /api/modules/b5/run",
        "receiver": "B1 接收 MemoryResult，并把命中记忆拼入模型上下文",
        "integration": "B5 在推理前提供历史上下文，在任务完成后形成记忆闭环。",
    },
}

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def _read_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    payload = json.loads((handler.rfile.read(length) if length else b"{}").decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload

def _reply(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))

def _presence_payload(module: str | None = None) -> dict:
    selected = module or "all"
    meta = MODULE_META.get(module) if module else None
    return {
        "status": "ok",
        "service": "module_demos",
        "module": selected,
        "title": meta["title"] if isinstance(meta, dict) else "B1-B5 Agent Module Studio",
        "timestamp": datetime.now().isoformat(),
    }

def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"未找到可用文件: {joined}")

def _find_latest_output_dir(root: Path, required_files: list[str]) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and all((path / name).exists() for name in required_files)
    ]
    return max(candidates, key=lambda item: item.name) if candidates else None

def _resolve_b4_model_config() -> Path:
    return _first_existing_path(
        PROJECT_ROOT / "configs" / "model_new.yaml",
        PROJECT_ROOT / "configs" / "model.yaml",
    )

def _resolve_b4_tools_schema(body: dict) -> list[dict]:
    schema = body.get("tools_schema")
    if schema is None:
        schema = _read_json(PROJECT_ROOT / "data" / "messages" / "tools_schema_basic.json")
    if not isinstance(schema, list):
        raise ValueError("tools_schema must be a JSON array")
    return schema

def _b4_single_turn_multi_tool_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a local tool-using agent. If tools are needed, you may request one or multiple tool calls "
                "in the same response when they are independent."
            ),
        },
        {
            "role": "user",
            "content": "请同时阅读 docs/agent_intro.txt 和 docs/tool_calling.md，比较这两个文档的关注点差异。",
        },
    ]

def _b4_multi_tool_roundtrip_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a local tool-using agent. If tools are needed, you may request one or multiple tool calls "
                "in the same response when they are independent. After receiving ToolMessage results, decide whether "
                "to request more tools or provide the final answer."
            ),
        },
        {
            "role": "user",
            "content": "请同时阅读 docs/agent_intro.txt 和 docs/tool_calling.md，比较这两个文档的关注点差异。",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_001",
                    "name": "file_reader",
                    "args": {
                        "path": "docs/agent_intro.txt",
                        "max_chars": 2000,
                    },
                },
                {
                    "id": "call_002",
                    "name": "file_reader",
                    "args": {
                        "path": "docs/tool_calling.md",
                        "max_chars": 2000,
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "name": "file_reader",
            "content": (
                "{\"skill_name\":\"file_reader\",\"status\":\"success\",\"input\":{\"path\":\"docs/agent_intro.txt\","
                "\"max_chars\":2000},\"output\":{\"content\":\"文档A聚焦 Agent 的组成结构。\",\"num_chars\":18,"
                "\"source\":\"docs/agent_intro.txt\",\"truncated\":false},\"error\":null,\"latency_ms\":1.0}"
            ),
            "status": "success",
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "name": "file_reader",
            "content": (
                "{\"skill_name\":\"file_reader\",\"status\":\"success\",\"input\":{\"path\":\"docs/tool_calling.md\","
                "\"max_chars\":2000},\"output\":{\"content\":\"文档B聚焦 Tool Calling 的闭环机制。\",\"num_chars\":27,"
                "\"source\":\"docs/tool_calling.md\",\"truncated\":false},\"error\":null,\"latency_ms\":1.0}"
            ),
            "status": "success",
        },
    ]

def _b4_plan_execute_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a local tool-using agent.",
        },
        {
            "role": "user",
            "content": (
                "请以 Plan-and-Execute 的方式完成一个稳定的三步任务："
                "1) 读取 docs/agent_intro.txt，总结 Agent 系统的 3 条核心组成；"
                "2) 读取 docs/tool_calling.md，总结 Tool Calling 闭环的 3 条关键机制，并指出其中哪 1 条与第1步最相关；"
                "3) 读取 tables/results.csv，说明这个表里最值得关注的 2 个字段，并判断它更适合用于展示“结果统计”还是“过程追踪”。"
                "最后给出一段整合结论。"
            ),
        },
    ]

def _b4_react_one_round_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a local tool-using agent.",
        },
        {
            "role": "user",
            "content": "请读取 docs/agent_intro.txt，并用中文总结 3 条要点。尽量用最少轮次完成。",
        },
    ]

def _b4_adaptive_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a local tool-using agent.",
        },
        {
            "role": "user",
            "content": (
                "请判断这个任务更适合 ReAct 还是 Plan-and-Execute，并完成它："
                "先比较 docs/agent_intro.txt 与 docs/tool_calling.md 的关注点差异，"
                "再看 tables/results.csv 是否能补充一个“评测结果统计”的例子，最后给出你的结论。"
            ),
        },
    ]

def _b4_model_routing_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a local tool-using agent.",
        },
        {
            "role": "user",
            "content": (
                "请比较 docs/agent_intro.txt 与 docs/tool_calling.md 的关注点差异，"
                "并说明这个任务在 direct / plan / execute 三个阶段里更适合怎样的模型配置。"
            ),
        },
    ]

def _b4_execution_engine_messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a local tool-using agent.",
        },
        {
            "role": "user",
            "content": (
                "请演示计划状态管理与证据传播："
                "1) 读取 docs/agent_intro.txt，总结 Agent 系统的 3 个核心组成；"
                "2) 再读取 docs/tool_calling.md，总结 Tool Calling 闭环的 3 个关键机制，并指出其中哪 1 条与第1步最相关；"
                "3) 最后读取 tables/results.csv，指出最值得关注的 2 个字段，并说明它更适合用于展示“结果统计”还是“过程追踪”。"
                "要求按步骤推进；每一步只基于当前真实工具证据完成，若证据有限请直接说明限制，不要额外搜索无关文件。"
            ),
        },
    ]

def _default_b4_messages(action: str) -> list[dict]:
    if action == "plan_execute":
        return _b4_plan_execute_messages()
    if action == "react_one_round_execute":
        return _b4_react_one_round_messages()
    if action == "adaptive_execute":
        return _b4_adaptive_messages()
    if action == "execution_engine":
        return _b4_execution_engine_messages()
    if action == "model_routing":
        return _b4_model_routing_messages()
    if action in {"multi_tool_roundtrip", "first_module_roundtrip"}:
        return _b4_multi_tool_roundtrip_messages()
    if action in {"generation", "single_turn_generation", "first_module_generate"}:
        return _b4_single_turn_multi_tool_messages()
    return _b4_single_turn_multi_tool_messages()

def _resolve_b4_messages(body: dict, action: str) -> list[dict]:
    messages = body.get("messages")
    if messages is None:
        messages = _default_b4_messages(action)
    if not isinstance(messages, list):
        raise ValueError("messages must be a JSON array")
    return messages

def _profile_label(profile: str) -> str:
    labels = {
        "qwen_4b": "Qwen3.5-4B",
        "qwen_7b": "Qwen2.5-7B",
        "default": "Default",
        "planner": "Planner",
    }
    return labels.get(profile, profile)

def _flatten_eval_summary(summary: dict, scope_label: str) -> dict:
    flattened: dict[str, dict] = {}
    for key, item in summary.items():
        if "::" in key:
            profile_name, mode_name = key.split("::", 1)
        else:
            profile_name, mode_name = key, "unknown"
        label = f"{scope_label} · {_profile_label(profile_name)} · {mode_name}"
        flattened[label] = dict(item)
        flattened[label]["profile"] = profile_name
        flattened[label]["mode"] = mode_name
        flattened[label]["scope"] = scope_label
    return flattened

def _load_b4_mock_plan_fixture() -> tuple[dict, str]:
    fixture_dir = _find_latest_output_dir(OUTPUT_ROOT / "b4", ["trace.json", "plan_preview.md", "messages.json"])
    if fixture_dir is None:
        raise FileNotFoundError("module_demos/outputs/b4 中未找到可用的 Plan-and-Execute 产物")
    trace = _read_json(fixture_dir / "trace.json")
    payload = {
        "status": trace.get("status"),
        "final_answer": (fixture_dir / "final_answer.md").read_text(encoding="utf-8") if (fixture_dir / "final_answer.md").exists() else "",
        "messages": _read_json(fixture_dir / "messages.json"),
        "trace": trace,
        "plan_preview": (fixture_dir / "plan_preview.md").read_text(encoding="utf-8"),
        "source": "module_demo_artifact",
    }
    return payload, str(fixture_dir)

def _load_latest_b4_comparison() -> tuple[dict, str] | None:
    fixture_dir = _find_latest_output_dir(OUTPUT_ROOT / "b4", ["comparison.json"])
    if fixture_dir is None:
        return None
    return {"comparison": _read_json(fixture_dir / "comparison.json"), "source": "module_demo_artifact"}, str(fixture_dir)

def _load_latest_b4_eval_summary() -> tuple[dict, str] | None:
    fixture_dir = _find_latest_output_dir(OUTPUT_ROOT / "b4", ["eval_summary.json"])
    if fixture_dir is None:
        return None
    return {"evaluation": _read_json(fixture_dir / "eval_summary.json"), "source": "module_demo_artifact"}, str(fixture_dir)

def _build_mock_b4_comparison(messages: list[dict]) -> dict:
    input_hint = sum(len(str(item.get("content") or "")) for item in messages)
    return {
        "prompt_json": {
            "status": "success",
            "tool_call_count": 2,
            "tool_name_correct": True,
            "structured_output_success": True,
            "tool_match": True,
            "args_complete": True,
            "input_tokens": max(180, round(input_hint / 2.2) + 140),
            "output_tokens": 112,
            "elapsed_seconds": 0.0,
        },
        "native_tools": {
            "status": "success",
            "tool_call_count": 2,
            "tool_name_correct": True,
            "structured_output_success": True,
            "tool_match": True,
            "args_complete": True,
            "input_tokens": max(120, round(input_hint / 3.1) + 96),
            "output_tokens": 98,
            "elapsed_seconds": 0.0,
        },
        "adaptive": {
            "status": "success",
            "tool_call_count": 2,
            "tool_name_correct": True,
            "structured_output_success": True,
            "args_complete": True,
            "input_tokens": max(150, round(input_hint / 2.8) + 108),
            "output_tokens": 86,
            "elapsed_seconds": 0.0,
            "selected_mode": "native_tools",
            "strategy": "react_one_round",
            "complexity": "low",
            "confidence": 0.86,
            "llm_call_count": 2,
            "tool_rounds_used": 1,
        },
    }

def _build_mock_b4_eval_summary() -> dict:
    combined_summary = {
        "单轮 · Qwen3.5-4B · prompt_json": {"cases": 6, "success_rate": 0.8333, "structured_output_rate": 0.8333, "tool_match_rate": 0.8333, "args_complete_rate": 0.8333, "avg_input_tokens": 648.5, "avg_output_tokens": 121.3, "avg_elapsed_seconds": 0.0},
        "单轮 · Qwen3.5-4B · native_tools": {"cases": 6, "success_rate": 0.8333, "structured_output_rate": 1.0, "tool_match_rate": 0.8333, "args_complete_rate": 0.8333, "avg_input_tokens": 522.1, "avg_output_tokens": 109.4, "avg_elapsed_seconds": 0.0},
        "单轮 · Qwen2.5-7B · prompt_json": {"cases": 6, "success_rate": 1.0, "structured_output_rate": 1.0, "tool_match_rate": 1.0, "args_complete_rate": 1.0, "avg_input_tokens": 651.2, "avg_output_tokens": 126.8, "avg_elapsed_seconds": 0.0},
        "单轮 · Qwen2.5-7B · native_tools": {"cases": 6, "success_rate": 1.0, "structured_output_rate": 1.0, "tool_match_rate": 1.0, "args_complete_rate": 1.0, "avg_input_tokens": 529.7, "avg_output_tokens": 114.2, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen3.5-4B · prompt_json": {"cases": 6, "success_rate": 0.5, "structured_output_rate": 0.6667, "tool_match_rate": 0.6667, "args_complete_rate": 0.6667, "avg_input_tokens": 1512.4, "avg_output_tokens": 286.2, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen3.5-4B · native_tools": {"cases": 6, "success_rate": 0.5, "structured_output_rate": 0.6667, "tool_match_rate": 0.6667, "args_complete_rate": 0.6667, "avg_input_tokens": 1398.6, "avg_output_tokens": 274.7, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen3.5-4B · adaptive": {"cases": 6, "success_rate": 0.6667, "structured_output_rate": 0.8333, "tool_match_rate": 0.8333, "args_complete_rate": 0.8333, "avg_input_tokens": 1186.5, "avg_output_tokens": 219.8, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen2.5-7B · prompt_json": {"cases": 6, "success_rate": 0.8333, "structured_output_rate": 0.8333, "tool_match_rate": 0.8333, "args_complete_rate": 0.8333, "avg_input_tokens": 1568.9, "avg_output_tokens": 301.5, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen2.5-7B · native_tools": {"cases": 6, "success_rate": 0.8333, "structured_output_rate": 0.8333, "tool_match_rate": 0.8333, "args_complete_rate": 0.8333, "avg_input_tokens": 1452.7, "avg_output_tokens": 286.9, "avg_elapsed_seconds": 0.0},
        "多轮/自适应 · Qwen2.5-7B · adaptive": {"cases": 6, "success_rate": 1.0, "structured_output_rate": 1.0, "tool_match_rate": 1.0, "args_complete_rate": 1.0, "avg_input_tokens": 1234.8, "avg_output_tokens": 232.4, "avg_elapsed_seconds": 0.0},
    }
    return {
        "profiles": ["qwen_4b", "qwen_7b"],
        "modes": ["prompt_json", "native_tools", "adaptive"],
        "combined_summary": combined_summary,
        "source": "synthetic_mock",
    }

def _b1_fixture_payloads(user_inputs: list[str]) -> tuple[list[dict], dict]:
    ai_messages = []
    tool_messages = {}
    for index, user_input in enumerate(user_inputs, 1):
        call_id = f"demo_b1_call_{index:03d}"
        lowered = user_input.casefold()
        if "工具" in user_input or "tool" in lowered:
            source = "docs/tool_calling.md"
            source_content = "模型生成 tool_calls，运行时调用工具，再将 ToolMessage 回传模型形成闭环。"
            answer = "工具调用循环分为三步：模型生成 tool_calls；运行时执行并生成 ToolMessage；模型结合工具结果继续调用或输出最终答案。"
        elif "记忆" in user_input or "memory" in lowered:
            source = "docs/agent_intro.txt"
            source_content = "Memory 在推理前提供历史上下文，在任务结束后保存新的对话结果。"
            answer = "记忆模块负责检索相关历史、控制注入上下文长度，并在任务完成后保存 messages、trace 与最终回答，使后续任务能够复用经验。"
        else:
            source = "docs/agent_intro.txt"
            answer = "Agent 的核心组成包括模型、工具、记忆和执行循环：模型负责决策，工具负责执行，记忆提供上下文，运行时维护消息闭环。"
        source_path = PROJECT_ROOT / "data" / source
        if source_path.exists():
            source_content = source_path.read_text(encoding="utf-8").strip()[:1200]
        ai_messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "name": "file_reader", "args": {"path": source, "max_chars": 1200}}]})
        ai_messages.append({"role": "assistant", "content": f"第 {index} 轮回答：{answer}", "tool_calls": []})
        skill_result = {"skill_name": "file_reader", "status": "success", "input": {"path": source, "max_chars": 1200}, "output": {"content": source_content, "num_chars": len(source_content), "source": source, "truncated": False}, "error": None, "latency_ms": 0.0}
        tool_messages[call_id] = {"role": "tool", "tool_call_id": call_id, "name": "file_reader", "content": json.dumps(skill_result, ensure_ascii=False), "status": "success"}
    return ai_messages, tool_messages

def _write_b1_fixture_files(outdir: Path, user_inputs: list[str]) -> dict:
    ai_messages, tool_messages = _b1_fixture_payloads(user_inputs)
    ai_path = outdir / "fixture_ai_messages.json"
    tool_path = outdir / "fixture_tool_messages.json"
    _write_json(ai_path, ai_messages)
    _write_json(tool_path, tool_messages)
    return {
        "selected_memory_path": str((PROJECT_ROOT / "data" / "b1_fixtures" / "preset_memory.json").resolve()),
        "tools_schema_path": str((PROJECT_ROOT / "data" / "messages" / "tools_schema_basic.json").resolve()),
        "ai_messages_path": str(ai_path.resolve()),
        "tool_messages_path": str(tool_path.resolve()),
    }

def _run_b1(body: dict, outdir: Path) -> dict:
    from b1_agent_runtime import run_agent, run_batch_tasks
    demo_mode = str(body.get("demo_mode") or "single")
    task = str(body.get("task") or "读取 docs/agent_intro.txt，并总结三条中文要点。").strip()
    default_inputs = [task, "继续说明工具调用循环。", "最后总结记忆模块的作用。"]
    user_inputs = body.get("user_inputs") or default_inputs
    if not isinstance(user_inputs, list) or not all(isinstance(item, str) and item.strip() for item in user_inputs): raise ValueError("user_inputs must be a non-empty string array")
    if demo_mode == "batch":
        batch_payload = {"tasks": []}
        for index, batch_task in enumerate((body.get("batch_tasks") or ["读取 Agent 简介并总结。", "说明 Tool Calling 消息闭环。"]), 1):
            fixture_dir = outdir / f"batch_fixture_{index:02d}"
            fixture_dir.mkdir(parents=True, exist_ok=True)
            fixtures = _write_b1_fixture_files(fixture_dir, [str(batch_task)])
            batch_payload["tasks"].append({"task_id": f"web_batch_{index:02d}", "runtime_input": {"conversation_id": f"demo_batch_{_now_tag()}_{index}", "execution_mode": "fixture", "user_input": str(batch_task), "system_prompt_path": str((PROJECT_ROOT / "prompts" / "local_tool_agent.txt").resolve()), "toolset": "basic_tools", "max_turns": 3, "save_memory": "none", "fixtures": fixtures}})
        batch_path = outdir / "batch_input.json"
        _write_json(batch_path, batch_payload)
        batch_result = run_batch_tasks(str(batch_path), None, None, None, str(outdir / "batch_runs"))
        params = {"demo_mode": demo_mode, "batch_total": batch_result["total_tasks"], "batch_success": batch_result["success_count"], "batch_errors": batch_result["error_count"], "elapsed_ms": batch_result["elapsed_ms"]}
        return {"module": "B1", "input": {"demo_mode": demo_mode, "batch_tasks": body.get("batch_tasks") or ["读取 Agent 简介并总结。", "说明 Tool Calling 消息闭环。"]}, "output": {"batch_result": batch_result, "result_parameters": params}, "artifacts": {"run_dir": str(outdir)}}

    selected_inputs = [task] if demo_mode in {"single", "resume", "multi_tool_loop"} else user_inputs
    fixtures = _write_b1_fixture_files(outdir, selected_inputs)
    if demo_mode == "multi_tool_loop":
        ai_messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "loop_read_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 800}},
                {"id": "loop_read_002", "name": "file_reader", "args": {"path": "docs/tool_calling.md", "max_chars": 800}},
            ]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "loop_calc_003", "name": "calculator", "args": {"expression": "23 * 17 + 9"}}]},
            {"role": "assistant", "content": "已完成两轮工具循环：第一轮并行读取两个文件，第二轮执行计算，最终结果为 400。", "tool_calls": []},
        ]
        tool_messages = {
            "loop_read_001": {"role": "tool", "tool_call_id": "loop_read_001", "name": "file_reader", "content": json.dumps({"skill_name": "file_reader", "status": "success", "input": {}, "output": {"content": "Agent 由模型、工具、记忆和循环组成。"}, "error": None, "latency_ms": 0.0}, ensure_ascii=False), "status": "success"},
            "loop_read_002": {"role": "tool", "tool_call_id": "loop_read_002", "name": "file_reader", "content": json.dumps({"skill_name": "file_reader", "status": "success", "input": {}, "output": {"content": "Tool Calling 形成 assistant-tool-assistant 闭环。"}, "error": None, "latency_ms": 0.0}, ensure_ascii=False), "status": "success"},
            "loop_calc_003": {"role": "tool", "tool_call_id": "loop_calc_003", "name": "calculator", "content": json.dumps({"skill_name": "calculator", "status": "success", "input": {"expression": "23 * 17 + 9"}, "output": {"result": 400}, "error": None, "latency_ms": 0.0}, ensure_ascii=False), "status": "success"},
        }
        _write_json(Path(fixtures["ai_messages_path"]), ai_messages)
        _write_json(Path(fixtures["tool_messages_path"]), tool_messages)
    runtime = {
        "execution_mode": "fixture", "conversation_id": f"demo_b1_{_now_tag()}",
        "system_prompt_path": str((PROJECT_ROOT / "prompts" / "local_tool_agent.txt").resolve()),
        "toolset": "basic_tools", "max_turns": 3, "save_memory": "none",
        "fixtures": fixtures,
    }
    if len(selected_inputs) == 1: runtime["user_input"] = selected_inputs[0]
    else: runtime["user_inputs"] = selected_inputs
    if demo_mode == "history_compression": runtime["history_compression"] = {"enabled": True, "max_messages": 6, "keep_recent_messages": 2, "summary_max_chars": 1200}
    if demo_mode == "prompt_switch": runtime["system_prompt_events"] = [
        {"user_turn_index": 1, "mode": "add", "label": "brief_answer_style", "system_prompt_path": str((PROJECT_ROOT / "prompts" / "brief_answer_prompt.txt").resolve())},
        {"user_turn_index": 2, "mode": "switch", "label": "strict_tool_agent", "system_prompt_path": str((PROJECT_ROOT / "prompts" / "strict_tool_prompt.txt").resolve())},
    ]
    input_path = outdir / "runtime_input.json"
    _write_json(input_path, runtime)
    result = run_agent(str(input_path), None, None, None, str(outdir))
    resumed = False
    if demo_mode == "resume":
        result = run_agent(str(input_path), None, None, None, str(outdir), resume=True)
        resumed = True
    messages = json.loads(Path(result["messages_path"]).read_text(encoding="utf-8"))
    trace = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
    params = {
        "demo_mode": demo_mode, "user_turn_count": trace.get("user_turn_count", len(selected_inputs)),
        "tool_rounds_used": trace.get("tool_rounds_used", 0), "llm_call_count": trace.get("llm_call_count", 0),
        "checkpoint_exists": (outdir / "checkpoint.json").exists(), "resumed_from_checkpoint": resumed,
        "history_compression_count": trace.get("history_compression_count", 0),
        "system_prompt_event_count": len(trace.get("system_prompt_events_applied", [])),
        "active_system_prompt": (trace.get("active_system_prompt") or {}).get("label", "initial"),
    }
    return {"module": "B1", "input": {"task": task, "demo_mode": demo_mode, "user_inputs": selected_inputs, "execution_mode": "fixture"}, "output": {"final_answer": result["final_answer"], "messages": messages, "trace": trace, "result_parameters": params}, "artifacts": {"run_dir": str(outdir)}}

def _run_b2(body: dict, outdir: Path) -> dict:
    from b2_run_skill import run_skill
    skill = str(body.get("skill") or "calculator")
    args = body.get("args", {"expression": "23 * 17 + 9"})
    if not isinstance(args, dict): raise ValueError("args must be a JSON object")
    result = run_skill(skill, args, str(PROJECT_ROOT / "data"), str(outdir))
    _write_json(outdir / f"{skill}_result.json", result)
    return {"module": "B2", "input": {"skill": skill, "args": args}, "output": result, "artifacts": {"run_dir": str(outdir)}}

def _run_b3(body: dict, outdir: Path) -> dict:
    from b3_tool_layer import evaluate_tool_call_accuracy, execute_tool_calls, get_tools_schema
    action, toolset = str(body.get("action") or "execute"), str(body.get("toolset") or "basic_tools")
    retry_limit = int(body.get("retry_limit", 1))
    cache_enabled = bool(body.get("cache_enabled", True))
    config = str(PROJECT_ROOT / "configs" / "tools.yaml")
    schema = get_tools_schema(config, toolset, str(outdir), bool(body.get("auto_from_code", False)))
    output = {"tools_schema": schema}
    stats = {"total_tool_calls": 0, "total_errors": 0, "failure_rate": 0.0, "by_tool": {}}
    cache_hit = False
    if action == "execute":
        calls = body.get("tool_calls") or [{"id": "demo_call_001", "name": "calculator", "args": {"expression": "8 * 12 + 4"}}]
        if not isinstance(calls, list): raise ValueError("tool_calls must be a JSON array")
        output["tool_messages"] = execute_tool_calls(calls, config, toolset, str(outdir), retry_limit, cache_enabled)
        stats_path = outdir / "tool_call_stats.json"
        if stats_path.exists(): stats = json.loads(stats_path.read_text(encoding="utf-8"))
        log_path = outdir / "tool_call_log.jsonl"
        if log_path.exists():
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            cache_hit = any(bool(record.get("cache_hit")) for record in records)
    accuracy = evaluate_tool_call_accuracy(str(PROJECT_ROOT / "data" / "messages" / "b3_schema_description_eval_cases.json"), str(outdir))
    total_calls = int(stats.get("total_tool_calls") or 0)
    total_retries = sum(int(item.get("retries") or 0) for item in stats.get("by_tool", {}).values())
    weighted_latency = sum(float(item.get("avg_latency_ms") or 0) * int(item.get("tool_calls") or 0) for item in stats.get("by_tool", {}).values())
    result_parameters = {
        "cache_hit": cache_hit,
        "cache_enabled": cache_enabled,
        "total_tool_calls": total_calls,
        "failure_rate": float(stats.get("failure_rate") or 0),
        "retry_count": total_retries,
        "avg_latency_ms": round(weighted_latency / total_calls, 3) if total_calls else 0.0,
        "schema_accuracy": {name: item.get("accuracy", 0.0) for name, item in accuracy.get("variants", {}).items()},
    }
    output["result_parameters"] = result_parameters
    return {"module": "B3", "input": {"action": action, "toolset": toolset, "retry_limit": retry_limit, "cache_enabled": cache_enabled}, "output": output, "artifacts": {"run_dir": str(outdir)}}

def _run_b4(body: dict, outdir: Path) -> dict:
    from b4_local_agent_llm import (
        _classify_task_profile,
        _detect_routing_phase,
        _load_model_config,
        _merged_model_settings,
        _resolve_model_profile,
        classify_task_complexity,
        compare_tools_injection_modes,
        generate_ai_message,
        run_adaptive_execute,
        run_batch_evaluation,
        run_batch_plan_execute_evaluation,
        run_plan_execute,
        run_react_one_round_execute,
    )
    action_aliases = {
        "react_one_round": "react_one_round_execute",
        "react": "react_one_round_execute",
        "adaptive": "adaptive_execute",
        "single_turn_generation": "generation",
        "first_module_generate": "generation",
        "first_module_roundtrip": "multi_tool_roundtrip",
    }
    action = action_aliases.get(str(body.get("action") or "generation"), str(body.get("action") or "generation"))
    mode = str(body.get("mode") or "mock")
    profile = str(body.get("profile") or "qwen_4b")
    messages = _resolve_b4_messages(body, action)
    schema = _resolve_b4_tools_schema(body)
    if not isinstance(messages, list) or not isinstance(schema, list): raise ValueError("messages and tools_schema must be JSON arrays")
    model_config = _resolve_b4_model_config()
    tools_config = PROJECT_ROOT / "configs" / "tools.yaml"
    input_payload: dict = {"action": action, "mode": mode, "profile": profile, "messages": messages, "tools_count": len(schema), "model_config": str(model_config)}
    if action == "multi_tool_roundtrip":
        execution_mode = "mock" if mode == "mock" else mode
        first = generate_ai_message(str(model_config), messages, schema, execution_mode, str(outdir), "multi_request", profile)
        received_messages = _b4_multi_tool_roundtrip_messages()
        second = generate_ai_message(str(model_config), received_messages, schema, execution_mode, str(outdir), "multi_response", profile)
        output = {
            "first_ai_message": first["ai_message"],
            "received_tool_messages": [item for item in received_messages if item.get("role") == "tool"],
            "second_ai_message": second["ai_message"],
            "capability_checklist": {
                "core_function": "generate_ai_message",
                "supported_modes": ["mock", "prompt_json", "native_tools"],
                "supports_parallel_tool_calls": True,
                "supports_multiple_tool_messages": True,
                "supports_retry": True,
                "supports_compress_tool_messages": True,
            },
        }
        params = {
            "action": action,
            "demo_module": "第一模块",
            "core_function": "generate_ai_message",
            "generated_tool_calls": len(first["ai_message"].get("tool_calls", [])),
            "received_tool_messages": len(output["received_tool_messages"]),
            "followup_tool_calls": len(second["ai_message"].get("tool_calls", [])),
            "repair_attempted": bool(first.get("raw_record", {}).get("repair_attempted") or second.get("raw_record", {}).get("repair_attempted")),
            "attempt_count": len(first.get("raw_record", {}).get("attempts") or []) + len(second.get("raw_record", {}).get("attempts") or []),
            "selected_profile": profile,
            "mode": execution_mode,
            "compression_feature": "compress_tool_messages",
        }
        input_payload["messages"] = {
            "first_round_request_messages": messages,
            "second_round_actual_messages": received_messages,
        }
        input_payload["message_display_mode"] = "actual_round_messages"
        input_payload["demo_module"] = "第一模块：单轮 AI 消息生成与多工具调用"
        input_payload["message_note"] = "未显式传入 messages 时，默认使用内置双文件示例，先验证首轮并行 tool_calls，再验证接收多个 ToolMessage 后的续推能力。"
    elif action in {"plan_execute", "adaptive_execute", "react_one_round_execute"}:
        if mode == "mock":
            output, artifact_source = _load_b4_mock_plan_fixture()
            trace = output.get("trace", {})
            plan_preview = str(output.get("plan_preview") or "")
            data_source = f"module_demos 已保存产物: {artifact_source}"
            verified_plan_steps = len(re.findall(r"^\d+\.", plan_preview, flags=re.MULTILINE))
            strategy_used = "plan_execute" if action == "plan_execute" else action
            routing = {"strategy": strategy_used, "complexity": "unknown", "confidence": 0.0, "reason": "mock 模式直接回放已保存运行产物。"}
        else:
            routing = classify_task_complexity(str(model_config), messages, str(outdir / "adaptive_router"), forced_profile=profile)
            if action == "adaptive_execute":
                output = run_adaptive_execute(str(model_config), messages, schema, str(tools_config), "basic_tools", str(outdir), 3, int(body.get("max_plan_steps") or 4), "lite", profile, "native_tools", "prompt_json")
                strategy_used = str(routing.get("strategy") or "adaptive_execute")
            elif action == "react_one_round_execute":
                output = run_react_one_round_execute(str(model_config), messages, schema, str(tools_config), "basic_tools", mode, str(outdir), 1, profile)
                strategy_used = "react_one_round_execute"
            else:
                output = run_plan_execute(str(model_config), messages, schema, str(tools_config), "basic_tools", mode, str(outdir), 3, int(body.get("max_plan_steps") or 4), "lite", profile)
                strategy_used = "plan_execute"
            trace = output.get("trace", {})
            data_source = f"本次实时执行 · 路由建议 {routing.get('strategy')} ({routing.get('complexity')}, conf {float(routing.get('confidence') or 0.0):.2f})"
            verified_plan_steps = len((trace.get("plan_state") or {}).get("steps", []))
        output["adaptive_routing"] = routing
        plan_state = trace.get("plan_state", {})
        params = {"action": "plan_execute", "requested_action": action, "execution_strategy": strategy_used, "plan_steps": verified_plan_steps if mode == "mock" else len(plan_state.get("steps", [])), "plan_status": output.get("status"), "llm_call_count": trace.get("llm_call_count", 0), "tool_rounds_used": trace.get("tool_rounds_used", 0), "selected_profile": profile, "mode": mode, "data_source": data_source}
    elif action == "model_routing":
        config_path, loaded_config = _load_model_config(str(model_config))
        routing_config = loaded_config.get("routing") if isinstance(loaded_config.get("routing"), dict) else {}
        model_pool = loaded_config.get("model_pool") if isinstance(loaded_config.get("model_pool"), dict) else {}

        def _probe_resolution(probe_messages: list[dict]) -> dict:
            resolved_profile = _resolve_model_profile(loaded_config, probe_messages) or "default"
            merged = _merged_model_settings(loaded_config, resolved_profile)
            return {
                "phase": _detect_routing_phase(probe_messages),
                "resolved_profile": resolved_profile,
                "resolved_model_target": str(merged.get("model_name_or_path") or ""),
            }

        direct_probe_messages = deepcopy(messages)
        plan_probe_messages = deepcopy(messages) + [{"role": "user", "content": "请先生成一个可执行的计划，最多 4 步。"}]
        execute_probe_messages = deepcopy(messages) + [{"role": "user", "content": "<plan_state>{\"mode\":\"plan_execute\",\"steps\":[{\"step\":1,\"title\":\"读取 docs/agent_intro.txt\",\"status\":\"pending\"}]}</plan_state>\n当前计划状态：请继续完成第 1 步。"}]

        auto_profile = _resolve_model_profile(loaded_config, messages) or "default"
        auto_settings = _merged_model_settings(loaded_config, auto_profile)
        pool_summary = [
            {
                "profile": name,
                "label": _profile_label(name),
                "model_target": str((settings or {}).get("model_name_or_path") or ""),
            }
            for name, settings in model_pool.items()
            if isinstance(settings, dict)
        ]
        live_verification = {
            "requested_profile": profile,
            "selected_profile": "",
            "resolved_model_path": "",
            "mode": mode,
            "status": "not_run" if mode == "mock" else "pending",
        }
        data_source = "读取 model.yaml 配置并解析 routing/model_pool；mock 模式未执行实时验证。"
        if mode != "mock":
            live_result = generate_ai_message(str(config_path), messages, schema, mode, str(outdir), "model_routing", profile)
            raw_record = live_result.get("raw_record") or {}
            live_verification = {
                "requested_profile": profile,
                "selected_profile": raw_record.get("model_profile") or profile,
                "resolved_model_path": raw_record.get("resolved_model_path") or "",
                "mode": raw_record.get("mode") or mode,
                "status": live_result.get("status"),
            }
            data_source = "读取 model.yaml 配置，并额外执行一次 generate_ai_message 验证 forced_profile。"
        output = {
            "routing_configuration": {
                "config_path": str(config_path),
                "routing": routing_config,
                "model_pool": pool_summary,
            },
            "resolution_report": {
                "current_messages": {
                    "phase": _detect_routing_phase(messages),
                    "task_profile": _classify_task_profile(messages),
                    "resolved_profile": auto_profile,
                    "resolved_model_target": str(auto_settings.get("model_name_or_path") or ""),
                },
                "phase_probes": {
                    "direct": _probe_resolution(direct_probe_messages),
                    "plan": _probe_resolution(plan_probe_messages),
                    "execute": _probe_resolution(execute_probe_messages),
                },
                "forced_profile": {
                    "requested_profile": profile,
                    "resolved_model_target": str(_merged_model_settings(loaded_config, profile).get("model_name_or_path") or ""),
                },
            },
            "live_verification": live_verification,
        }
        params = {
            "action": action,
            "current_phase": _detect_routing_phase(messages),
            "task_profile": _classify_task_profile(messages),
            "auto_profile": auto_profile,
            "auto_model_target": str(auto_settings.get("model_name_or_path") or ""),
            "forced_profile": profile,
            "forced_model_target": str(_merged_model_settings(loaded_config, profile).get("model_name_or_path") or ""),
            "default_profile": routing_config.get("default_profile"),
            "plan_profile": routing_config.get("plan_profile"),
            "execute_profile": routing_config.get("execute_profile"),
            "live_selected_profile": live_verification.get("selected_profile") or "mock 未执行",
            "live_model_target": live_verification.get("resolved_model_path") or "",
            "current_model_target": (
                live_verification.get("resolved_model_path")
                or str(_merged_model_settings(loaded_config, profile).get("model_name_or_path") or "")
                or str(auto_settings.get("model_name_or_path") or "")
            ),
            "data_source": data_source,
        }
    elif action == "injection_compare":
        cached = _load_latest_b4_comparison() if mode == "mock" else None
        if cached is not None:
            output, artifact_source = cached
            comparison = output["comparison"]
            data_source = f"module_demos 已保存产物: {artifact_source}"
        elif mode == "mock":
            comparison = _build_mock_b4_comparison(messages)
            _write_json(outdir / "comparison.json", comparison)
            output = {"comparison": comparison, "source": "synthetic_mock"}
            data_source = "mock 合成对比统计"
        else:
            comparison = compare_tools_injection_modes(
                str(model_config),
                messages,
                schema,
                str(outdir),
                str(tools_config),
                "basic_tools",
                forced_profile=profile,
                max_turns=3,
                max_plan_steps=int(body.get("max_plan_steps") or 4),
                evidence_policy="lite",
                low_mode="native_tools",
                high_mode="prompt_json",
            )
            output = {"comparison": comparison, "source": "live_evaluation"}
            data_source = "本次实时对比"
        if not isinstance(comparison.get("adaptive"), dict):
            comparison["adaptive"] = {
                "status": "unavailable",
                "tool_call_count": 0,
                "tool_name_correct": False,
                "args_complete": False,
                "structured_output_success": False,
                "input_tokens": None,
                "output_tokens": None,
                "elapsed_seconds": None,
                "selected_mode": None,
                "strategy": None,
                "complexity": None,
                "confidence": None,
                "llm_call_count": None,
                "tool_rounds_used": None,
            }
        output["comparison"] = comparison
        prompt_metrics = comparison.get("prompt_json", {}) if isinstance(comparison.get("prompt_json"), dict) else {}
        native_metrics = comparison.get("native_tools", {}) if isinstance(comparison.get("native_tools"), dict) else {}
        adaptive_metrics = comparison.get("adaptive", {}) if isinstance(comparison.get("adaptive"), dict) else {}
        params = {
            "action": action,
            "prompt_json_success": prompt_metrics.get("structured_output_success"),
            "prompt_json_tool_name_correct": prompt_metrics.get("tool_name_correct"),
            "prompt_json_args_complete": prompt_metrics.get("args_complete"),
            "prompt_json_tool_call_count": prompt_metrics.get("tool_call_count"),
            "prompt_json_input_tokens": prompt_metrics.get("input_tokens"),
            "prompt_json_output_tokens": prompt_metrics.get("output_tokens"),
            "prompt_json_elapsed_seconds": prompt_metrics.get("elapsed_seconds"),
            "native_tools_success": native_metrics.get("structured_output_success"),
            "native_tools_tool_name_correct": native_metrics.get("tool_name_correct"),
            "native_tools_args_complete": native_metrics.get("args_complete"),
            "native_tools_tool_call_count": native_metrics.get("tool_call_count"),
            "native_tools_input_tokens": native_metrics.get("input_tokens"),
            "native_tools_output_tokens": native_metrics.get("output_tokens"),
            "native_tools_elapsed_seconds": native_metrics.get("elapsed_seconds"),
            "adaptive_success": adaptive_metrics.get("structured_output_success"),
            "adaptive_tool_name_correct": adaptive_metrics.get("tool_name_correct"),
            "adaptive_args_complete": adaptive_metrics.get("args_complete"),
            "adaptive_tool_call_count": adaptive_metrics.get("tool_call_count"),
            "adaptive_input_tokens": adaptive_metrics.get("input_tokens"),
            "adaptive_output_tokens": adaptive_metrics.get("output_tokens"),
            "adaptive_elapsed_seconds": adaptive_metrics.get("elapsed_seconds"),
            "adaptive_selected_mode": adaptive_metrics.get("selected_mode"),
            "adaptive_strategy": adaptive_metrics.get("strategy"),
            "adaptive_complexity": adaptive_metrics.get("complexity"),
            "adaptive_confidence": adaptive_metrics.get("confidence"),
            "adaptive_llm_call_count": adaptive_metrics.get("llm_call_count"),
            "adaptive_tool_rounds_used": adaptive_metrics.get("tool_rounds_used"),
            "data_source": data_source,
        }
    elif action == "batch_eval":
        selected_eval_mode = str(body.get("mode") or "prompt_json").strip() or "prompt_json"
        cached = _load_latest_b4_eval_summary() if selected_eval_mode == "mock" else None
        if cached is not None:
            output, artifact_source = cached
            evaluation = output["evaluation"]
            combined_summary = evaluation.get("combined_summary", {})
            data_source = f"module_demos 已保存产物: {artifact_source}"
        elif selected_eval_mode == "mock":
            evaluation = _build_mock_b4_eval_summary()
            combined_summary = evaluation["combined_summary"]
            _write_json(outdir / "eval_summary.json", evaluation)
            output = {"evaluation": evaluation, "source": "synthetic_mock"}
            data_source = "mock 合成批量评测（含自适应）"
        else:
            if selected_eval_mode not in {"prompt_json", "native_tools", "adaptive"}:
                raise ValueError("batch_eval mode must be one of: prompt_json, native_tools, adaptive, mock")
            profiles = body.get("profiles")
            if profiles is not None and (not isinstance(profiles, list) or not all(isinstance(item, str) and item.strip() for item in profiles)):
                raise ValueError("profiles must be a string array when provided")
            single_eval = {"summary": {}}
            if selected_eval_mode in {"prompt_json", "native_tools"}:
                single_eval = run_batch_evaluation(
                    str(model_config),
                    schema,
                    str(PROJECT_ROOT / "data" / "messages" / "eval_cases_feature5.json"),
                    str(outdir / "single_turn"),
                    modes=[selected_eval_mode],
                    profiles=profiles,
                )
            plan_eval = run_batch_plan_execute_evaluation(
                str(model_config),
                schema,
                str(PROJECT_ROOT / "data" / "messages" / "eval_cases_feature5_extended.json"),
                str(tools_config),
                "basic_tools",
                str(outdir / "multi_turn"),
                modes=[selected_eval_mode],
                profiles=profiles,
                max_turns=3,
                max_plan_steps=int(body.get("max_plan_steps") or 4),
                evidence_policy="lite",
            )
            combined_summary = {}
            combined_summary.update(_flatten_eval_summary(single_eval.get("summary", {}), "单轮"))
            combined_summary.update(_flatten_eval_summary(plan_eval.get("summary", {}), "多轮/自适应"))
            evaluation = {
                "profiles": profiles or ["qwen_4b", "qwen_7b"],
                "modes": [selected_eval_mode],
                "selected_mode": selected_eval_mode,
                "single_turn": single_eval,
                "multi_turn": plan_eval,
                "combined_summary": combined_summary,
            }
            _write_json(outdir / "eval_summary.json", evaluation)
            output = {"evaluation": evaluation, "source": "live_evaluation"}
            data_source = f"batch_eval live ({selected_eval_mode})"
        params = {
            "action": action,
            "profiles": evaluation.get("profiles", []),
            "modes": evaluation.get("modes", []),
            "selected_mode": evaluation.get("selected_mode") or selected_eval_mode,
            "model_comparison": combined_summary,
            "data_source": data_source,
            "judgement_note": "失败不等于无法输出结果；批量评测中的失败通常表示工具路径、参数完整性或结构化输出未达到预设最优标准。",
        }
    elif action == "execution_engine":
        if mode == "mock":
            output, artifact_source = _load_b4_mock_plan_fixture()
            trace = output.get("trace", {})
            data_source = f"module_demos 已保存产物: {artifact_source}"
        else:
            output = run_plan_execute(
                str(model_config),
                messages,
                schema,
                str(tools_config),
                "basic_tools",
                mode,
                str(outdir),
                3,
                int(body.get("max_plan_steps") or 4),
                "lite",
                profile,
            )
            trace = output.get("trace", {})
            data_source = "本次实时执行（计划状态管理与证据传播）"
        plan_state = trace.get("plan_state", {}) if isinstance(trace.get("plan_state"), dict) else {}
        steps = plan_state.get("steps") if isinstance(plan_state.get("steps"), list) else []
        completed_steps = sum(1 for step in steps if str(step.get("status") or "") == "completed")
        failed_steps = sum(1 for step in steps if str(step.get("status") or "") == "failed")
        pending_steps = sum(1 for step in steps if str(step.get("status") or "") == "pending")
        artifact_flags = {
            "trace_written": True if mode == "mock" else (outdir / "trace.json").exists(),
            "plan_preview_written": True if mode == "mock" else (outdir / "plan_preview.md").exists(),
            "progress_written": True if mode == "mock" else (outdir / "progress.md").exists(),
        }
        execution_engine_report = {
            "plan_mode": plan_state.get("mode") or "plan_execute",
            "evidence_policy": plan_state.get("evidence_policy") or "lite",
            "total_steps": len(steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "artifacts": artifact_flags,
        }
        output["execution_engine_report"] = execution_engine_report
        params = {
            "action": action,
            "total_steps": execution_engine_report["total_steps"],
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "evidence_policy": execution_engine_report["evidence_policy"],
            "evidence_guard_enabled": True,
            "trace_written": execution_engine_report["artifacts"]["trace_written"],
            "plan_preview_written": execution_engine_report["artifacts"]["plan_preview_written"],
            "progress_written": execution_engine_report["artifacts"]["progress_written"],
            "tool_rounds_used": trace.get("tool_rounds_used", 0),
            "llm_call_count": trace.get("llm_call_count", 0),
            "selected_profile": profile,
            "mode": mode,
            "data_source": data_source,
        }
    else:
        result = generate_ai_message(str(model_config), messages, schema, mode, str(outdir), "demo", profile)
        output = result
        raw_record = result.get("raw_record", {})
        usage = raw_record.get("usage") or {}
        output["capability_checklist"] = {
            "core_function": "generate_ai_message",
            "supported_modes": ["mock", "prompt_json", "native_tools"],
            "supports_parallel_tool_calls": True,
            "supports_multiple_tool_messages": True,
            "supports_retry": True,
            "supports_compress_tool_messages": True,
        }
        params = {
            "action": action,
            "demo_module": "第一模块",
            "core_function": "generate_ai_message",
            "generated_tool_calls": len(result.get("ai_message", {}).get("tool_calls", [])),
            "received_tool_messages": sum(1 for item in messages if item.get("role") == "tool"),
            "followup_tool_calls": 0,
            "repair_attempted": bool(raw_record.get("repair_attempted")),
            "attempt_count": len(raw_record.get("attempts") or []),
            "selected_profile": raw_record.get("model_profile") or profile,
            "resolved_model_path": raw_record.get("resolved_model_path"),
            "mode": raw_record.get("mode"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "compression_feature": "compress_tool_messages",
        }
    output["result_parameters"] = params
    return {"module": "B4", "input": input_payload, "output": output, "artifacts": {"run_dir": str(outdir)}}

def _run_b5(body: dict, outdir: Path) -> dict:
    from b5_memory import load_memory_advanced
    from common.io_utils import read_yaml
    query, mode, top_k = str(body.get("query") or "Agent 系统如何调用工具？").strip(), str(body.get("search_mode") or "keyword"), int(body.get("top_k") or 5)
    selected_ids = body.get("selected_memory_ids") or []
    if not isinstance(selected_ids, list): raise ValueError("selected_memory_ids must be a JSON array")
    memory_config = PROJECT_ROOT / "configs" / "memory.yaml"
    result = load_memory_advanced(str(memory_config), selected_ids, bool(body.get("use_global_memory", True)), query, top_k, mode, str(outdir))
    config = read_yaml(memory_config)
    ranked_results = result.get("results") if isinstance(result.get("results"), list) else []
    scores = [float(item.get("score") or item.get("similarity") or 0) for item in ranked_results]
    result_parameters = {
        "requested_top_k": top_k,
        "returned_count": len(ranked_results) if ranked_results else len(result.get("selected_memory_docs", [])),
        "top_k_scores_desc": scores,
        "scores_descending": scores == sorted(scores, reverse=True),
        "max_memory_chars": int(config["memory"]["max_memory_chars"]),
        "total_included_chars": int(result.get("total_chars") or 0),
        "content_truncated": bool(result.get("truncated", False)),
        "length_management_applied": "selected_memory_docs" in result,
    }
    if result.get("search_mode") == "vector":
        result_parameters["vector_evaluation"] = {
            "method": result.get("vector_method"),
            "dimensions": result.get("vector_dimensions"),
            "similarity_metric": "cosine_similarity",
            "minimum_score": 0.01,
        }
    result["result_parameters"] = result_parameters
    return {"module": "B5", "input": {"query": query, "search_mode": mode, "top_k": top_k}, "output": result, "artifacts": {"run_dir": str(outdir)}}

RUNNERS = {"b1": _run_b1, "b2": _run_b2, "b3": _run_b3, "b4": _run_b4, "b5": _run_b5}

def make_handler(module: str | None = None):
    class DemoHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(STATIC_DIR), **kwargs)
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/modules":
                _reply(self, {"status": "ok", "modules": MODULE_META})
                return
            if path == "/api/presence":
                _reply(self, _presence_payload(module))
                return
            if path == "/api/meta":
                if module:
                    _reply(self, {"status": "ok", "module": module, **MODULE_META[module]})
                else:
                    _reply(self, {"status": "ok", "module": "all", "title": "B1-B5 Agent Module Studio", "modules": MODULE_META})
                return
            if self.path == "/": self.path = "/index.html"
            super().do_GET()
        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/presence":
                _reply(self, _presence_payload(module))
                return
            selected_module = module
            path_parts = path.strip("/").split("/")
            if len(path_parts) == 4 and path_parts[:2] == ["api", "modules"] and path_parts[3] == "run":
                selected_module = path_parts[2]
            elif path != "/api/run":
                self.send_error(404)
                return
            if selected_module not in RUNNERS:
                _reply(self, {"status": "error", "error": {"type": "ValueError", "message": "unknown module"}}, 404)
                return
            try:
                outdir = OUTPUT_ROOT / selected_module / _now_tag(); outdir.mkdir(parents=True, exist_ok=True)
                _reply(self, {"status": "ok", **RUNNERS[selected_module](_read_body(self), outdir)})
            except Exception as exc: _reply(self, {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, 400)
        def log_message(self, format, *args): print(f"[{(module or 'all').upper()}] {format % args}")
    return DemoHandler

def run_server(module: str | None = None, host: str = "127.0.0.1", port: int | None = None) -> None:
    selected_port = port or (MODULE_META[module]["port"] if module else 8100)
    server = ThreadingHTTPServer((host, selected_port), make_handler(module))
    title = MODULE_META[module]["title"] if module else "B1-B5 Agent Module Studio"
    print(f"{title} running: http://{host}:{selected_port}/")
    server.serve_forever()

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--module", choices=sorted(MODULE_META)); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int)
    args = parser.parse_args(); run_server(args.module, args.host, args.port); return 0

if __name__ == "__main__": raise SystemExit(main())
