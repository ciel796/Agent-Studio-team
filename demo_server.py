from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import time
from urllib.parse import urlparse

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
CODE_DIR = PROJECT_ROOT / "code"
STATIC_DIR = DEMO_DIR / "static"
OUTPUT_ROOT = DEMO_DIR / "outputs"
PRESENCE_TTL_SECONDS = 35
PRESENCE_LOCK = Lock()
ACTIVE_CLIENTS: dict[str, dict] = {}
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
        "title": "B4 Local LLM Console", "owner": "C", "summary": "输入 messages 和推理模式，独立展示模型产生的标准 AIMessage。",
        "input": "messages + tools_schema + mode", "output": "AIMessage + raw record", "port": 8104,
        "input_from": "B1 传入 messages，B3 传入 Tools Schema",
        "output_interface": "POST /api/modules/b4/run",
        "receiver": "B1 接收 AIMessage；若包含 tool_calls，B1 将调用请求转交 B3",
        "integration": "B4 只负责模型推理和工具选择，不直接读取文件或执行 Skill。",
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

def _safe_client_id(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or "anonymous"))[:64]
    return normalized or "anonymous"

def _presence_snapshot() -> dict:
    cutoff = time() - PRESENCE_TTL_SECONDS
    with PRESENCE_LOCK:
        expired = [client_id for client_id, item in ACTIVE_CLIENTS.items() if float(item.get("last_seen", 0)) < cutoff]
        for client_id in expired:
            ACTIVE_CLIENTS.pop(client_id, None)
        users = [{"client_id": client_id, "display_name": item["display_name"]} for client_id, item in ACTIVE_CLIENTS.items()]
    users.sort(key=lambda item: (item["display_name"], item["client_id"]))
    return {"online_count": len(users), "users": users, "ttl_seconds": PRESENCE_TTL_SECONDS}

def _touch_presence(client_id: object, display_name: object, remote_address: str = "") -> dict:
    safe_id = _safe_client_id(client_id)
    safe_name = str(display_name or f"演示者-{safe_id[-4:]}").strip()[:32] or f"演示者-{safe_id[-4:]}"
    with PRESENCE_LOCK:
        ACTIVE_CLIENTS[safe_id] = {"display_name": safe_name, "last_seen": time(), "remote_address": remote_address}
    return {"client_id": safe_id, "display_name": safe_name, **_presence_snapshot()}

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
    from b4_local_agent_llm import generate_ai_message, run_plan_execute
    action = str(body.get("action") or "generation")
    mode = str(body.get("mode") or "mock")
    profile = str(body.get("profile") or "qwen_4b")
    messages = body.get("messages") or [{"role": "system", "content": "You are a local tool-using agent."}, {"role": "user", "content": "读取 docs/agent_intro.txt 并总结三点。"}]
    schema = body.get("tools_schema")
    if schema is None: schema = json.loads((PROJECT_ROOT / "data" / "messages" / "tools_schema_basic.json").read_text(encoding="utf-8"))
    if not isinstance(messages, list) or not isinstance(schema, list): raise ValueError("messages and tools_schema must be JSON arrays")
    model_config = PROJECT_ROOT / "configs" / "model_new.yaml"
    if action == "multi_tool_roundtrip":
        first = generate_ai_message(str(model_config), messages, schema, "mock", str(outdir), "multi_request", profile)
        received_messages = json.loads((PROJECT_ROOT / "data" / "messages" / "messages_with_multi_tool_batch_test.json").read_text(encoding="utf-8"))
        second = generate_ai_message(str(model_config), received_messages, schema, "mock", str(outdir), "multi_response", profile)
        output = {"first_ai_message": first["ai_message"], "received_tool_messages": [item for item in received_messages if item.get("role") == "tool"], "second_ai_message": second["ai_message"]}
        params = {"action": action, "generated_tool_calls": len(first["ai_message"].get("tool_calls", [])), "received_tool_messages": len(output["received_tool_messages"]), "followup_tool_calls": len(second["ai_message"].get("tool_calls", [])), "selected_profile": profile, "mode": "mock"}
    elif action == "plan_execute":
        if mode == "mock":
            verified_dir = PROJECT_ROOT / "outputs" / "B1" / "test_b1_plan_execute"
            trace = json.loads((verified_dir / "trace.json").read_text(encoding="utf-8"))
            plan_preview = (verified_dir / "plan_preview.md").read_text(encoding="utf-8")
            output = {"status": trace.get("status"), "final_answer": (verified_dir / "final_answer.md").read_text(encoding="utf-8"), "messages": json.loads((verified_dir / "messages.json").read_text(encoding="utf-8")), "trace": trace, "plan_preview": plan_preview, "source": "verified_server_artifact"}
            data_source = "服务器成功运行轨迹"
            verified_plan_steps = len(re.findall(r"^\d+\.", plan_preview, flags=re.MULTILINE))
        else:
            plan_messages = json.loads((PROJECT_ROOT / "data" / "messages" / "messages_life_rent_plan_execute.json").read_text(encoding="utf-8"))
            output = run_plan_execute(str(model_config), plan_messages, schema, str(PROJECT_ROOT / "configs" / "tools.yaml"), "basic_tools", mode, str(outdir), 3, int(body.get("max_plan_steps") or 4), "lite", profile)
            trace = output.get("trace", {})
            data_source = "本次实时执行"
        plan_state = trace.get("plan_state", {})
        params = {"action": action, "plan_steps": verified_plan_steps if mode == "mock" else len(plan_state.get("steps", [])), "plan_status": output.get("status"), "llm_call_count": trace.get("llm_call_count", 0), "tool_rounds_used": trace.get("tool_rounds_used", 0), "selected_profile": profile, "mode": mode, "data_source": data_source}
    elif action == "injection_compare":
        comparison = json.loads((PROJECT_ROOT / "outputs" / "B3" / "compare_tools_injection" / "comparison.json").read_text(encoding="utf-8"))
        output = {"comparison": comparison, "source": "verified_server_artifact"}
        params = {"action": action, "prompt_json_success": comparison["prompt_json"]["structured_output_success"], "native_tools_success": comparison["native_tools"]["structured_output_success"], "prompt_json_input_tokens": comparison["prompt_json"]["input_tokens"], "native_tools_input_tokens": comparison["native_tools"]["input_tokens"], "prompt_json_output_tokens": comparison["prompt_json"]["output_tokens"], "native_tools_output_tokens": comparison["native_tools"]["output_tokens"], "data_source": "服务器实测产物"}
    elif action == "batch_eval":
        evaluation = json.loads((PROJECT_ROOT / "outputs" / "B4_compat" / "test_b4_compat_batch_eval" / "eval_summary.json").read_text(encoding="utf-8"))
        output = {"evaluation": evaluation, "source": "verified_server_artifact"}
        params = {"action": action, "profiles": evaluation.get("profiles", []), "modes": evaluation.get("modes", []), "model_comparison": evaluation.get("summary", {}), "data_source": "服务器 6 条批量样例"}
    else:
        result = generate_ai_message(str(model_config), messages, schema, mode, str(outdir), "demo", profile)
        output = result
        raw_record = result.get("raw_record", {})
        usage = raw_record.get("usage") or {}
        params = {"action": action, "generated_tool_calls": len(result.get("ai_message", {}).get("tool_calls", [])), "received_tool_messages": sum(1 for item in messages if item.get("role") == "tool"), "selected_profile": raw_record.get("model_profile") or profile, "resolved_model_path": raw_record.get("resolved_model_path"), "mode": raw_record.get("mode"), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
    output["result_parameters"] = params
    return {"module": "B4", "input": {"action": action, "mode": mode, "profile": profile, "messages": messages, "tools_count": len(schema)}, "output": output, "artifacts": {"run_dir": str(outdir)}}

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
            if path == "/api/presence":
                _reply(self, {"status": "ok", **_presence_snapshot()})
                return
            if path == "/api/modules":
                _reply(self, {"status": "ok", "modules": MODULE_META})
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
                try:
                    body = _read_body(self)
                    presence = _touch_presence(body.get("client_id"), body.get("display_name"), self.client_address[0])
                    _reply(self, {"status": "ok", **presence})
                except Exception as exc:
                    _reply(self, {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, 400)
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
                body = _read_body(self)
                presence = _touch_presence(body.get("_client_id"), body.get("_display_name"), self.client_address[0])
                client_id = presence["client_id"]
                outdir = OUTPUT_ROOT / selected_module / client_id / _now_tag(); outdir.mkdir(parents=True, exist_ok=True)
                _reply(self, {"status": "ok", "session": {"client_id": client_id, "display_name": presence["display_name"]}, **RUNNERS[selected_module](body, outdir)})
            except Exception as exc: _reply(self, {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, 400)
        def log_message(self, format, *args): print(f"[{(module or 'all').upper()}] {format % args}")
    return DemoHandler

def run_server(module: str | None = None, host: str = "127.0.0.1", port: int | None = None) -> None:
    selected_port = port or (MODULE_META[module]["port"] if module else 8100)
    server = ThreadingHTTPServer((host, selected_port), make_handler(module))
    server.daemon_threads = True
    title = MODULE_META[module]["title"] if module else "B1-B5 Agent Module Studio"
    print(f"{title} running: http://{host}:{selected_port}/")
    server.serve_forever()

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--module", choices=sorted(MODULE_META)); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int)
    args = parser.parse_args(); run_server(args.module, args.host, args.port); return 0

if __name__ == "__main__": raise SystemExit(main())
