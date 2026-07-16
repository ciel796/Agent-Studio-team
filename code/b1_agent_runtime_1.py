from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, ensure_dir, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import validate_ai_message


CHECKPOINT_FILENAME = "checkpoint.json"
PENDING_QUESTION_FILENAME = "pending_question.md"
SUPPORTED_AGENT_MODES = {"integrated", "react_one_round", "plan_execute", "adaptive_execute"}
SUPPORTED_LLM_MODES = {"mock", "prompt_json", "native_tools", "adaptive"}
EXIT_WORDS = {"", "exit", "quit", "q", "结束", "退出"}


def _validate_runtime_input(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("runtime_input.json must contain an object")
    execution_mode = payload.setdefault("execution_mode", "integrated")
    if execution_mode not in {"integrated", "fixture"}:
        raise ValueError("execution_mode must be integrated or fixture")
    agent_mode = str(payload.setdefault("agent_mode", "integrated")).strip() or "integrated"
    if agent_mode not in SUPPORTED_AGENT_MODES:
        raise ValueError(f"agent_mode must be one of: {', '.join(sorted(SUPPORTED_AGENT_MODES))}")
    payload["agent_mode"] = agent_mode
    agent_options = payload.get("agent_options")
    if agent_options is None:
        agent_options = {}
    if not isinstance(agent_options, dict):
        raise ValueError("agent_options must be an object when provided")
    payload["agent_options"] = deepcopy(agent_options)
    requested_llm_mode = payload["agent_options"].get("llm_mode")
    if requested_llm_mode is not None:
        if not isinstance(requested_llm_mode, str) or not requested_llm_mode.strip():
            raise ValueError("agent_options.llm_mode must be a non-empty string when provided")
        requested_llm_mode = requested_llm_mode.strip()
        if requested_llm_mode not in SUPPORTED_LLM_MODES:
            raise ValueError(f"agent_options.llm_mode must be one of: {', '.join(sorted(SUPPORTED_LLM_MODES))}")
        payload["agent_options"]["llm_mode"] = requested_llm_mode
        if requested_llm_mode == "adaptive" and agent_mode != "adaptive_execute":
            raise ValueError("agent_options.llm_mode=adaptive requires agent_mode=adaptive_execute")
        if agent_mode == "adaptive_execute" and requested_llm_mode == "adaptive":
            payload["agent_options"]["low_mode"] = str(payload["agent_options"].get("low_mode") or "native_tools")
            payload["agent_options"]["high_mode"] = str(payload["agent_options"].get("high_mode") or "prompt_json")
    tools_schema_source = str(payload.get("tools_schema_source", "dynamic") or "dynamic").strip().lower()
    if tools_schema_source not in {"dynamic", "static"}:
        raise ValueError("tools_schema_source must be dynamic or static")
    payload["tools_schema_source"] = tools_schema_source
    tools_schema_path = payload.get("tools_schema_path")
    if tools_schema_path is not None and not isinstance(tools_schema_path, str):
        raise ValueError("tools_schema_path must be a string when provided")
    if tools_schema_source == "static" and not isinstance(tools_schema_path, str):
        raise ValueError("tools_schema_source=static requires tools_schema_path")
    required = ["conversation_id", "system_prompt_path", "toolset", "max_turns", "save_memory"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"runtime input missing: {', '.join(missing)}")
    if not isinstance(payload["conversation_id"], str) or not payload["conversation_id"]:
        raise ValueError("conversation_id must be a non-empty string")

    user_inputs = payload.get("user_inputs")
    if user_inputs is None:
        user_input = payload.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string when user_inputs is not provided")
        payload["user_inputs"] = [user_input]
    else:
        if not isinstance(user_inputs, list) or not user_inputs:
            raise ValueError("user_inputs must be a non-empty list of strings")
        cleaned_inputs = []
        for index, item in enumerate(user_inputs):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"user_inputs[{index}] must be a non-empty string")
            cleaned_inputs.append(item.strip())
        payload["user_inputs"] = cleaned_inputs
    payload["user_input"] = payload["user_inputs"][0]

    if not isinstance(payload["max_turns"], int) or isinstance(payload["max_turns"], bool) or payload["max_turns"] < 1:
        raise ValueError("max_turns must be a positive integer")
    if payload["save_memory"] not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")
    memory_context_mode = str(payload.setdefault("memory_context_mode", "full")).strip().lower()
    if memory_context_mode not in {"full", "summary"}:
        raise ValueError("memory_context_mode must be full or summary")
    payload["memory_context_mode"] = memory_context_mode
    if execution_mode == "fixture":
        fixtures = payload.get("fixtures")
        if not isinstance(fixtures, dict):
            raise ValueError("fixture mode requires a fixtures object")
        required_fixtures = [
            "selected_memory_path",
            "tools_schema_path",
            "ai_messages_path",
            "tool_messages_path",
        ]
        missing_fixtures = [field for field in required_fixtures if not isinstance(fixtures.get(field), str)]
        if missing_fixtures:
            raise ValueError(f"fixtures missing paths: {', '.join(missing_fixtures)}")
        if payload["save_memory"] != "none":
            raise ValueError("fixture mode requires save_memory=none")
    else:
        selected_ids = payload.setdefault("selected_memory_ids", [])
        if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
            raise ValueError("selected_memory_ids must be a list of strings")
        payload.setdefault("use_global_memory", False)
        if not isinstance(payload["use_global_memory"], bool):
            raise ValueError("use_global_memory must be boolean")
    _validate_history_compression_config(payload)
    _validate_system_prompt_events(payload)
    return payload

def _memory_context(selected_memory: dict, mode: str = "full") -> str:
    sections = []
    for document in selected_memory.get("selected_memory_docs", []):
        content = document.get("content", "")
        if mode == "summary" and str(document.get("summary") or "").strip():
            content = str(document["summary"]).strip()
        sections.append(
            f'<memory id="{document["memory_id"]}" type="{document["memory_type"]}" context_mode="{mode}">\n'
            f'{str(content).strip()}\n</memory>'
        )
    return "\n\n".join(sections)


def _conversation_history_context(runtime: dict, max_chars: int = 12000, max_messages: int = 12) -> list[dict]:
    history = runtime.get("conversation_history")
    if not isinstance(history, list):
        return []
    normalized = [
        {"role": item.get("role"), "content": str(item.get("content") or "").strip()}
        for item in history[-max(1, max_messages):]
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and str(item.get("content") or "").strip()
    ]
    current_input = str(runtime.get("user_input") or "").strip()
    if normalized and normalized[-1]["role"] == "user" and normalized[-1]["content"] == current_input:
        normalized.pop()
    selected = []
    used_chars = 0
    for message in reversed(normalized):
        message_chars = len(message["content"])
        if selected and used_chars + message_chars > max_chars:
            break
        selected.append(message)
        used_chars += message_chars
    return list(reversed(selected))


def _conversation_summary_context(history: list[dict]) -> str:
    if not history:
        return ""
    from b5_memory import _conversation_summary

    summary = _conversation_summary(history)
    if not summary or summary.startswith("暂无需要跨轮次保留"):
        return ""
    current_turn = sum(1 for message in history if message.get("role") == "user") + 1
    return (
        '<conversation_context priority="active_user_requirements">\n'
        "以下是当前聊天窗口中仍然有效的用户要求、偏好和关键上下文。"
        "其中“用户要求与偏好”必须在本轮回答中继续遵守，除非用户明确取消或修改。\n"
        f"当前即将回答该聊天窗口的第 {current_turn} 个用户轮次。\n"
        f"{summary}\n"
        "</conversation_context>"
    )


def _compose_system_prompt(prompt_text: str, memory_context: str = "") -> str:
    prompt = prompt_text.strip()
    if memory_context:
        prompt = f"{prompt}\n\n{memory_context}"
    return prompt


def _validate_system_prompt_events(payload: dict) -> list[dict]:
    events = payload.setdefault("system_prompt_events", [])
    if events is None:
        events = []
    if not isinstance(events, list):
        raise ValueError("system_prompt_events must be a list")
    normalized = []
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError(f"system_prompt_events[{index}] must be an object")
        user_turn_index = event.get("user_turn_index", event.get("turn_index"))
        if not isinstance(user_turn_index, int) or isinstance(user_turn_index, bool) or user_turn_index < 1:
            raise ValueError(f"system_prompt_events[{index}].user_turn_index must be a positive integer")
        mode = str(event.get("mode", "add")).strip().lower()
        aliases = {"replace": "switch", "append": "add"}
        mode = aliases.get(mode, mode)
        if mode not in {"switch", "add"}:
            raise ValueError(f"system_prompt_events[{index}].mode must be switch or add")
        path_value = event.get("system_prompt_path", event.get("prompt_path"))
        content = event.get("content", event.get("system_prompt"))
        if not isinstance(path_value, str) and not isinstance(content, str):
            raise ValueError(f"system_prompt_events[{index}] requires system_prompt_path or content")
        normalized.append(
            {
                "event_index": index,
                "user_turn_index": user_turn_index,
                "mode": mode,
                "label": str(event.get("label") or f"system_prompt_event_{index:03d}"),
                "system_prompt_path": path_value if isinstance(path_value, str) else None,
                "content": content if isinstance(content, str) else None,
            }
        )
    payload["system_prompt_events"] = normalized
    return normalized


def _load_system_prompt_event_text(event: dict, input_file: Path) -> tuple[str, str | None]:
    if event.get("system_prompt_path"):
        prompt_path = resolve_from_file(event["system_prompt_path"], input_file)
        return read_text(prompt_path).strip(), str(prompt_path)
    content = str(event.get("content") or "").strip()
    if not content:
        raise ValueError(f"system prompt event {event.get('event_index')} has empty content")
    return content, None


def _apply_system_prompt_events(state: dict, output_dir: Path, user_turn_index: int) -> None:
    events = state.get("runtime", {}).get("system_prompt_events", [])
    if not events:
        return
    applied_keys = {
        (item.get("event_index"), item.get("user_turn_index"))
        for item in state.setdefault("system_prompt_events_applied", [])
    }
    input_file = Path(state.get("input_path", ".")).resolve()
    memory_context = state.get("memory_context", "")
    applied_now = False
    for event in events:
        key = (event.get("event_index"), event.get("user_turn_index"))
        if event.get("user_turn_index") != user_turn_index or key in applied_keys:
            continue
        prompt_text, resolved_path = _load_system_prompt_event_text(event, input_file)
        if event["mode"] == "switch":
            if not state.get("messages"):
                state["messages"] = []
            if not state["messages"] or state["messages"][0].get("role") != "system":
                state["messages"].insert(0, {"role": "system", "content": ""})
            state["messages"][0]["content"] = _compose_system_prompt(prompt_text, memory_context)
            state["active_system_prompt"] = {
                "label": event["label"],
                "source_path": resolved_path,
                "updated_at_turn": user_turn_index,
            }
        else:
            state["messages"].append(
                {
                    "role": "system",
                    "content": (
                        f"<additional_system_prompt label=\"{event['label']}\" "
                        f"turn=\"{user_turn_index}\">\n{prompt_text}\n</additional_system_prompt>"
                    ),
                }
            )
        state["system_prompt_events_applied"].append(
            {
                "event_index": event["event_index"],
                "user_turn_index": user_turn_index,
                "mode": event["mode"],
                "label": event["label"],
                "source_path": resolved_path,
                "applied_at": now_iso(),
            }
        )
        applied_now = True
    if applied_now:
        _save_checkpoint(output_dir, state)


def _default_llm_mode(model_config: Path) -> str:
    config = read_yaml(model_config)
    return config.get("runtime", {}).get("default_mode", "mock")


def generate_ai_message(*args, **kwargs) -> dict:
    """Lazy B4 proxy retained as the integrated-mode injection point."""
    from b4_local_agent_llm import generate_ai_message as b4_generate_ai_message

    return b4_generate_ai_message(*args, **kwargs)


def run_react_one_round_execute(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    tools_config: str,
    toolset: str,
    mode: str,
    outdir: str,
    max_tool_rounds: int = 1,
    forced_profile: str | None = None,
    routing: dict | None = None,
    runtime_bridge: dict | None = None,
) -> dict:
    from b4_local_agent_llm import _run_react_one_round_execute_impl

    return _run_react_one_round_execute_impl(
        model_config,
        messages,
        tools_schema,
        tools_config,
        toolset,
        mode,
        outdir,
        max_tool_rounds=max_tool_rounds,
        forced_profile=forced_profile,
        routing=routing,
        runtime_bridge=runtime_bridge,
    )


def run_adaptive_execute(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    tools_config: str,
    toolset: str,
    outdir: str,
    max_turns: int = 3,
    max_plan_steps: int = 6,
    evidence_policy: str = "strict",
    forced_profile: str | None = None,
    low_mode: str = "native_tools",
    high_mode: str = "prompt_json",
    runtime_bridge: dict | None = None,
) -> dict:
    from b4_local_agent_llm import _run_adaptive_execute_impl

    return _run_adaptive_execute_impl(
        model_config,
        messages,
        tools_schema,
        tools_config,
        toolset,
        outdir,
        max_turns=max_turns,
        max_plan_steps=max_plan_steps,
        evidence_policy=evidence_policy,
        forced_profile=forced_profile,
        low_mode=low_mode,
        high_mode=high_mode,
        runtime_bridge=runtime_bridge,
    )


def run_plan_execute(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    tools_config: str,
    toolset: str,
    mode: str,
    outdir: str,
    max_turns: int = 3,
    max_plan_steps: int = 6,
    evidence_policy: str = "strict",
    forced_profile: str | None = None,
    runtime_bridge: dict | None = None,
) -> dict:
    from b4_local_agent_llm import _run_plan_execute_impl

    return _run_plan_execute_impl(
        model_config,
        messages,
        tools_schema,
        tools_config,
        toolset,
        mode,
        outdir,
        max_turns=max_turns,
        max_plan_steps=max_plan_steps,
        evidence_policy=evidence_policy,
        forced_profile=forced_profile,
        runtime_bridge=runtime_bridge,
    )


def _load_fixture_inputs(input_file: Path, runtime: dict) -> dict:
    fixtures = runtime["fixtures"]
    selected_memory = read_json(resolve_from_file(fixtures["selected_memory_path"], input_file))
    tools_schema = read_json(resolve_from_file(fixtures["tools_schema_path"], input_file))
    ai_messages = read_json(resolve_from_file(fixtures["ai_messages_path"], input_file))
    tool_messages = read_json(resolve_from_file(fixtures["tool_messages_path"], input_file))
    if not isinstance(selected_memory, dict):
        raise ValueError("preset memory must be a JSON object")
    if not isinstance(tools_schema, list):
        raise ValueError("preset tools_schema must be a JSON array")
    if not isinstance(ai_messages, list) or not ai_messages:
        raise ValueError("preset AI messages must be a non-empty JSON array")
    if not isinstance(tool_messages, dict):
        raise ValueError("preset ToolMessages must be an object keyed by tool_call_id")
    for message in ai_messages:
        validate_ai_message(message)
    return {
        "selected_memory": selected_memory,
        "tools_schema": tools_schema,
        "ai_messages": ai_messages,
        "tool_messages": tool_messages,
    }


def _fixture_tool_messages(tool_calls: list[dict], preset_messages: dict) -> list[dict]:
    results = []
    for call in tool_calls:
        call_id = call.get("id")
        message = deepcopy(preset_messages.get(call_id))
        if not isinstance(message, dict):
            raise ValueError(f"fixture ToolMessage does not exist for tool_call_id: {call_id}")
        if message.get("role") != "tool" or message.get("tool_call_id") != call_id:
            raise ValueError(f"invalid fixture ToolMessage for tool_call_id: {call_id}")
        if message.get("name") != call.get("name"):
            raise ValueError(f"fixture ToolMessage name does not match call: {call_id}")
        results.append(message)
    return results


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_FILENAME


def _pending_question_path(output_dir: Path) -> Path:
    return output_dir / PENDING_QUESTION_FILENAME


def _save_checkpoint(output_dir: Path, state: dict) -> None:
    snapshot = deepcopy(state)
    snapshot["updated_at"] = now_iso()
    write_json(snapshot, _checkpoint_path(output_dir))


def _load_checkpoint(output_dir: Path) -> dict:
    checkpoint = _checkpoint_path(output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    state = read_json(checkpoint)
    if not isinstance(state, dict) or state.get("checkpoint_version") != 1:
        raise ValueError("checkpoint.json is not a valid B1 checkpoint")
    return state


def _clear_pending_question(output_dir: Path) -> None:
    pending_path = _pending_question_path(output_dir)
    if pending_path.exists():
        pending_path.unlink()


def _write_pending_question(output_dir: Path, question: str) -> str:
    normalized = question.strip()
    write_text(normalized + "\n", _pending_question_path(output_dir))
    return normalized


def _last_message(state: dict) -> dict | None:
    messages = state.get("messages") or []
    return messages[-1] if messages else None


def _last_user_turn(state: dict) -> dict | None:
    turns = state.get("user_turns") or []
    return turns[-1] if turns else None


def _set_pending_question(state: dict, output_dir: Path, question: str, source: str = "assistant") -> None:
    normalized = _write_pending_question(output_dir, question)
    current = _last_user_turn(state)
    pending_payload = {
        "question": normalized,
        "source": source,
        "requested_at": now_iso(),
        "user_turn_index": current.get("user_turn_index") if isinstance(current, dict) else None,
    }
    state["status"] = "needs_user"
    state["pending_question"] = pending_payload
    state["terminal_error"] = {
        "type": "HumanInTheLoop",
        "message": "agent requests user confirmation",
        "question": normalized,
    }
    if current and current.get("status") == "running":
        current["status"] = "needs_user"
        current["error"] = deepcopy(state["terminal_error"])
        current["final_answer"] = ""


def _resolve_pending_question(state: dict, output_dir: Path) -> None:
    pending = state.get("pending_question")
    if isinstance(pending, dict):
        resolved = deepcopy(pending)
        resolved["resolved_at"] = now_iso()
        state.setdefault("resolved_pending_questions", []).append(resolved)
    state["pending_question"] = None
    if state.get("status") == "needs_user":
        state["status"] = "running"
    _clear_pending_question(output_dir)


def _merge_runtime_input_for_resume(state: dict, resume_input_file: Path) -> None:
    if not resume_input_file.exists():
        return
    payload = _validate_runtime_input(read_json(resume_input_file))
    runtime = state["runtime"]
    if payload["conversation_id"] != runtime.get("conversation_id"):
        raise ValueError("resume runtime_input conversation_id must match checkpoint")
    if payload["execution_mode"] != state.get("execution_mode"):
        raise ValueError("resume runtime_input execution_mode must match checkpoint")
    if payload["agent_mode"] != runtime.get("agent_mode"):
        raise ValueError("resume runtime_input agent_mode must match checkpoint")

    current_user_inputs = runtime.get("user_inputs") if isinstance(runtime.get("user_inputs"), list) else []
    resume_user_inputs = payload.get("user_inputs") if isinstance(payload.get("user_inputs"), list) else []
    if resume_user_inputs:
        if len(resume_user_inputs) >= len(current_user_inputs) and resume_user_inputs[: len(current_user_inputs)] == current_user_inputs:
            runtime["user_inputs"] = resume_user_inputs
        elif state.get("status") == "needs_user":
            runtime["user_inputs"] = current_user_inputs + resume_user_inputs
    runtime["system_prompt_events"] = deepcopy(payload.get("system_prompt_events") or runtime.get("system_prompt_events") or [])
    state["input_path"] = str(resume_input_file)


def _final_answer_text(final_answers: list[dict]) -> str:
    if not final_answers:
        return ""
    if len(final_answers) == 1:
        return final_answers[0]["content"]
    sections = []
    for item in final_answers:
        sections.append(f"第{item['user_turn_index']}轮回答：\n{item['content']}")
    return "\n\n".join(sections)


def _validate_history_compression_config(payload: dict) -> dict:
    raw = payload.setdefault("history_compression", {"enabled": False})
    if raw is None:
        raw = {"enabled": False}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        raise ValueError("history_compression must be an object or boolean")
    enabled = bool(raw.get("enabled", False))
    max_messages = int(raw.get("max_messages", 12))
    keep_recent_messages = int(raw.get("keep_recent_messages", 4))
    summary_max_chars = int(raw.get("summary_max_chars", 2000))
    if max_messages < 4:
        raise ValueError("history_compression.max_messages must be at least 4")
    if keep_recent_messages < 1:
        raise ValueError("history_compression.keep_recent_messages must be positive")
    if keep_recent_messages >= max_messages:
        raise ValueError("history_compression.keep_recent_messages must be less than max_messages")
    if summary_max_chars < 200:
        raise ValueError("history_compression.summary_max_chars must be at least 200")
    config = {
        "enabled": enabled,
        "max_messages": max_messages,
        "keep_recent_messages": keep_recent_messages,
        "summary_max_chars": summary_max_chars,
    }
    payload["history_compression"] = config
    return config


def _message_brief(message: dict, index: int, limit: int = 260) -> str:
    role = message.get("role", "unknown")
    content = str(message.get("content", "")).replace("\n", " ").strip()
    if len(content) > limit:
        content = content[: limit - 3] + "..."
    if role == "assistant" and message.get("tool_calls"):
        tool_names = ", ".join(call.get("name", "unknown") for call in message.get("tool_calls", []))
        return f"{index}. assistant 请求工具：{tool_names}"
    if role == "tool":
        return f"{index}. tool {message.get('name', 'unknown')} 状态={message.get('status', 'unknown')}：{content}"
    return f"{index}. {role}：{content}"


def _summarize_messages(messages: list[dict], max_chars: int) -> str:
    lines = [
        "此前对话摘要：",
        "以下内容由 B1 在历史消息过长时自动压缩，用于在保留关键上下文的同时继续后续对话。",
    ]
    for index, message in enumerate(messages, 1):
        if message.get("role") == "system":
            continue
        lines.append(_message_brief(message, index))
    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 20].rstrip() + "\n...（摘要已截断）"
    return summary


def _maybe_compress_history(state: dict, output_dir: Path) -> None:
    config = state.get("runtime", {}).get("history_compression", {})
    if not isinstance(config, dict) or not config.get("enabled", False):
        return
    messages = state.get("messages") or []
    if len(messages) <= int(config["max_messages"]):
        return
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return
    keep_recent = int(config["keep_recent_messages"])
    first_system = messages[0]
    older = messages[1:-keep_recent]
    recent = messages[-keep_recent:]
    if not older:
        return
    compression_index = len(state.setdefault("history_compressions", [])) + 1
    summary = _summarize_messages(older, int(config["summary_max_chars"]))
    summary_message = {
        "role": "system",
        "content": f"<compressed_history index=\"{compression_index}\">\n{summary}\n</compressed_history>",
    }
    state["messages"] = [first_system, summary_message, *recent]
    record = {
        "compression_index": compression_index,
        "created_at": now_iso(),
        "before_message_count": len(messages),
        "after_message_count": len(state["messages"]),
        "compressed_message_count": len(older),
        "kept_recent_messages": keep_recent,
        "summary_chars": len(summary),
        "summary_path": f"history_summary_{compression_index:03d}.md",
    }
    state["history_compressions"].append(record)
    write_text(summary + "\n", output_dir / record["summary_path"])
    _save_checkpoint(output_dir, state)


def _build_initial_state(
    input_file: Path,
    runtime: dict,
    output_dir: Path,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    llm_mode: str | None,
) -> tuple[dict, dict | None, Path | None, Path | None, Path | None]:
    execution_mode = runtime["execution_mode"]
    prompt_path = resolve_from_file(runtime["system_prompt_path"], input_file)
    system_prompt = read_text(prompt_path).strip()
    fixture_data = None
    tools_file = memory_file = model_file = None

    if execution_mode == "fixture":
        fixture_data = _load_fixture_inputs(input_file, runtime)
        selected_memory = fixture_data["selected_memory"]
        tools_schema = fixture_data["tools_schema"]
        mode = "fixture"
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError("integrated mode requires tools_config, memory_config, and model_config")
        from b3_tool_layer import get_tools_schema
        from b5_memory import load_memory

        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        selected_memory = load_memory(
            str(memory_file),
            runtime["selected_memory_ids"],
            runtime["use_global_memory"],
            runtime["user_input"],
            str(output_dir),
        )
        if runtime.get("tools_schema_source") == "static":
            tools_schema_path = resolve_from_file(str(runtime["tools_schema_path"]), input_file)
            tools_schema = read_json(tools_schema_path)
            if not isinstance(tools_schema, list):
                raise ValueError("static tools_schema_path must point to a JSON array")
            write_json(tools_schema, output_dir / "tools_schema.json")
            write_json(
                {
                    "status": "success",
                    "toolset": runtime["toolset"],
                    "tool_count": len(tools_schema),
                    "schema_source": "static_file",
                    "tools_schema_path": str(tools_schema_path),
                },
                output_dir / "tool_schema_report.json",
            )
        else:
            tools_schema = get_tools_schema(str(tools_file), runtime["toolset"], str(output_dir))
        mode = _specialized_llm_mode(runtime, llm_mode) or _default_llm_mode(model_file)

    memory_context = _memory_context(selected_memory, runtime.get("memory_context_mode", "full"))
    if memory_context:
        system_prompt = f"{system_prompt}\n\n{memory_context}"
    conversation_history = _conversation_history_context(runtime)
    full_conversation_history = _conversation_history_context(runtime, max_chars=100000, max_messages=200)
    conversation_summary_context = _conversation_summary_context(full_conversation_history)
    if conversation_summary_context:
        system_prompt = f"{system_prompt}\n\n{conversation_summary_context}"
    warnings = []
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")

    state = {
        "checkpoint_version": 1,
        "created_at": now_iso(),
        "completed": False,
        "input_path": str(input_file),
        "tools_config": str(tools_file) if tools_file else None,
        "memory_config": str(memory_file) if memory_file else None,
        "model_config": str(model_file) if model_file else None,
        "runtime": runtime,
        "execution_mode": execution_mode,
        "llm_mode": mode,
        "selected_memory": selected_memory,
        "tools_schema": tools_schema,
        "messages": [{"role": "system", "content": system_prompt}, *conversation_history],
        "memory_context": memory_context,
        "conversation_summary_context": conversation_summary_context,
        "active_system_prompt": {"label": "initial", "source_path": str(prompt_path), "updated_at_turn": 0},
        "system_prompt_events_applied": [],
        "next_user_index": 0,
        "user_turns": [],
        "turns": [],
        "all_tool_messages": [],
        "final_answers": [],
        "tool_rounds": 0,
        "llm_calls": 0,
        "status": "running",
        "terminal_error": None,
        "pending_question": None,
        "resolved_pending_questions": [],
        "warnings": warnings,
        "memory_save": {"requested": runtime["save_memory"], "status": "not_requested"},
        "history_compressions": [],
    }
    return state, fixture_data, tools_file, memory_file, model_file


def _restore_runtime_handles(
    state: dict,
    input_file: Path,
) -> tuple[dict | None, Path | None, Path | None, Path | None]:
    fixture_data = None
    tools_file = memory_file = model_file = None
    if state["execution_mode"] == "fixture":
        fixture_data = _load_fixture_inputs(input_file, state["runtime"])
    else:
        from b3_tool_layer import execute_tool_calls  # noqa: F401

        if state.get("tools_config"):
            tools_file = Path(state["tools_config"]).resolve()
        if state.get("memory_config"):
            memory_file = Path(state["memory_config"]).resolve()
        if state.get("model_config"):
            model_file = Path(state["model_config"]).resolve()
    return fixture_data, tools_file, memory_file, model_file


def _append_next_user_input(state: dict, output_dir: Path, interactive: bool) -> bool:
    _maybe_compress_history(state, output_dir)
    runtime = state["runtime"]
    user_inputs = runtime["user_inputs"]
    next_user_index = int(state.get("next_user_index") or 0)
    if next_user_index >= len(user_inputs):
        if not interactive:
            if state.get("pending_question"):
                state["status"] = "needs_user"
            return False
        next_text = input("请输入下一轮用户问题，直接回车或输入 exit 结束：").strip()
        if next_text.lower() in EXIT_WORDS:
            if state.get("pending_question"):
                state["status"] = "needs_user"
            return False
        user_inputs.append(next_text)
    _resolve_pending_question(state, output_dir)
    user_turn_index = next_user_index + 1
    _apply_system_prompt_events(state, output_dir, user_turn_index)
    user_input = user_inputs[next_user_index]
    state.setdefault("messages", [])
    state.setdefault("user_turns", [])
    state["messages"].append({"role": "user", "content": user_input})
    state["user_turns"].append(
        {
            "user_turn_index": user_turn_index,
            "user_input": user_input,
            "status": "running",
            "tool_rounds_used": 0,
            "llm_call_indexes": [],
            "final_answer": "",
            "error": None,
        }
    )
    state["next_user_index"] = next_user_index + 1
    _save_checkpoint(output_dir, state)
    print(f"user_input[{user_turn_index}]: {user_input}")
    return True


def _current_pending_action(state: dict) -> str:
    last = _last_message(state)
    if last is None:
        return "need_user"
    role = last.get("role")
    if role in {"system", "assistant"} and not last.get("tool_calls"):
        return "need_user"
    if role in {"user", "tool"}:
        return "need_llm"
    if role == "assistant" and last.get("tool_calls"):
        return "need_tools"
    raise ValueError(f"unsupported message state for resume: role={role}")


def _record_terminal_error(state: dict, status: str, error_type: str, message: str, extra: dict | None = None) -> None:
    state["status"] = status
    error = {"type": error_type, "message": message}
    if extra:
        error.update(extra)
    state["terminal_error"] = error
    current = _last_user_turn(state)
    if current and current.get("status") == "running":
        current["status"] = status
        current["error"] = error


def _run_llm_step(
    state: dict,
    output_dir: Path,
    fixture_data: dict | None,
    model_file: Path | None,
) -> None:
    state["llm_calls"] += 1
    llm_calls = state["llm_calls"]
    turn_start = perf_counter()
    execution_mode = state["execution_mode"]
    current = _last_user_turn(state)
    user_turn_index = current["user_turn_index"] if current else None

    if execution_mode == "fixture":
        if llm_calls > len(fixture_data["ai_messages"]):
            raise ValueError("fixture AIMessage sequence ended before a final answer")
        ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
        llm_status = "success"
        llm_error = None
    else:
        llm_result = generate_ai_message(
            str(model_file),
            state["messages"],
            state["tools_schema"],
            state["llm_mode"],
            str(output_dir / "llm_calls"),
            f"llm_call_{llm_calls:03d}",
        )
        if not isinstance(llm_result, dict) or not isinstance(llm_result.get("ai_message"), dict):
            raise ValueError("B4 result must contain an ai_message object")
        ai_message = llm_result["ai_message"]
        llm_status = llm_result.get("status")
        llm_error = llm_result.get("error")

    state["messages"].append(ai_message)
    if current is not None:
        current["llm_call_indexes"].append(llm_calls)
    turn = {
        "turn_index": llm_calls,
        "user_turn_index": user_turn_index,
        "ai_message": ai_message,
        "llm_status": llm_status,
        "llm_error": llm_error,
        "tool_messages": [],
        "llm_latency_ms": round((perf_counter() - turn_start) * 1000, 3),
        "tool_latency_ms": None,
        "latency_ms": None,
    }
    state["turns"].append(turn)

    if llm_status != "success":
        _record_terminal_error(
            state,
            "llm_parse_error",
            "LLMParseError",
            "B4 failed to parse the model output as a valid AIMessage JSON object.",
            {"llm_call_index": llm_calls, "cause": llm_error},
        )
        return

    tool_calls = ai_message.get("tool_calls", [])
    if not tool_calls:
        final_answer = str(ai_message.get("content") or "")
        if final_answer.strip().startswith("ASK_USER:"):
            _set_pending_question(state, output_dir, final_answer, source="assistant")
            _save_checkpoint(output_dir, state)
            return
        print(f"content[{user_turn_index}]: {final_answer}")
        state["final_answers"].append({"user_turn_index": user_turn_index, "content": final_answer})
        if current is not None:
            current["status"] = "success"
            current["final_answer"] = final_answer
    _save_checkpoint(output_dir, state)


def _run_tool_step(
    state: dict,
    output_dir: Path,
    fixture_data: dict | None,
    tools_file: Path | None,
) -> None:
    last = _last_message(state)
    tool_calls = last.get("tool_calls", [])
    current = _last_user_turn(state)
    current_rounds = current.get("tool_rounds_used", 0) if current else 0
    if current_rounds >= state["runtime"]["max_turns"]:
        requested = ", ".join(call.get("name", "unknown") for call in tool_calls)
        final_answer = "任务因超过最大工具调用轮次而终止，最后一次模型仍请求调用工具：" f"{requested}。"
        _record_terminal_error(
            state,
            "max_turns_exceeded",
            "MaxTurnsExceeded",
            final_answer,
            {"unexecuted_tool_calls": tool_calls},
        )
        state["final_answers"].append({"user_turn_index": current["user_turn_index"] if current else None, "content": final_answer})
        _save_checkpoint(output_dir, state)
        return

    tool_start = perf_counter()
    if state["execution_mode"] == "fixture":
        tool_messages = _fixture_tool_messages(tool_calls, fixture_data["tool_messages"])
    else:
        from b3_tool_layer import execute_tool_calls

        tool_messages = execute_tool_calls(
            tool_calls,
            str(tools_file),
            state["runtime"]["toolset"],
            str(output_dir),
            cache_enabled=bool(state["runtime"].get("tool_cache_enabled", True)),
        )
    tool_latency_ms = round((perf_counter() - tool_start) * 1000, 3)
    state["tool_rounds"] += 1
    if current is not None:
        current["tool_rounds_used"] = current.get("tool_rounds_used", 0) + 1
    state["messages"].extend(tool_messages)
    state["all_tool_messages"].extend(tool_messages)
    if state["turns"]:
        state["turns"][-1]["tool_messages"] = tool_messages
        state["turns"][-1]["tool_latency_ms"] = tool_latency_ms
        state["turns"][-1]["latency_ms"] = round(
            (state["turns"][-1].get("llm_latency_ms") or 0) + tool_latency_ms,
            3,
        )
    _save_checkpoint(output_dir, state)


def _finish_run(
    state: dict,
    output_dir: Path,
    memory_file: Path | None,
    started: float,
) -> dict:
    final_answer = "" if state.get("status") == "needs_user" else _final_answer_text(state["final_answers"])
    if state["status"] == "running":
        state["status"] = "success"
    if state["status"] != "needs_user":
        _clear_pending_question(output_dir)

    write_json(state["messages"], output_dir / "messages.json")
    if state["execution_mode"] == "integrated":
        write_json(state["all_tool_messages"], output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")

    memory_save = state.get("memory_save") or {"requested": state["runtime"]["save_memory"], "status": "not_requested"}
    if state["status"] != "success" and state["runtime"]["save_memory"] != "none":
        memory_save = {"requested": state["runtime"]["save_memory"], "status": "skipped", "reason": state["status"]}

    trace = {
        "conversation_id": state["runtime"]["conversation_id"],
        "execution_mode": state["execution_mode"],
        "agent_mode": state["runtime"].get("agent_mode", "integrated"),
        "status": state["status"],
        "llm_mode": state["llm_mode"],
        "toolset": state["runtime"]["toolset"],
        "max_turns": state["runtime"]["max_turns"],
        "user_turn_count": len(state["user_turns"]),
        "user_turns": state["user_turns"],
        "tool_rounds_used": state["tool_rounds"],
        "llm_call_count": state["llm_calls"],
        "turns": state["turns"],
        "checkpoint_path": CHECKPOINT_FILENAME,
        "final_answer_path": "final_answer.md",
        "memory_save": memory_save,
        "history_compressions": state.get("history_compressions", []),
        "history_compression_count": len(state.get("history_compressions", [])),
        "active_system_prompt": state.get("active_system_prompt"),
        "system_prompt_events_applied": state.get("system_prompt_events_applied", []),
        "conversation_summary_context": state.get("conversation_summary_context", ""),
        "pending_question": state.get("pending_question"),
        "pending_question_path": PENDING_QUESTION_FILENAME if state.get("pending_question") else None,
        "warnings": state["warnings"],
        "error": state["terminal_error"],
    }
    write_json(trace, output_dir / "trace.json")

    saved_memory = None
    if state["execution_mode"] == "integrated" and state["runtime"]["save_memory"] != "none" and trace["status"] == "success":
        try:
            from b5_memory import save_memory

            memory_messages_path = output_dir / "messages.json"
            conversation_history = state["runtime"].get("conversation_history")
            if isinstance(conversation_history, list) and conversation_history:
                memory_messages = [
                    deepcopy(item)
                    for item in conversation_history
                    if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
                ]
                if final_answer and (
                    not memory_messages
                    or memory_messages[-1].get("role") != "assistant"
                    or str(memory_messages[-1].get("content") or "").strip() != final_answer.strip()
                ):
                    memory_messages.append({"role": "assistant", "content": final_answer.strip()})
                memory_messages_path = output_dir / "memory_messages.json"
                write_json(memory_messages, memory_messages_path)

            saved_memory = save_memory(
                str(memory_file),
                state["runtime"]["conversation_id"],
                state["runtime"]["save_memory"],
                str(memory_messages_path),
                str(output_dir / "trace.json"),
                str(output_dir / "final_answer.md"),
                str(output_dir),
            )
            trace["memory_save"] = {"requested": state["runtime"]["save_memory"], "status": "success"}
        except Exception as exc:
            trace["memory_save"] = {
                "requested": state["runtime"]["save_memory"],
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            trace["warnings"].append("memory save failed")
            if trace["status"] == "success":
                trace["status"] = "partial"
                state["status"] = "partial"
        write_json(trace, output_dir / "trace.json")

    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    result = {
        "conversation_id": state["runtime"]["conversation_id"],
        "execution_mode": state["execution_mode"],
        "agent_mode": state["runtime"].get("agent_mode", "integrated"),
        "status": trace["status"],
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "checkpoint_path": str(_checkpoint_path(output_dir)),
        "pending_question_path": str(_pending_question_path(output_dir)) if state.get("pending_question") else None,
        "selected_memory": state["selected_memory"],
        "saved_memory": saved_memory,
        "elapsed_ms": elapsed_ms,
    }
    if state["execution_mode"] == "integrated":
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": state["runtime"]["conversation_id"],
                "execution_mode": state["execution_mode"],
                "agent_mode": state["runtime"].get("agent_mode", "integrated"),
                "status": trace["status"],
                "llm_mode": state["llm_mode"],
                "user_turn_count": len(state["user_turns"]),
                "tool_rounds_used": state["tool_rounds"],
                "llm_call_count": state["llm_calls"],
                "elapsed_ms": elapsed_ms,
            },
            output_dir / "runtime_log.jsonl",
        )

    state["completed"] = trace["status"] != "needs_user"
    state["memory_save"] = trace["memory_save"]
    state["status"] = trace["status"]
    _save_checkpoint(output_dir, state)
    return result


def _uses_specialized_agent_mode(runtime: dict) -> bool:
    return runtime.get("execution_mode") == "integrated" and runtime.get("agent_mode") in {
        "react_one_round",
        "plan_execute",
        "adaptive_execute",
    }


def _specialized_llm_mode(runtime: dict, llm_mode: str | None) -> str | None:
    options = runtime.get("agent_options") if isinstance(runtime.get("agent_options"), dict) else {}
    requested = options.get("llm_mode")
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return llm_mode


def _ensure_specialized_runtime_supported(state: dict) -> None:
    runtime = state["runtime"]
    user_inputs = runtime.get("user_inputs") if isinstance(runtime.get("user_inputs"), list) else []
    if state.get("next_user_index", 0) == 0 and len(user_inputs) != 1:
        raise ValueError("specialized B1 agent_mode requires exactly one initial user_input; later turns should arrive via ASK_USER/resume")


def _prepare_specialized_user_turn(state: dict, output_dir: Path, interactive: bool) -> bool:
    runtime = state["runtime"]
    user_inputs = runtime.get("user_inputs") if isinstance(runtime.get("user_inputs"), list) else []
    next_user_index = int(state.get("next_user_index") or 0)
    if not state.get("user_turns"):
        return _append_next_user_input(state, output_dir, interactive)
    if next_user_index < len(user_inputs):
        return _append_next_user_input(state, output_dir, interactive)
    return True


def _specialized_trace_payload(state: dict, base_trace: dict, memory_save: dict | None = None) -> dict:
    runtime = state["runtime"]
    trace = deepcopy(base_trace) if isinstance(base_trace, dict) else {}
    trace.update(
        {
            "conversation_id": runtime["conversation_id"],
            "execution_mode": state["execution_mode"],
            "agent_mode": runtime["agent_mode"],
            "status": trace.get("status") or state.get("status") or "running",
            "toolset": runtime["toolset"],
            "max_turns": runtime["max_turns"],
            "user_turn_count": len(state.get("user_turns") or []),
            "user_turns": deepcopy(state.get("user_turns") or []),
            "tool_rounds_used": int(trace.get("tool_rounds_used") or state.get("tool_rounds") or 0),
            "llm_call_count": int(trace.get("llm_call_count") or state.get("llm_calls") or 0),
            "checkpoint_path": CHECKPOINT_FILENAME,
            "final_answer_path": "final_answer.md",
            "memory_save": deepcopy(memory_save or state.get("memory_save") or {"requested": runtime["save_memory"], "status": "not_requested"}),
            "history_compressions": deepcopy(state.get("history_compressions") or []),
            "history_compression_count": len(state.get("history_compressions") or []),
            "active_system_prompt": deepcopy(state.get("active_system_prompt")),
            "system_prompt_events_applied": deepcopy(state.get("system_prompt_events_applied") or []),
            "pending_question": deepcopy(state.get("pending_question")),
            "pending_question_path": PENDING_QUESTION_FILENAME if state.get("pending_question") else None,
            "warnings": deepcopy(state.get("warnings") or []),
            "error": deepcopy(state.get("terminal_error")),
        }
    )
    return trace


def _sync_specialized_runtime_snapshot(
    state: dict,
    output_dir: Path,
    snapshot: dict,
    user_turn_index: int,
) -> None:
    if not isinstance(snapshot, dict):
        return
    result_messages = deepcopy(snapshot.get("messages") or state.get("messages") or [])
    result_trace = deepcopy(snapshot.get("trace") or {})
    result_status = str(snapshot.get("status") or result_trace.get("status") or "running")
    final_answer = str(snapshot.get("final_answer") or "")
    pending_question = str(snapshot.get("pending_question") or "").strip()
    all_tool_messages = [message for message in result_messages if isinstance(message, dict) and message.get("role") == "tool"]

    state["messages"] = result_messages
    state["all_tool_messages"] = deepcopy(all_tool_messages)
    state["tool_rounds"] = int(result_trace.get("tool_rounds_used") or state.get("tool_rounds") or 0)
    state["llm_calls"] = int(result_trace.get("llm_call_count") or state.get("llm_calls") or 0)
    state["turns"] = deepcopy(result_trace.get("turns") or state.get("turns") or [])
    state["terminal_error"] = deepcopy(result_trace.get("error"))
    state["status"] = result_status

    current_turn = _last_user_turn(state)
    if isinstance(current_turn, dict):
        current_turn["status"] = result_status
        current_turn["tool_rounds_used"] = state["tool_rounds"]
        current_turn["final_answer"] = final_answer if result_status == "success" else ""
        current_turn["error"] = deepcopy(state["terminal_error"])

    if pending_question:
        normalized = _write_pending_question(output_dir, pending_question)
        state["pending_question"] = {
            "question": normalized,
            "source": "specialized_agent",
            "requested_at": now_iso(),
            "user_turn_index": user_turn_index,
        }
    else:
        state["pending_question"] = None
        _clear_pending_question(output_dir)

    state["final_answers"] = [{"user_turn_index": user_turn_index, "content": final_answer}] if final_answer else []
    write_json(result_messages, output_dir / "messages.json")
    write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(("" if result_status == "needs_user" else final_answer).strip() + "\n", output_dir / "final_answer.md")
    trace = _specialized_trace_payload(state, result_trace)
    write_json(trace, output_dir / "trace.json")
    state["completed"] = False
    _save_checkpoint(output_dir, state)


def _build_specialized_runtime_bridge(state: dict, output_dir: Path, user_turn_index: int) -> dict:
    def _execute_tool_calls_bridge(tool_calls: list[dict], tools_config: str, toolset: str, tool_outdir: str) -> list[dict]:
        from b3_tool_layer import execute_tool_calls

        return execute_tool_calls(tool_calls, tools_config, toolset, tool_outdir)

    def _emit_bridge(snapshot: dict) -> None:
        _sync_specialized_runtime_snapshot(state, output_dir, snapshot, user_turn_index)

    return {
        "execute_tool_calls": _execute_tool_calls_bridge,
        "emit": _emit_bridge,
    }


def _run_specialized_agent_mode(
    state: dict,
    output_dir: Path,
    memory_file: Path | None,
    started: float,
) -> dict:
    runtime = state["runtime"]
    options = runtime.get("agent_options") if isinstance(runtime.get("agent_options"), dict) else {}
    messages = deepcopy(state["messages"])
    current_turn = _last_user_turn(state)
    user_input = str(current_turn.get("user_input") or runtime["user_inputs"][0]) if isinstance(current_turn, dict) else runtime["user_inputs"][0]
    user_turn_index = int(current_turn.get("user_turn_index") or state.get("next_user_index") or 1) if isinstance(current_turn, dict) else 1

    model_config = str(state["model_config"])
    tools_config = str(state["tools_config"])
    toolset = str(runtime["toolset"])
    tools_schema = deepcopy(state["tools_schema"])
    forced_profile = options.get("forced_profile")
    forced_profile = str(forced_profile).strip() if isinstance(forced_profile, str) and forced_profile.strip() else None
    runtime_bridge = _build_specialized_runtime_bridge(state, output_dir, user_turn_index)

    agent_mode = runtime["agent_mode"]
    if agent_mode == "react_one_round":
        exec_result = run_react_one_round_execute(
            model_config,
            messages,
            tools_schema,
            tools_config,
            toolset,
            str(options.get("llm_mode") or state["llm_mode"] or "prompt_json"),
            str(ensure_dir(output_dir)),
            max_tool_rounds=int(options.get("max_tool_rounds") or runtime["max_turns"]),
            forced_profile=forced_profile,
            runtime_bridge=runtime_bridge,
        )
    elif agent_mode == "plan_execute":
        exec_result = run_plan_execute(
            model_config,
            messages,
            tools_schema,
            tools_config,
            toolset,
            str(options.get("llm_mode") or state["llm_mode"] or "prompt_json"),
            str(ensure_dir(output_dir)),
            max_turns=int(runtime["max_turns"]),
            max_plan_steps=int(options.get("max_plan_steps") or 6),
            evidence_policy=str(options.get("evidence_policy") or "strict"),
            forced_profile=forced_profile,
            runtime_bridge=runtime_bridge,
        )
    else:
        exec_result = run_adaptive_execute(
            model_config,
            messages,
            tools_schema,
            tools_config,
            toolset,
            str(ensure_dir(output_dir)),
            max_turns=int(runtime["max_turns"]),
            max_plan_steps=int(options.get("max_plan_steps") or 6),
            evidence_policy=str(options.get("evidence_policy") or "strict"),
            forced_profile=forced_profile,
            low_mode=str(options.get("low_mode") or "native_tools"),
            high_mode=str(options.get("high_mode") or "prompt_json"),
            runtime_bridge=runtime_bridge,
        )

    result_messages = exec_result.get("messages") if isinstance(exec_result.get("messages"), list) else messages
    result_trace = deepcopy(exec_result.get("trace") or {})
    result_status = str(exec_result.get("status") or result_trace.get("status") or "success")
    final_answer = str(exec_result.get("final_answer") or "")
    all_tool_messages = [message for message in result_messages if message.get("role") == "tool"]

    state["messages"] = result_messages
    state["all_tool_messages"] = deepcopy(all_tool_messages)
    state["tool_rounds"] = int(result_trace.get("tool_rounds_used") or 0)
    state["llm_calls"] = int(result_trace.get("llm_call_count") or 0)
    state["turns"] = deepcopy(result_trace.get("turns") or [])
    if isinstance(current_turn, dict):
        current_turn["status"] = "success" if result_status == "success" else result_status
        current_turn["tool_rounds_used"] = int(result_trace.get("tool_rounds_used") or 0)
        current_turn["final_answer"] = final_answer
        current_turn["error"] = result_trace.get("error")
    state["next_user_index"] = max(int(state.get("next_user_index") or 0), user_turn_index)
    if result_status == "needs_user":
        question = str(
            (result_trace.get("error") or {}).get("question")
            or exec_result.get("pending_question")
            or final_answer
            or ""
        ).strip()
        if question:
            _set_pending_question(state, output_dir, question, source="specialized_agent")
    else:
        state["pending_question"] = None
        _clear_pending_question(output_dir)
    state["final_answers"] = [{"user_turn_index": user_turn_index, "content": final_answer}] if final_answer else []
    state["terminal_error"] = deepcopy(result_trace.get("error"))
    state["status"] = result_status

    write_json(result_messages, output_dir / "messages.json")
    write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(("" if result_status == "needs_user" else final_answer).strip() + "\n", output_dir / "final_answer.md")

    memory_save = state.get("memory_save") or {"requested": runtime["save_memory"], "status": "not_requested"}
    if result_status != "success" and runtime["save_memory"] != "none":
        memory_save = {"requested": runtime["save_memory"], "status": "skipped", "reason": result_status}

    trace = deepcopy(result_trace)
    trace.update(
        {
            "conversation_id": runtime["conversation_id"],
            "execution_mode": state["execution_mode"],
            "agent_mode": agent_mode,
            "status": result_status,
            "toolset": runtime["toolset"],
            "max_turns": runtime["max_turns"],
            "user_turn_count": len(state["user_turns"]),
            "user_turns": state["user_turns"],
            "tool_rounds_used": state["tool_rounds"],
            "llm_call_count": state["llm_calls"],
            "checkpoint_path": CHECKPOINT_FILENAME,
            "final_answer_path": "final_answer.md",
            "memory_save": memory_save,
            "history_compressions": state.get("history_compressions", []),
            "history_compression_count": len(state.get("history_compressions", [])),
            "active_system_prompt": state.get("active_system_prompt"),
            "system_prompt_events_applied": state.get("system_prompt_events_applied", []),
            "pending_question": state.get("pending_question"),
            "pending_question_path": PENDING_QUESTION_FILENAME if state.get("pending_question") else None,
            "warnings": state["warnings"],
            "error": state["terminal_error"],
        }
    )
    write_json(trace, output_dir / "trace.json")

    saved_memory = None
    if runtime["save_memory"] != "none" and trace["status"] == "success":
        try:
            from b5_memory import save_memory

            saved_memory = save_memory(
                str(memory_file),
                runtime["conversation_id"],
                runtime["save_memory"],
                str(output_dir / "messages.json"),
                str(output_dir / "trace.json"),
                str(output_dir / "final_answer.md"),
                str(output_dir),
            )
            trace["memory_save"] = {"requested": runtime["save_memory"], "status": "success"}
        except Exception as exc:
            trace["memory_save"] = {
                "requested": runtime["save_memory"],
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            trace["warnings"].append("memory save failed")
            if trace["status"] == "success":
                trace["status"] = "partial"
                state["status"] = "partial"
        write_json(trace, output_dir / "trace.json")

    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    append_jsonl(
        {
            "timestamp": now_iso(),
            "conversation_id": runtime["conversation_id"],
            "execution_mode": state["execution_mode"],
            "agent_mode": agent_mode,
            "status": trace["status"],
            "llm_mode": state["llm_mode"],
            "user_turn_count": len(state["user_turns"]),
            "tool_rounds_used": state["tool_rounds"],
            "llm_call_count": state["llm_calls"],
            "elapsed_ms": elapsed_ms,
        },
        output_dir / "runtime_log.jsonl",
    )

    state["completed"] = trace["status"] != "needs_user"
    state["memory_save"] = trace["memory_save"]
    state["status"] = trace["status"]
    _save_checkpoint(output_dir, state)
    return {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": state["execution_mode"],
        "agent_mode": agent_mode,
        "status": trace["status"],
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "checkpoint_path": str(_checkpoint_path(output_dir)),
        "pending_question_path": str(_pending_question_path(output_dir)) if state.get("pending_question") else None,
        "selected_memory": state["selected_memory"],
        "saved_memory": saved_memory,
        "elapsed_ms": elapsed_ms,
    }


def _return_completed_result(state: dict, output_dir: Path) -> dict:
    final_answer_path = output_dir / "final_answer.md"
    final_answer = ""
    if state.get("status") != "needs_user":
        final_answer = read_text(final_answer_path).strip() if final_answer_path.exists() else _final_answer_text(state.get("final_answers", []))
    return {
        "conversation_id": state["runtime"]["conversation_id"],
        "execution_mode": state["execution_mode"],
        "agent_mode": state["runtime"].get("agent_mode", "integrated"),
        "status": state.get("status", "success"),
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(final_answer_path),
        "checkpoint_path": str(_checkpoint_path(output_dir)),
        "pending_question_path": str(_pending_question_path(output_dir)) if state.get("pending_question") else None,
        "selected_memory": state.get("selected_memory"),
        "saved_memory": None,
        "elapsed_ms": 0.0,
    }


def run_agent(
    input_path: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
    resume: bool = False,
    interactive: bool = False,
) -> dict:
    started = perf_counter()
    input_file = Path(input_path).resolve()
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = None
    fixture_data = tools_file = memory_file = model_file = None

    if resume:
        state = _load_checkpoint(output_dir)
        _merge_runtime_input_for_resume(state, input_file)
        input_file = Path(state.get("input_path") or input_file).resolve()
        if state.get("completed") and state.get("status") != "needs_user" and not interactive:
            return _return_completed_result(state, output_dir)
        fixture_data, tools_file, memory_file, model_file = _restore_runtime_handles(state, input_file)
        state["completed"] = False
        if state.get("status") in {"success", "needs_user", "stopped"}:
            state["status"] = "running"
        else:
            state["status"] = state.get("status", "running")
    else:
        runtime = _validate_runtime_input(read_json(input_file))
        state, fixture_data, tools_file, memory_file, model_file = _build_initial_state(
            input_file,
            runtime,
            output_dir,
            tools_config,
            memory_config,
            model_config,
            llm_mode,
        )
        _save_checkpoint(output_dir, state)

    if _uses_specialized_agent_mode(state["runtime"]):
        _ensure_specialized_runtime_supported(state)
        if not _prepare_specialized_user_turn(state, output_dir, interactive):
            return _finish_run(state, output_dir, memory_file, started)
        return _run_specialized_agent_mode(state, output_dir, memory_file, started)

    while state.get("status") == "running":
        action = _current_pending_action(state)
        if action == "need_user":
            if not _append_next_user_input(state, output_dir, interactive):
                break
        elif action == "need_llm":
            _run_llm_step(state, output_dir, fixture_data, model_file)
        elif action == "need_tools":
            _run_tool_step(state, output_dir, fixture_data, tools_file)
        else:
            raise ValueError(f"unsupported pending action: {action}")

    return _finish_run(state, output_dir, memory_file, started)


def _safe_task_id(value: str, index: int) -> str:
    raw = value.strip() if isinstance(value, str) and value.strip() else f"task_{index:03d}"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return cleaned or f"task_{index:03d}"


def _resolve_optional_batch_path(value: str | None, batch_file: Path) -> str | None:
    if not value:
        return None
    return str(resolve_from_file(value, batch_file))


def _looks_like_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}


def _resolve_embedded_runtime_path(value: str, batch_file: Path) -> str:
    resolved = resolve_from_file(value, batch_file)
    if resolved.exists() or not _looks_like_windows_absolute_path(value):
        return str(resolved)

    project_root = batch_file.parent.parent
    normalized = value.replace("\\", "/")
    for marker in ("prompts/", "data/", "memory/", "configs/"):
        marker_index = normalized.find("/" + marker)
        if marker_index >= 0:
            portable_tail = normalized[marker_index + 1 :]
            return str((project_root / portable_tail).resolve())
    return str(resolved)


def _prepare_embedded_runtime_input(runtime_input: dict, batch_file: Path) -> dict:
    payload = deepcopy(runtime_input)
    if isinstance(payload.get("system_prompt_path"), str):
        payload["system_prompt_path"] = _resolve_embedded_runtime_path(payload["system_prompt_path"], batch_file)

    events = payload.get("system_prompt_events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("system_prompt_path"), str):
                event["system_prompt_path"] = _resolve_embedded_runtime_path(event["system_prompt_path"], batch_file)

    fixtures = payload.get("fixtures")
    if isinstance(fixtures, dict):
        for key, value in list(fixtures.items()):
            if key.endswith("_path") and isinstance(value, str):
                fixtures[key] = _resolve_embedded_runtime_path(value, batch_file)
    return payload

def _load_batch_tasks(batch_input: str | Path) -> tuple[Path, dict, list[dict]]:
    batch_file = Path(batch_input).resolve()
    payload = read_json(batch_file)
    if isinstance(payload, list):
        payload = {"tasks": payload}
    if not isinstance(payload, dict):
        raise ValueError("batch input must be an object or task array")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("batch input must contain a non-empty tasks array")
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f"batch task {index} must be an object")
        if not isinstance(task.get("input_path"), str) and not isinstance(task.get("runtime_input"), dict):
            raise ValueError(f"batch task {index} requires input_path or runtime_input")
    return batch_file, payload, tasks


def _batch_report_markdown(batch_result: dict) -> str:
    lines = [
        "# B1 Batch Run Report",
        "",
        f"- Status: `{batch_result['status']}`",
        f"- Total tasks: `{batch_result['total_tasks']}`",
        f"- Success: `{batch_result['success_count']}`",
        f"- Error: `{batch_result['error_count']}`",
        "",
        "## Tasks",
        "",
    ]
    for item in batch_result["tasks"]:
        lines.append(f"### {item['task_id']}")
        lines.append("")
        lines.append(f"- Status: `{item['status']}`")
        lines.append(f"- Output dir: `{item['outdir']}`")
        if item.get("final_answer_path"):
            lines.append(f"- Final answer: `{item['final_answer_path']}`")
        if item.get("error"):
            error = item["error"]
            lines.append(f"- Error: `{error.get('type')}: {error.get('message')}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_batch_tasks(
    batch_input: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
) -> dict:
    started = perf_counter()
    batch_file, payload, tasks = _load_batch_tasks(batch_input)
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = payload.get("common") if isinstance(payload.get("common"), dict) else {}
    task_results = []
    for index, task in enumerate(tasks, 1):
        task_id = _safe_task_id(str(task.get("task_id") or task.get("id") or ""), index)
        task_outdir_value = task.get("outdir") or task_id
        task_outdir = Path(task_outdir_value)
        if not task_outdir.is_absolute():
            task_outdir = output_dir / task_outdir
        task_outdir.mkdir(parents=True, exist_ok=True)
        if isinstance(task.get("runtime_input"), dict):
            task_input_path = task_outdir / "runtime_input.json"
            runtime_payload = _prepare_embedded_runtime_input(task["runtime_input"], batch_file)
            write_json(runtime_payload, task_input_path)
        else:
            task_input_path = resolve_from_file(task["input_path"], batch_file)
        task_tools_config = _resolve_optional_batch_path(task.get("tools_config") or common.get("tools_config"), batch_file) or tools_config
        task_memory_config = _resolve_optional_batch_path(task.get("memory_config") or common.get("memory_config"), batch_file) or memory_config
        task_model_config = _resolve_optional_batch_path(task.get("model_config") or common.get("model_config"), batch_file) or model_config
        task_llm_mode = task.get("llm_mode") or common.get("llm_mode") or llm_mode
        try:
            result = run_agent(
                str(task_input_path),
                task_tools_config,
                task_memory_config,
                task_model_config,
                str(task_outdir),
                task_llm_mode,
                bool(task.get("resume", False)),
                False,
            )
            task_results.append(
                {
                    "task_id": task_id,
                    "status": result["status"],
                    "input_path": str(task_input_path),
                    "outdir": str(task_outdir),
                    "final_answer_path": result.get("final_answer_path"),
                    "trace_path": result.get("trace_path"),
                    "messages_path": result.get("messages_path"),
                    "elapsed_ms": result.get("elapsed_ms"),
                    "error": None,
                }
            )
        except Exception as exc:
            task_results.append(
                {
                    "task_id": task_id,
                    "status": "error",
                    "input_path": str(task_input_path),
                    "outdir": str(task_outdir),
                    "final_answer_path": None,
                    "trace_path": None,
                    "messages_path": None,
                    "elapsed_ms": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    success_count = sum(1 for item in task_results if item["status"] == "success")
    error_count = len(task_results) - success_count
    batch_result = {
        "batch_input": str(batch_file),
        "status": "success" if error_count == 0 else "partial",
        "total_tasks": len(task_results),
        "success_count": success_count,
        "error_count": error_count,
        "tasks": task_results,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    write_json(batch_result, output_dir / "batch_results.json")
    write_text(_batch_report_markdown(batch_result), output_dir / "batch_report.md")
    append_jsonl(
        {
            "timestamp": now_iso(),
            "batch_input": str(batch_file),
            "status": batch_result["status"],
            "total_tasks": batch_result["total_tasks"],
            "success_count": success_count,
            "error_count": error_count,
            "elapsed_ms": batch_result["elapsed_ms"],
        },
        output_dir / "batch_runtime_log.jsonl",
    )
    return batch_result


def _case_messages_to_runtime_input(
    case: dict,
    prompt_path: Path,
    cases_file: Path,
    agent_mode: str,
    toolset: str,
    llm_mode: str | None,
    max_turns: int,
    max_plan_steps: int,
    evidence_policy: str,
    save_memory: str,
    tools_schema_source: str,
    tools_schema_path: str | None,
    forced_profile: str | None,
) -> dict:
    messages = case.get("messages") if isinstance(case.get("messages"), list) else []
    system_chunks = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system" and str(message.get("content") or "").strip()
    ]
    user_inputs = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    if not system_chunks:
        raise ValueError(f"case {case.get('id')}: at least one system message is required")
    if not user_inputs:
        raise ValueError(f"case {case.get('id')}: at least one user message is required")
    if agent_mode in {"react_one_round", "plan_execute", "adaptive_execute"} and len(user_inputs) != 1:
        raise ValueError(f"case {case.get('id')}: agent_mode={agent_mode} currently requires exactly one user message")

    runtime_input = {
        "conversation_id": f"eval_{case['id']}",
        "execution_mode": "integrated",
        "agent_mode": agent_mode,
        "agent_options": {},
        "system_prompt_path": str(prompt_path),
        "user_inputs": user_inputs,
        "selected_memory_ids": [],
        "use_global_memory": False,
        "toolset": toolset,
        "max_turns": int(max_turns),
        "save_memory": save_memory,
        "tools_schema_source": tools_schema_source,
    }
    if tools_schema_source == "static" and tools_schema_path:
        runtime_input["tools_schema_path"] = str(resolve_from_file(tools_schema_path, cases_file))
    if llm_mode:
        runtime_input["agent_options"]["llm_mode"] = llm_mode
    if forced_profile:
        runtime_input["agent_options"]["forced_profile"] = forced_profile
    if agent_mode in {"plan_execute", "adaptive_execute"}:
        runtime_input["agent_options"]["max_plan_steps"] = int(max_plan_steps)
        runtime_input["agent_options"]["evidence_policy"] = str(evidence_policy)
    if agent_mode == "adaptive_execute":
        runtime_input["agent_options"]["low_mode"] = "native_tools"
        runtime_input["agent_options"]["high_mode"] = "prompt_json"
    if agent_mode == "react_one_round":
        runtime_input["agent_options"]["max_tool_rounds"] = int(max_turns)
    return runtime_input


def _build_eval_batch_payload(
    cases: list[dict],
    cases_file: Path,
    output_dir: Path,
    agent_mode: str,
    toolset: str,
    llm_mode: str | None,
    max_turns: int,
    max_plan_steps: int,
    evidence_policy: str,
    save_memory: str,
    tools_schema_source: str,
    tools_schema_path: str | None,
    forced_profile: str | None,
) -> dict:
    tasks: list[dict] = []
    for case in cases:
        case_outdir = output_dir / "artifacts" / agent_mode / case["id"]
        case_outdir.mkdir(parents=True, exist_ok=True)
        system_prompt_text = "\n\n".join(
            str(message.get("content") or "").strip()
            for message in case["messages"]
            if isinstance(message, dict) and message.get("role") == "system" and str(message.get("content") or "").strip()
        ).strip()
        prompt_path = case_outdir / "system_prompt.txt"
        write_text(system_prompt_text + "\n", prompt_path)
        runtime_input = _case_messages_to_runtime_input(
            case,
            prompt_path,
            cases_file,
            agent_mode,
            toolset,
            llm_mode,
            max_turns,
            max_plan_steps,
            evidence_policy,
            save_memory,
            tools_schema_source,
            tools_schema_path,
            forced_profile,
        )
        tasks.append(
            {
                "task_id": case["id"],
                "outdir": str(case_outdir),
                "runtime_input": runtime_input,
            }
        )
    return {"tasks": tasks}


def _batch_eval_markdown(summary_payload: dict, rows: list[dict]) -> str:
    summary = summary_payload.get("summary") if isinstance(summary_payload.get("summary"), dict) else {}
    lines = [
        "# B1 End-to-End Eval Report",
        "",
        f"- Agent mode: `{summary_payload.get('agent_mode')}`",
        f"- Cases path: `{summary_payload.get('cases_path')}`",
        f"- Total rows: `{len(rows)}`",
        "",
        "## Summary",
        "",
    ]
    for key, item in summary.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- Cases: `{item.get('cases')}`")
        lines.append(f"- Success rate: `{item.get('success_rate')}`")
        lines.append(f"- Structured output rate: `{item.get('structured_output_rate')}`")
        lines.append(f"- Tool match rate: `{item.get('tool_match_rate')}`")
        lines.append(f"- Args complete rate: `{item.get('args_complete_rate')}`")
        lines.append("")
    lines.append("## Cases")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.get('case_id')}")
        lines.append("")
        lines.append(f"- Title: `{row.get('title')}`")
        lines.append(f"- Expected status: `{row.get('expected_status')}`")
        lines.append(f"- Actual status: `{row.get('actual_status')}`")
        lines.append(f"- Success: `{row.get('success')}`")
        lines.append(f"- Expected tools: `{row.get('expected_tools')}`")
        lines.append(f"- Actual tools: `{row.get('actual_tools')}`")
        if row.get("error_type") or row.get("error_message"):
            lines.append(f"- Error: `{row.get('error_type')}: {row.get('error_message')}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_eval_setting(explicit_value, file_value, fallback_value):
    if explicit_value is not None:
        return explicit_value
    if file_value is not None:
        return file_value
    return fallback_value


def _load_eval_tools_schema(
    *,
    cases_file: Path,
    tools_config: str,
    toolset: str,
    output_dir: Path,
    tools_schema_source: str,
    tools_schema_path: str | None,
) -> tuple[list[dict], str | None]:
    if tools_schema_source == "static":
        if not tools_schema_path:
            raise ValueError("tools_schema_source=static requires tools_schema_path")
        resolved_path = resolve_from_file(tools_schema_path, cases_file)
        tools_schema = read_json(resolved_path)
        if not isinstance(tools_schema, list):
            raise ValueError("tools_schema_path must point to a JSON array")
        snapshot_dir = ensure_dir(output_dir / "_schema_snapshot")
        write_json(tools_schema, snapshot_dir / "tools_schema.json")
        write_json(
            {
                "status": "success",
                "toolset": toolset,
                "tool_count": len(tools_schema),
                "schema_source": "static_file",
                "tools_schema_path": str(resolved_path),
            },
            snapshot_dir / "tool_schema_report.json",
        )
        return tools_schema, str(resolved_path)
    from b3_tool_layer import get_tools_schema

    tools_schema = get_tools_schema(str(Path(tools_config).resolve()), toolset, str(output_dir / "_schema_snapshot"))
    return tools_schema, None


def run_batch_eval_cases(
    cases_path: str,
    tools_config: str,
    memory_config: str,
    model_config: str,
    outdir: str,
    *,
    agent_mode: str | None = None,
    toolset: str | None = None,
    llm_mode: str | None = None,
    max_turns: int | None = None,
    max_plan_steps: int | None = None,
    evidence_policy: str | None = None,
    save_memory: str | None = None,
) -> dict:
    from b4_local_agent_llm import (
        _aggregate_llm_usage,
        _evaluate_plan_execute_trace,
        _load_eval_cases,
        _load_eval_cases_payload,
        _normalize_name_list,
        _optional_internal_tools_satisfied,
        _question_matches_expectation,
        _summarize_eval_rows,
        _tool_match,
        _write_eval_report_csv,
    )

    cases_file = Path(cases_path).resolve()
    payload = _load_eval_cases_payload(cases_file)
    batch_eval = payload.get("batch_eval") if isinstance(payload.get("batch_eval"), dict) else {}
    agent_mode = str(_resolve_eval_setting(agent_mode, batch_eval.get("agent_mode"), "adaptive_execute"))
    toolset = str(_resolve_eval_setting(toolset, batch_eval.get("toolset"), "basic_tools"))
    llm_mode = _resolve_eval_setting(llm_mode, batch_eval.get("llm_mode"), None)
    max_turns = int(_resolve_eval_setting(max_turns, batch_eval.get("max_turns"), 3))
    max_plan_steps = int(_resolve_eval_setting(max_plan_steps, batch_eval.get("max_plan_steps"), 6))
    evidence_policy = str(_resolve_eval_setting(evidence_policy, batch_eval.get("evidence_policy"), "lite"))
    save_memory = str(_resolve_eval_setting(save_memory, batch_eval.get("save_memory"), "none"))
    tools_schema_source = str(_resolve_eval_setting(None, batch_eval.get("tools_schema_source"), "dynamic"))
    tools_schema_path = _resolve_eval_setting(None, batch_eval.get("tools_schema_path"), None)
    forced_profile = _resolve_eval_setting(None, batch_eval.get("forced_profile"), None)

    if agent_mode not in SUPPORTED_AGENT_MODES:
        raise ValueError(f"agent_mode must be one of: {', '.join(sorted(SUPPORTED_AGENT_MODES))}")
    if save_memory not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")
    if tools_schema_source not in {"dynamic", "static"}:
        raise ValueError("tools_schema_source must be dynamic or static")

    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_eval_cases(cases_file)
    batch_payload = _build_eval_batch_payload(
        cases,
        cases_file,
        output_dir,
        agent_mode,
        toolset,
        llm_mode,
        max_turns,
        max_plan_steps,
        evidence_policy,
        save_memory,
        tools_schema_source,
        tools_schema_path,
        forced_profile,
    )
    batch_input_path = output_dir / "generated_batch_input.json"
    write_json(batch_payload, batch_input_path)
    tools_schema, resolved_tools_schema_path = _load_eval_tools_schema(
        cases_file=cases_file,
        tools_config=tools_config,
        toolset=toolset,
        output_dir=output_dir,
        tools_schema_source=tools_schema_source,
        tools_schema_path=tools_schema_path,
    )
    batch_result = run_batch_tasks(
        str(batch_input_path),
        str(Path(tools_config).resolve()),
        str(Path(memory_config).resolve()),
        str(Path(model_config).resolve()),
        str(output_dir / "batch_run"),
        llm_mode,
    )

    task_results_by_id = {str(item.get("task_id")): item for item in batch_result.get("tasks", []) if isinstance(item, dict)}
    rows: list[dict] = []
    for case in cases:
        task_result = task_results_by_id.get(case["id"], {})
        trace = {}
        error = task_result.get("error") if isinstance(task_result.get("error"), dict) else {}
        trace_path = task_result.get("trace_path")
        if isinstance(trace_path, str) and trace_path:
            trace_file = Path(trace_path)
            if trace_file.exists():
                maybe_trace = read_json(trace_file)
                if isinstance(maybe_trace, dict):
                    trace = maybe_trace
                    if isinstance(trace.get("error"), dict):
                        error = trace.get("error")
        actual_status = str(task_result.get("status") or trace.get("status") or "error")
        expected_status = str(case.get("expected_status") or "success").strip() or "success"
        structured_ok = actual_status == expected_status
        eval_detail = _evaluate_plan_execute_trace(trace, tools_schema)
        actual_tools = eval_detail.get("actual_tools") if isinstance(eval_detail.get("actual_tools"), list) else []
        expected_tools = _normalize_name_list(case.get("expected_tools", []))
        raw_match_type = str(case.get("expected_match_type") or "").strip()
        match_type = raw_match_type or ("exact" if not expected_tools else "contains")
        tool_match = _tool_match(
            expected_tools,
            actual_tools,
            bool(case.get("tool_order_matters")),
            match_type,
        )
        if not tool_match and _optional_internal_tools_satisfied(expected_tools, actual_tools, evidence_policy):
            tool_match = True
        args_complete = bool(eval_detail.get("args_complete"))
        question_ok = _question_matches_expectation(trace, str(case.get("expected_question_contains") or ""))
        if expected_status == "needs_user":
            tool_match = True
            args_complete = True
        structured_ok = structured_ok and question_ok
        success = structured_ok and tool_match and args_complete
        usage = _aggregate_llm_usage(Path(task_result.get("outdir") or "") / "llm_calls")
        rows.append(
            {
                "case_id": case["id"],
                "title": case["title"],
                "category": case["category"],
                "mode": agent_mode,
                "model_profile": "b1_e2e",
                "model_name": Path(model_config).name,
                "model_target": str(Path(model_config).resolve()),
                "expected_status": expected_status,
                "actual_status": actual_status,
                "expected_tools": "|".join(expected_tools),
                "actual_tools": "|".join(actual_tools),
                "tool_match": tool_match,
                "structured_output_success": structured_ok,
                "args_complete": args_complete,
                "success": success,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "elapsed_seconds": usage.get("elapsed_seconds"),
                "error_type": error.get("type", "") if isinstance(error, dict) else "",
                "error_message": error.get("message", "") if isinstance(error, dict) else "",
            }
        )

    summary = _summarize_eval_rows(rows)
    summary_payload = {
        "cases_path": str(Path(cases_path).resolve()),
        "agent_mode": agent_mode,
        "llm_mode": llm_mode,
        "toolset": toolset,
        "tools_schema_source": tools_schema_source,
        "tools_schema_path": resolved_tools_schema_path,
        "forced_profile": forced_profile,
        "summary": summary,
        "batch_result_path": str(output_dir / "batch_run" / "batch_results.json"),
        "evaluation_mode": "b1_end_to_end",
    }
    write_json(summary_payload, output_dir / "eval_summary.json")
    _write_eval_report_csv(rows, output_dir / "eval_report.csv")
    write_text(_batch_eval_markdown(summary_payload, rows), output_dir / "eval_report.md")
    return {"rows": rows, "summary": summary, "outdir": str(output_dir), "batch_result": batch_result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Agent message and tool loop.")
    parser.add_argument("--input")
    parser.add_argument("--batch_input", help="JSON file containing multiple B1 runtime tasks")
    parser.add_argument("--eval_cases", help="JSON file containing B1 end-to-end evaluation cases")
    parser.add_argument("--tools_config")
    parser.add_argument("--memory_config")
    parser.add_argument("--model_config")
    parser.add_argument("--agent_mode", choices=sorted(SUPPORTED_AGENT_MODES), default=None)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--llm_mode", choices=sorted(SUPPORTED_LLM_MODES), default=None)
    parser.add_argument("--save_memory", choices=["none", "conversation", "global"], default=None)
    parser.add_argument("--max_turns", type=int, default=None)
    parser.add_argument("--max_plan_steps", type=int, default=None)
    parser.add_argument("--evidence_policy", choices=["strict", "lite"], default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true", help="resume from outdir/checkpoint.json")
    parser.add_argument("--interactive", action="store_true", help="continue asking for user input after each answer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.eval_cases:
            if not args.tools_config or not args.memory_config or not args.model_config:
                raise ValueError("--tools_config, --memory_config and --model_config are required with --eval_cases")
            run_batch_eval_cases(
                str(resolve_cli_path(args.eval_cases)),
                str(resolve_cli_path(args.tools_config)),
                str(resolve_cli_path(args.memory_config)),
                str(resolve_cli_path(args.model_config)),
                str(resolve_cli_path(args.outdir)),
                agent_mode=args.agent_mode,
                toolset=args.toolset,
                llm_mode=args.llm_mode,
                max_turns=args.max_turns,
                max_plan_steps=args.max_plan_steps,
                evidence_policy=args.evidence_policy,
                save_memory=args.save_memory,
            )
            print(Path(resolve_cli_path(args.outdir)) / "eval_report.csv")
        elif args.batch_input:
            result = run_batch_tasks(
                str(resolve_cli_path(args.batch_input)),
                str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
                str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
                str(resolve_cli_path(args.model_config)) if args.model_config else None,
                str(resolve_cli_path(args.outdir)),
                args.llm_mode,
            )
            print(Path(resolve_cli_path(args.outdir)) / "batch_report.md")
        else:
            if not args.input:
                raise ValueError("--input is required unless --batch_input is provided")
            result = run_agent(
                str(resolve_cli_path(args.input)),
                str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
                str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
                str(resolve_cli_path(args.model_config)) if args.model_config else None,
                str(resolve_cli_path(args.outdir)),
                args.llm_mode,
                args.resume,
                args.interactive,
            )
            print(result["final_answer_path"])
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
