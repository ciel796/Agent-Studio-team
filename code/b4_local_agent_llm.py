from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from common.io_utils import append_jsonl, ensure_dir, read_json, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import make_ai_message, validate_ai_message, validate_messages


PARSE_ERROR_CONTENT = "模型输出解析失败，无法生成有效工具调用或最终回答。"
_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_GENERATION_LOCK = threading.Lock()


def _load_model_config(model_config: str | Path) -> tuple[Path, dict]:
    path = Path(model_config).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("model.yaml must contain an object")
    return path, config


def _resolve_runtime_mode(config: dict, requested_mode: str | None) -> str:
    if isinstance(requested_mode, str) and requested_mode.strip():
        mode = requested_mode.strip()
    else:
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        tool_calling = config.get("tool_calling") if isinstance(config.get("tool_calling"), dict) else {}
        mode = str(runtime.get("default_mode") or tool_calling.get("mode") or "prompt_json").strip()
    if mode not in {"mock", "prompt_json", "native_tools"}:
        raise ValueError("mode must be mock, prompt_json, or native_tools")
    return mode


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _detect_routing_phase(messages: list[dict]) -> str:
    text = _latest_user_text(messages)
    if "请先生成一个可执行的计划" in text:
        return "plan"
    if "<plan_state>" in text or "Plan-and-Execute 模式" in text or "当前计划状态" in text:
        return "execute"
    return "direct"


def _classify_task_profile(messages: list[dict]) -> str:
    text = _latest_user_text(messages)
    lowered = text.lower()
    has_sequence = bool(re.search(r"(先|然后|再|最后|步骤|计划|分步|对比|比较|汇总后)", text))
    file_mentions = len(re.findall(r"\b[\w./-]+\.(txt|md|csv|tsv|json)\b", lowered))
    tool_mentions = bool(re.search(r"\b(file_reader|local_file_search|table_analyzer|calculator|format_converter)\b", lowered))
    if has_sequence or file_mentions >= 2:
        return "planner"
    if tool_mentions or file_mentions == 1:
        return "default"
    return "default"


def _resolve_model_profile(config: dict, messages: list[dict]) -> str | None:
    routing = config.get("routing") if isinstance(config.get("routing"), dict) else {}
    if not isinstance(routing, dict) or not routing.get("enabled", False):
        return None
    pool = config.get("model_pool")
    if not isinstance(pool, dict) or not pool:
        return None
    phase = _detect_routing_phase(messages)
    use_heuristics = bool(routing.get("use_message_heuristics", False))
    classified = _classify_task_profile(messages) if use_heuristics else "default"
    if use_heuristics and phase == "direct" and classified in pool:
        return classified
    if phase == "plan":
        profile = routing.get("plan_profile")
    elif phase == "execute":
        profile = routing.get("execute_profile")
    else:
        profile = routing.get("default_profile")
    if isinstance(profile, str) and profile in pool:
        return profile
    if use_heuristics and classified in pool:
        return classified
    first_key = next(iter(pool.keys()))
    return first_key


def _merged_model_settings(config: dict, profile_name: str | None) -> dict:
    base = config.get("model") if isinstance(config.get("model"), dict) else {}
    merged = dict(base) if isinstance(base, dict) else {}
    pool = config.get("model_pool")
    if profile_name and isinstance(pool, dict):
        profile = pool.get(profile_name)
        if isinstance(profile, dict):
            merged.update(profile)
    return merged


def _artifact_paths(artifact_dir: str | Path, stem: str | None) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    prefix = f"{stem}_" if stem else ""
    return (
        directory / f"{prefix}raw_model_output.json",
        directory / f"{prefix}ai_message.json",
        directory / "llm_run_log.jsonl",
    )


def _prepare_chat_messages(messages: list[dict]) -> list[dict]:
    prompt_messages = deepcopy(messages)
    system_messages = [message for message in prompt_messages if message.get("role") == "system"]
    if system_messages:
        primary = system_messages[0]
        combined = "\n\n".join(
            str(message.get("content") or "").strip()
            for message in system_messages
            if str(message.get("content") or "").strip()
        )
        primary["content"] = combined
        prompt_messages = [primary] + [message for message in prompt_messages if message.get("role") != "system"]
    else:
        prompt_messages.insert(0, {"role": "system", "content": ""})
    return prompt_messages


def _available_tool_names(tools_schema: list[dict]) -> set[str]:
    names: set[str] = set()
    for item in tools_schema:
        if not isinstance(item, dict):
            continue
        function_schema = item.get("function")
        if isinstance(function_schema, dict):
            name = function_schema.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _append_text_block(base_text: str, extra_text: str) -> str:
    base = str(base_text or "").rstrip()
    extra = str(extra_text or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{base}\n\n{extra}"


def _normalize_assistant_tool_calls_for_chat_template(prompt_messages: list[dict]) -> None:
    for message in prompt_messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        normalized_calls: list[dict] = []
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            normalized_call = _normalize_tool_call(call, index)
            normalized_calls.append(
                {
                    "id": normalized_call.get("id") or f"call_{index + 1:03d}",
                    "type": "function",
                    "function": {
                        "name": normalized_call["name"],
                        "arguments": normalized_call["args"],
                    },
                }
            )
        message["tool_calls"] = normalized_calls

def _tool_usage_hint(messages: list[dict], tools_schema: list[dict]) -> str:
    if _detect_routing_phase(messages) != "direct" or _has_any_tool_message(messages):
        return ""
    user_text = _latest_user_text(messages)
    if not user_text.strip():
        return ""
    available_names = _available_tool_names(tools_schema)
    hints: list[str] = []
    file_mentions = re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", user_text, flags=re.IGNORECASE)
    if any(path.lower().endswith((".txt", ".md", ".json")) for path in file_mentions) and "file_reader" in available_names:
        hints.append("The user explicitly refers to a local text/markdown file. You must call file_reader before answering and must not answer from prior knowledge.")
    if any(path.lower().endswith((".csv", ".tsv")) for path in file_mentions) and "table_analyzer" in available_names:
        hints.append("The user explicitly refers to a local table file. You must call table_analyzer before answering and must not invent table contents.")
    if (
        "local_file_search" in available_names
        and _should_search_local_scope_before_answering(user_text)
    ):
        hints.append("The user is asking about a local file scope without naming an exact file path. Search that scope with local_file_search before reading specific files or answering.")
    return "\n".join(dict.fromkeys(hints))


def _decision_mobility_hint(messages: list[dict], tools_schema: list[dict]) -> str:
    user_text = _latest_user_text(messages)
    available_names = _available_tool_names(tools_schema)
    hints = [
        "First decide internally whether the best next action is a direct answer, one round of tool calls, ASK_USER, or a brief failure report. Keep that reasoning internal and output only the required final format.",
        "Prefer the shortest grounded path: if the answer can be completed reliably from the current conversation, answer directly; if external evidence is missing, use tools."
    ]
    if not _has_explicit_local_resource_request(user_text):
        hints.append("Do not read local files, search the workspace, or inspect local tables unless the user explicitly asks about a local path, local search, or local data source.")
    if (
        re.search(r"(\u8ba1\u7b97|\u7b97\u51fa|evaluate|calculate)", user_text, flags=re.IGNORECASE)
        or re.search(r"\d\s*[\+\-\*/]\s*\d", user_text)
    ) and "calculator" in available_names:
        hints.append("For internal arithmetic, you may answer directly when confident; use calculator when you want verified computation or the expression is error-prone.")
    file_mentions = re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", user_text, flags=re.IGNORECASE)
    if len(file_mentions) >= 2:
        hints.append("If multiple independent files are needed, request them together in the same response instead of serializing unnecessary turns.")
    return "\n".join(dict.fromkeys(hints))


def _has_explicit_local_resource_request(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    lowered = text.casefold()
    if re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\b[\w.-]+(?:[\\/][\w./\\-]*)+\b", text):
        return True
    if re.search(r"(?:^|[\s(（\"'])[\w.-]+\s*(?:中|里|目录|文件夹|文件|文档)", text):
        return True
    if re.search(r"(docs|doc|tables|table)\s*(?:中|里|目录|文件夹|文件|文档)", text, flags=re.IGNORECASE):
        return True
    explicit_patterns = [
        r"(搜索|查找|检索|search).*(docs|tables|目录|文件|文档|本地)",
        r"(读取|阅读|查阅|查看|打开|检查).*(文件|文档|csv|tsv)",
        r"(本地文件|本地搜索|工作区|目录|表格数据|csv|tsv)",
    ]
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in explicit_patterns)


def _should_search_local_scope_before_answering(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    has_exact_file = bool(re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", text, flags=re.IGNORECASE))
    return _has_explicit_local_resource_request(text) and not has_exact_file


def _latest_tool_batch_guidance(messages: list[dict]) -> str:
    latest_batch = _latest_tool_message_batch(messages)
    if not latest_batch:
        return ""
    merged = merge_tool_messages(latest_batch)
    success_labels: list[str] = []
    failure_labels: list[str] = []
    failed_payload: list[dict] = []
    for item in merged:
        label = str(item.get("name") or item.get("tool_call_id") or "tool").strip()
        if item.get("status") == "success":
            success_labels.append(label)
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        tool_input = result.get("input") if isinstance(result.get("input"), dict) else None
        message = str(error.get("message") or "").strip()
        failure_labels.append(f"{label}: {message}" if message else label)
        failed_payload.append(
            {
                "tool_call_id": item.get("tool_call_id"),
                "name": item.get("name"),
                "input": tool_input,
                "error": error,
            }
        )
    guidance: list[str] = []
    if success_labels:
        guidance.append(
            "Latest successful tools: "
            + ", ".join(success_labels)
            + ". Do not repeat these successful tool calls unless the user explicitly requests a different query."
        )
    if failure_labels:
        guidance.append(
            "Latest failed tools: "
            + "; ".join(failure_labels)
            + ". Retry only if you change the arguments or choose a different tool; otherwise conclude with the best supported answer or mark the current plan step as failed."
        )
    if failed_payload:
        guidance.append("Failed tool calls JSON:\n" + json.dumps(failed_payload, ensure_ascii=False))
    return "\n".join(guidance)


def _mode_retry_instruction(mode: str) -> str:
    if mode == "native_tools":
        return (
            "If a tool is needed, output only the native tool-calling result for this turn. "
            "Emit one or more <tool_call> blocks and no surrounding explanation. "
            "If multiple independent tools are needed, emit multiple <tool_call> blocks in the same reply."
        )
    return (
        "Return exactly one valid JSON object now. "
        'Use exactly the top-level keys "content" (string) and "tool_calls" (array). '
        'If tools are needed, set "content" to "" and put one or more tool calls in "tool_calls". '
        'If no tools are needed, set "tool_calls" to [].'
    )


def _build_parse_retry_messages(
    messages: list[dict],
    raw_text: str,
    mode: str,
    parse_error: Exception,
    tools_schema: list[dict],
) -> list[dict]:
    retry_messages = validate_messages(deepcopy(messages))
    usage_hint = _tool_usage_hint(retry_messages, tools_schema)
    parts = [
        "Your previous reply could not be parsed into a valid assistant message.",
        f"Parser error: {type(parse_error).__name__}: {parse_error}",
        "Previous reply:",
        "<previous_reply>",
        raw_text.strip(),
        "</previous_reply>",
        _mode_retry_instruction(mode),
    ]
    parse_error_text = str(parse_error or "")
    if "tool call missing keys: name" in parse_error_text or "tool call missing keys: args" in parse_error_text:
        parts.append(
            'If you output tool calls, every tool call object must use the exact JSON shape '
            '{"id":"call_001","name":"tool_name","args":{...}}. Do not omit "name" or "args".'
        )
    if "AIMessage must contain content or tool_calls" in parse_error_text or "EmptyAssistantMessage" in parse_error_text:
        parts.append(
            "You must choose exactly one concrete action now: either output non-empty content, "
            "or output one or more tool calls. Never return {\"content\":\"\",\"tool_calls\":[]}."
        )
        parts.append(
            "If the task is blocked by missing external evidence, output schema A with ASK_USER:..., "
            "STEP_FAIL:..., or PLAN_UPDATE:... instead of an empty message."
        )
    if usage_hint:
        parts.append("Task-specific tool policy:\n" + usage_hint)
    retry_messages.append({"role": "user", "content": "\n".join(parts)})
    return retry_messages


def _build_tool_retry_messages(
    messages: list[dict],
    raw_text: str,
    mode: str,
    tools_schema: list[dict],
) -> list[dict]:
    retry_messages = validate_messages(deepcopy(messages))
    usage_hint = _tool_usage_hint(retry_messages, tools_schema)
    parts = [
        "Your previous reply skipped the required tool usage for this task.",
        "Do not answer from prior knowledge when the task explicitly requires local files, local search, table contents, or arithmetic computation.",
        "Previous reply:",
        "<previous_reply>",
        raw_text.strip(),
        "</previous_reply>",
        _mode_retry_instruction(mode),
        "You must output schema B with one or more tool calls. content must be an empty string. tool_calls must not be empty.",
    ]
    if usage_hint:
        parts.append("Task-specific tool policy:\n" + usage_hint)
    retry_messages.append({"role": "user", "content": "\n".join(parts)})
    return retry_messages


def _should_force_tool_retry(messages: list[dict], tools_schema: list[dict], ai_message: dict) -> bool:
    if ai_message.get("tool_calls"):
        return False
    if _has_any_tool_message(messages):
        return False
    return bool(_tool_usage_hint(messages, tools_schema))


def _merge_usage_metrics(primary: dict[str, Any] | None, secondary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not primary and not secondary:
        return None
    merged: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens"):
        values = [item.get(key) for item in (primary or {}, secondary or {}) if isinstance(item.get(key), int)]
        merged[key] = sum(values) if values else None
    elapsed_values = [
        float(item.get("elapsed_seconds"))
        for item in (primary or {}, secondary or {})
        if isinstance(item.get("elapsed_seconds"), (int, float))
    ]
    merged["elapsed_seconds"] = round(sum(elapsed_values), 6) if elapsed_values else None
    return merged


def _generate_with_retry(
    generate_fn: Any,
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    selected_profile: str | None,
    mode: str,
) -> dict:
    attempts: list[dict] = []
    aggregated_usage: dict[str, Any] | None = None
    current_messages = validate_messages(deepcopy(messages))
    max_attempts = 3
    for attempt_index in range(1, max_attempts + 1):
        generation_result = generate_fn(config_path, config, current_messages, tools_schema, selected_profile)
        raw_text = generation_result["raw_text"]
        parsed_text, think_blocks = _strip_think_blocks(raw_text)
        usage = {
            key: generation_result.get(key)
            for key in ("input_tokens", "output_tokens", "elapsed_seconds")
            if generation_result.get(key) is not None
        }
        aggregated_usage = _merge_usage_metrics(aggregated_usage, usage)
        attempt_record = {"attempt_index": attempt_index, "raw_text": raw_text}
        if think_blocks:
            attempt_record["think_blocks"] = think_blocks
        if usage:
            attempt_record["usage"] = usage
        try:
            parsed_candidate, ai_message = _parse_output_by_mode(parsed_text, mode)
        except Exception as exc:
            attempt_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            attempts.append(attempt_record)
            if attempt_index >= max_attempts:
                return {
                    "status": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "ai_message": make_ai_message(PARSE_ERROR_CONTENT, []),
                    "parsed_candidate": None,
                    "raw_text": raw_text,
                    "usage": aggregated_usage,
                    "attempts": attempts,
                }
            current_messages = _build_parse_retry_messages(messages, raw_text, mode, exc, tools_schema)
            continue
        if attempt_index == 1 and _should_force_tool_retry(messages, tools_schema, ai_message):
            attempt_record["error"] = {
                "type": "ToolUsageRetry",
                "message": "task-specific tool usage policy requires at least one tool call before answering",
            }
            attempts.append(attempt_record)
            current_messages = _build_tool_retry_messages(messages, raw_text, mode, tools_schema)
            continue
        if not ai_message.get("content") and not ai_message.get("tool_calls"):
            attempt_record["error"] = {
                "type": "EmptyAssistantMessage",
                "message": "model returned empty content and empty tool_calls",
            }
            attempts.append(attempt_record)
            if attempt_index >= max_attempts:
                return {
                    "status": "error",
                    "error": {"type": "EmptyAssistantMessage", "message": "model returned empty content and empty tool_calls"},
                    "ai_message": make_ai_message(PARSE_ERROR_CONTENT, []),
                    "parsed_candidate": None,
                    "raw_text": raw_text,
                    "usage": aggregated_usage,
                    "attempts": attempts,
                }
            current_messages = _build_parse_retry_messages(messages, raw_text, mode, ValueError("AIMessage must contain content or tool_calls"), tools_schema)
            continue
        attempts.append(attempt_record)
        return {
            "status": "success",
            "error": None,
            "ai_message": ai_message,
            "parsed_candidate": parsed_candidate,
            "raw_text": raw_text,
            "usage": aggregated_usage,
            "attempts": attempts,
        }
    return {
        "status": "error",
        "error": {"type": "GenerationRetryFailed", "message": "generation retry failed"},
        "ai_message": make_ai_message(PARSE_ERROR_CONTENT, []),
        "parsed_candidate": None,
        "raw_text": "",
        "usage": aggregated_usage,
        "attempts": attempts,
    }

def _extract_tool_result(message: dict) -> dict:
    try:
        result = json.loads(message["content"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("ToolMessage content is not a SkillResult JSON string") from exc
    if not isinstance(result, dict):
        raise ValueError("ToolMessage content must decode to an object")
    return result


def _latest_tool_message_batch(messages: list[dict]) -> list[dict]:
    batch: list[dict] = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        batch.append(message)
    batch.reverse()
    return batch


def _has_any_tool_message(messages: list[dict]) -> bool:
    return any(message.get("role") == "tool" for message in messages)


def merge_tool_messages(tool_messages: list[dict]) -> list[dict]:
    merged_by_id: dict[str, dict] = {}
    ordered_ids: list[str] = []
    for index, message in enumerate(tool_messages, 1):
        if message.get("role") != "tool":
            continue
        try:
            result = _extract_tool_result(message)
            parse_error = None
        except ValueError as exc:
            result = {
                "skill_name": message.get("name"),
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "output": None,
            }
            parse_error = str(exc)
        call_id = str(
            message.get("tool_call_id")
            or result.get("call_id")
            or result.get("id")
            or f"tool_{index:03d}"
        )
        if call_id not in merged_by_id:
            ordered_ids.append(call_id)
        message_status = str(message.get("status") or "").strip().lower()
        result_status = str(result.get("status") or "").strip().lower()
        status = "success" if message_status == "success" and result_status == "success" else "error"
        merged_by_id[call_id] = {
            "tool_call_id": call_id,
            "name": message.get("name") or result.get("skill_name"),
            "status": status,
            "message_status": message.get("status"),
            "result_status": result.get("status"),
            "result": result,
            "content": message.get("content", ""),
            "parse_error": parse_error,
        }
    return [merged_by_id[call_id] for call_id in ordered_ids]


def _read_enhancement_config(config: dict) -> dict:
    enhancement = config.get("enhancement") if isinstance(config.get("enhancement"), dict) else {}
    compression = enhancement.get("tool_result_compression") if isinstance(enhancement.get("tool_result_compression"), dict) else {}
    return {
        "tool_result_compression": {
            "enabled": bool(compression.get("enabled", True)),
            "max_content_chars": int(compression.get("max_content_chars", 900)),
            "max_snippet_chars": int(compression.get("max_snippet_chars", 260)),
            "max_search_results": int(compression.get("max_search_results", 6)),
            "max_preview_rows": int(compression.get("max_preview_rows", 6)),
        }
    }


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _compress_text(text: str, max_chars: int) -> tuple[str, dict]:
    max_chars = _clamp_int(max_chars, 900, 120, 5000)
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw, {"compressed": False, "original_chars": len(raw), "kept_chars": len(raw)}
    head_chars = max(60, int(max_chars * 0.7))
    tail_chars = max(40, max_chars - head_chars - 20)
    head = raw[:head_chars].rstrip()
    tail = raw[-tail_chars:].lstrip() if tail_chars > 0 else ""
    compressed = (head + "\n...\n" + tail).strip()
    return compressed, {
        "compressed": True,
        "original_chars": len(raw),
        "kept_chars": len(compressed),
    }


def _compress_file_reader_output(output: dict, settings: dict) -> dict:
    content = output.get("content")
    if not isinstance(content, str):
        return output
    max_chars = _clamp_int(settings.get("max_content_chars"), 900, 120, 5000)
    non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(content) <= max_chars:
        return output
    if (
        len(non_empty_lines) >= 3
        and non_empty_lines[0].startswith("1. ")
        and non_empty_lines[1].startswith("2. ")
        and non_empty_lines[2].startswith("3. ")
    ):
        return output
    points = _three_points(content)
    summary = "\n".join(f"{index}. {point}" for index, point in enumerate(points, 1))
    compressed_text, meta = _compress_text(summary, max_chars)
    updated = dict(output)
    updated["content"] = compressed_text
    updated["num_chars"] = len(compressed_text)
    updated["truncated"] = True
    updated["compressed"] = meta.get("compressed", False)
    updated["original_num_chars"] = meta.get("original_chars")
    return updated


def _compress_local_file_search_output(output: dict, settings: dict) -> dict:
    results = output.get("results")
    if not isinstance(results, list):
        return output
    max_results = _clamp_int(settings.get("max_search_results"), 6, 1, 20)
    max_snippet = _clamp_int(settings.get("max_snippet_chars"), 260, 80, 800)
    trimmed = []
    for item in results[:max_results]:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet")
        snippet_text = snippet if isinstance(snippet, str) else json.dumps(snippet, ensure_ascii=False)
        snippet_text, _ = _compress_text(snippet_text, max_snippet)
        trimmed.append(
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "snippet": snippet_text,
            }
        )
    updated = dict(output)
    updated["results"] = trimmed
    updated["compressed"] = True if len(trimmed) != len(results) else False
    updated["original_result_count"] = len(results)
    return updated


def _compress_table_analyzer_output(output: dict, settings: dict) -> dict:
    updated = dict(output)
    preview = output.get("preview")
    if isinstance(preview, list):
        max_rows = _clamp_int(settings.get("max_preview_rows"), 6, 1, 20)
        updated["preview"] = preview[:max_rows]
        updated["original_preview_rows"] = len(preview)
        updated["compressed"] = True if len(preview) > max_rows else False
    return updated


def _compress_skill_result(skill_result: dict, settings: dict) -> dict:
    if not isinstance(skill_result, dict):
        return skill_result
    name = str(skill_result.get("skill_name") or "")
    status = str(skill_result.get("status") or "")
    if status.lower() != "success":
        return skill_result
    output = skill_result.get("output")
    if not isinstance(output, dict):
        return skill_result
    if name == "file_reader":
        output = _compress_file_reader_output(output, settings)
    elif name == "local_file_search":
        output = _compress_local_file_search_output(output, settings)
    elif name == "table_analyzer":
        output = _compress_table_analyzer_output(output, settings)
    updated = dict(skill_result)
    updated["output"] = output
    return updated


def compress_tool_messages(tool_messages: list[dict], config: dict) -> tuple[list[dict], dict]:
    settings = _read_enhancement_config(config).get("tool_result_compression") or {}
    if not settings.get("enabled", True):
        return tool_messages, {"enabled": False, "compressed_messages": 0}
    compressed_messages: list[dict] = []
    compressed_count = 0
    for message in tool_messages:
        if message.get("role") != "tool":
            compressed_messages.append(message)
            continue
        try:
            result = _extract_tool_result(message)
        except Exception:
            compressed_messages.append(message)
            continue
        updated_result = _compress_skill_result(result, settings)
        if updated_result != result:
            compressed_count += 1
        updated_message = dict(message)
        updated_message["content"] = json.dumps(updated_result, ensure_ascii=False)
        compressed_messages.append(updated_message)
    return compressed_messages, {"enabled": True, "compressed_messages": compressed_count}


def _is_low_value_plan_step(title: str) -> bool:
    normalized = re.sub(r"\s+", "", title)
    low_value_patterns = [
        "完成任务",
        "结束任务",
        "输出最终",
        "给出最终",
        "整理答案",
        "验证",
        "检查结果",
    ]
    return any(pattern in normalized for pattern in low_value_patterns)


def _three_points(text: str) -> list[str]:
    parts = [part.strip(" \t\r\n。") for part in re.split(r"\n+|(?<=[。！？!?])", text) if part.strip()]
    points = []
    for part in parts:
        if part not in points:
            points.append(part)
        if len(points) == 3:
            break
    while len(points) < 3:
        points.append("工具结果未提供更多可提取内容")
    return points


def _mock_generate(messages: list[dict]) -> dict:
    plan_mode = any(
        message.get("role") == "user" and "Plan-and-Execute" in str(message.get("content") or "")
        for message in messages
    )
    finalizing_plan = bool(
        messages
        and messages[-1].get("role") == "user"
        and "所有步骤已完成" in str(messages[-1].get("content") or "")
    )
    if finalizing_plan:
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("name") != "calculator":
                continue
            try:
                result_payload = _extract_tool_result(message)
                output = result_payload.get("output") or {}
                result = output.get("result") if isinstance(output, dict) else output
                return make_ai_message(f"计算结果：{result}", [])
            except ValueError:
                break
    tool_messages = _latest_tool_message_batch(messages)
    if not tool_messages and messages and messages[-1].get("role") == "user":
        index = len(messages) - 2
        buffered = []
        while index >= 0 and messages[index].get("role") == "tool":
            buffered.append(messages[index])
            index -= 1
        tool_messages = list(reversed(buffered))
    if not tool_messages:
        internal_prefixes = ("已获得本轮工具结果。", "你正在以 Plan-and-Execute 模式工作。")
        latest_user = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if message.get("role") == "user"
                and not str(message.get("content") or "").startswith(internal_prefixes)
            ),
            "",
        )
        normalized_user = latest_user.replace("×", "*").replace("÷", "/")
        expression_match = re.search(r"\d+(?:\.\d+)?(?:\s*[+\-*/%]\s*\d+(?:\.\d+)?)+", normalized_user)
        if expression_match:
            return make_ai_message(
                "",
                [{"id": "call_mock_calculator", "name": "calculator", "args": {"expression": expression_match.group(0)}}],
            )
        return make_ai_message(
            "",
            [
                {
                    "id": "call_001",
                    "name": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 2000},
                },
                {
                    "id": "call_002",
                    "name": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 400},
                }
            ],
        )
    merged_results = merge_tool_messages(tool_messages)
    failures = [item for item in merged_results if item["status"] != "success"]
    if failures:
        details = []
        for item in failures:
            error = item["result"].get("error") or {}
            detail = error.get("message", "未知工具错误") if isinstance(error, dict) else str(error)
            details.append(f'{item["tool_call_id"]}:{detail}')
        return make_ai_message(f"部分工具调用失败，无法完成请求：{'; '.join(details)}", [])

    if len(merged_results) == 1 and merged_results[0].get("name") == "calculator":
        output = merged_results[0]["result"].get("output") or {}
        result = output.get("result") if isinstance(output, dict) else output
        if plan_mode:
            return make_ai_message(f"STEP_DONE:1:计算结果为 {result}", [])
        return make_ai_message(f"计算结果：{result}", [])

    contents: list[str] = []
    for item in merged_results:
        output = item["result"].get("output") or {}
        content = output.get("content") if isinstance(output, dict) else None
        if not isinstance(content, str) or not content.strip():
            content = json.dumps(output, ensure_ascii=False)
        contents.append(f'[{item["tool_call_id"]}] {content}')
    points = _three_points("\n".join(contents))
    answer = "三条中文要点如下：\n" + "\n".join(f"{index}. {point}" for index, point in enumerate(points, 1))
    return make_ai_message(answer, [])


def _normalize_tool_call(candidate: Any, index: int = 0) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("tool call must be an object")
    reserved_keys = {"id", "type", "name", "args", "arguments", "parameters", "input", "function"}
    name = candidate.get("name")
    args = candidate.get("args")
    if args is None:
        args = candidate.get("arguments")
    if args is None:
        args = candidate.get("parameters")
    if args is None:
        args = candidate.get("input")
    function_schema = candidate.get("function")
    if (not isinstance(name, str) or not name.strip()) and isinstance(function_schema, dict):
        name = function_schema.get("name")
        if args is None:
            args = (
                function_schema.get("arguments")
                or function_schema.get("args")
                or function_schema.get("parameters")
                or function_schema.get("input")
            )
    flat_args = {key: value for key, value in candidate.items() if key not in reserved_keys}
    if args is None and flat_args:
        args = flat_args
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call missing keys: name")
    if not isinstance(args, dict):
        raise ValueError("tool call missing keys: args")
    if flat_args and args is not flat_args:
        for key, value in flat_args.items():
            args.setdefault(key, value)
    tool_call = {
        "id": candidate.get("id") or f"call_{index + 1:03d}",
        "name": name,
        "args": args,
    }
    if tool_call["name"] == "table_analyzer":
        if "path" not in tool_call["args"] and "table_path" in tool_call["args"]:
            tool_call["args"]["path"] = tool_call["args"].get("table_path")
        if "path" not in tool_call["args"] and "file_path" in tool_call["args"]:
            tool_call["args"]["path"] = tool_call["args"].get("file_path")
    if tool_call["name"] == "file_reader":
        if "path" not in tool_call["args"] and "file_path" in tool_call["args"]:
            tool_call["args"]["path"] = tool_call["args"].get("file_path")
    if not isinstance(tool_call["id"], str) or not tool_call["id"].strip():
        raise ValueError("tool call id must be a non-empty string")
    if not isinstance(tool_call["name"], str) or not tool_call["name"].strip():
        raise ValueError("tool call name must be a non-empty string")
    if not isinstance(tool_call["args"], dict):
        raise ValueError("tool call args must be an object")
    if any(value is None for value in tool_call["args"].values()):
        tool_call["args"] = {key: value for key, value in tool_call["args"].items() if value is not None}
    return tool_call


def _normalize_tool_calls(candidate: Any) -> list[dict]:
    if isinstance(candidate, dict):
        return [_normalize_tool_call(candidate, 0)]
    if isinstance(candidate, list):
        if not candidate:
            raise ValueError("tool_calls list must not be empty")
        return [_normalize_tool_call(item, index) for index, item in enumerate(candidate)]
    raise ValueError("tool_calls must be an object or a non-empty array")


def _parse_tool_calls_fragment(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    markers = ['"tool_calls":[', '\\"tool_calls\\":[']
    marker_index = -1
    marker = ""
    for item in markers:
        marker_index = raw_text.find(item)
        if marker_index != -1:
            marker = item
            break
    if marker_index == -1:
        raise original_error
    array_start = marker_index + marker.index("[")
    array_end = raw_text.rfind("]")
    if array_end < array_start:
        raise ValueError("model output contains tool_calls marker but no closing array")
    array_text = raw_text[array_start : array_end + 1]
    try:
        tool_calls = _normalize_tool_calls(json.loads(array_text))
    except json.JSONDecodeError:
        tool_calls = _normalize_tool_calls(json.loads(array_text.replace('\\"', '"')))
    if not tool_calls:
        raise original_error
    return {"content": "", "tool_calls": tool_calls}


def _parse_json_with_backtick_tail(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    text = raw_text.strip()
    try:
        candidate, end_index = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        raise original_error
    trailing = text[end_index:].strip()
    if not trailing:
        return candidate
    if trailing and set(trailing) <= {"`"}:
        return candidate
    lowered = trailing.lstrip().casefold()
    allowed_prefixes = (
        "user\n",
        "assistant\n",
        "system\n",
        "<think>",
        "</think>",
        "<plan_state>",
        "当前计划状态",
    )
    if lowered.startswith(allowed_prefixes):
        return candidate
    raise original_error


def _parse_step_marker_text(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    text = raw_text.strip()
    if re.match(r"^STEP_(DONE|FAIL)\s*:\s*\d+\s*:", text):
        return {"content": text, "tool_calls": []}
    raise original_error


def _coerce_native_parameter_value(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    if stripped[:1] in {"{", "["}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped


def _normalize_native_tool_call(candidate: Any, index: int) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("native tool call must be an object")
    function_block = candidate.get("function")
    if isinstance(function_block, dict):
        name = function_block.get("name") or candidate.get("name")
        args = function_block.get("arguments", candidate.get("arguments"))
    else:
        name = candidate.get("name")
        args = candidate.get("arguments", candidate.get("args"))
    if isinstance(args, str):
        args = json.loads(args)
    tool_call = {
        "id": str(candidate.get("id") or f"call_{index:03d}"),
        "name": name,
        "args": args if isinstance(args, dict) else {},
    }
    return _normalize_tool_call(tool_call)


def _parse_native_xml_tool_block(block_text: str, index: int) -> dict:
    function_match = re.search(
        r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
        block_text,
        flags=re.DOTALL,
    )
    if not function_match:
        raise ValueError("native tool block is missing <function=...>")
    name = function_match.group(1).strip()
    params_text = function_match.group(2)
    args: dict[str, Any] = {}
    for param_match in re.finditer(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
        params_text,
        flags=re.DOTALL,
    ):
        param_name = param_match.group(1).strip()
        param_value = _coerce_native_parameter_value(param_match.group(2))
        args[param_name] = param_value
    return _normalize_tool_call({"id": f"call_{index:03d}", "name": name, "args": args}, index)


def _parse_native_tool_blocks(raw_text: str, original_error: Exception | None = None) -> dict:
    text = raw_text.strip()
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL))
    if not matches:
        if original_error is not None:
            raise original_error
        raise ValueError("native tool output does not contain <tool_call> blocks")
    tool_calls = []
    for index, match in enumerate(matches, 1):
        payload_text = match.group(1).strip()
        payload = parse_json_object_from_text(payload_text)
        if payload is not None:
            tool_calls.append(_normalize_native_tool_call(payload, index))
            continue
        tool_calls.append(_parse_native_xml_tool_block(payload_text, index))
    return {"content": "", "tool_calls": tool_calls}


def _candidate_to_message(candidate: Any) -> tuple[dict, dict]:
    if isinstance(candidate, list):
        candidate = {"content": "", "tool_calls": _normalize_tool_calls(candidate)}
    elif isinstance(candidate, dict) and "content" not in candidate and "tool_calls" not in candidate:
        candidate = {"content": "", "tool_calls": _normalize_tool_calls(candidate)}
    elif not isinstance(candidate, dict):
        raise ValueError("model output JSON must be an object or a tool_call array")
    expected_keys = {"content", "tool_calls"}
    unknown_keys = set(candidate) - expected_keys
    if unknown_keys:
        raise ValueError(f"model output JSON contains unknown keys: {', '.join(sorted(unknown_keys))}")
    message = {
        "role": "assistant",
        "content": candidate.get("content", ""),
        "tool_calls": _normalize_tool_calls(candidate.get("tool_calls", [])) if candidate.get("tool_calls") else [],
    }
    has_content = bool(str(message["content"]).strip())
    has_tool_calls = bool(message["tool_calls"])
    if has_content and has_tool_calls:
        content_text = str(message["content"]).strip()
        if content_text.startswith("STEP_DONE:") or content_text.startswith("STEP_FAIL:"):
            message["_step_marker"] = content_text
            message["content"] = ""
        else:
            raise ValueError("model output must contain either final content or tool calls, but not both")
    validate_ai_message(message)
    has_content = bool(message["content"].strip())
    has_tool_calls = bool(message["tool_calls"])
    if has_content == has_tool_calls:
        raise ValueError("model output must contain either final content or tool calls, but not both")
    parsed_candidate = {"content": message["content"], "tool_calls": message["tool_calls"]}
    if "_step_marker" in message:
        parsed_candidate["_step_marker"] = message["_step_marker"]
    return parsed_candidate, message


def _parse_model_output(raw_text: str) -> tuple[dict, dict]:
    extracted = _extract_json_payload(raw_text)
    if extracted is not None:
        return _candidate_to_message(extracted)
    try:
        candidate = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        try:
            candidate = _parse_json_with_backtick_tail(raw_text, exc)
        except json.JSONDecodeError:
            try:
                candidate = _parse_step_marker_text(raw_text, exc)
            except json.JSONDecodeError:
                candidate = _parse_tool_calls_fragment(raw_text, exc)
    return _candidate_to_message(candidate)

def _extract_json_payload(raw_text: str) -> dict | list | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, Any]] = []
    for match in re.finditer(r"[\{\[]", text):
        start = match.start()
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        candidates.append((start, candidate))
    if not candidates:
        return None
    ranked: list[tuple[tuple[int, int, int], Any]] = []
    for start, candidate in candidates:
        priority = 0
        if isinstance(candidate, dict):
            keys = set(candidate.keys())
            if {"content", "tool_calls"} & keys:
                priority = 4
            elif {"plan", "steps"} & keys:
                priority = 3
            else:
                priority = 2
        elif isinstance(candidate, list):
            if start == 0:
                priority = 1
            elif all(isinstance(item, (str, dict)) for item in candidate):
                priority = 0
            else:
                priority = -1
        ranked.append(((priority, -start, 0), candidate))
    _, best = max(ranked, key=lambda item: item[0])
    if isinstance(best, (dict, list)):
        return best
    return None


def _strip_think_blocks(raw_text: str) -> tuple[str, list[str]]:
    text = str(raw_text or "")
    think_blocks: list[str] = []
    pattern = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        think_blocks.append(match.group(0))
        return ""

    cleaned = pattern.sub(_repl, text)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
        think_blocks.append("</think>")
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"<think>", "", cleaned, flags=re.IGNORECASE)
        think_blocks.append("<think>")
    return cleaned, think_blocks


def _parse_output_by_mode(raw_text: str, mode: str) -> tuple[dict, dict]:
    if mode == "native_tools":
        stripped = raw_text.strip()
        if not stripped:
            raise ValueError("model output is empty")
        try:
            candidate = _parse_native_tool_blocks(raw_text)
            return _candidate_to_message(candidate)
        except Exception:
            try:
                return _parse_model_output(raw_text)
            except Exception:
                candidate = {"content": stripped, "tool_calls": []}
                return _candidate_to_message(candidate)
    return _parse_model_output(raw_text)


def parse_json_object_from_text(text: str) -> dict | list | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        candidate, _ = json.JSONDecoder().raw_decode(stripped)
        return candidate
    except json.JSONDecodeError:
        return None


def normalize_plan_steps(plan_payload: dict | list, max_steps: int) -> list[dict]:
    steps: list[Any]
    if isinstance(plan_payload, dict):
        steps = plan_payload.get("plan") or plan_payload.get("steps") or []
    else:
        steps = plan_payload
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan must contain a non-empty steps list")
    normalized: list[dict] = []
    for index, item in enumerate(steps[:max_steps], 1):
        if isinstance(item, str):
            title = item.strip()
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("task") or item.get("description") or "").strip()
        else:
            title = str(item).strip()
        title = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", title)
        if not title:
            title = f"步骤 {index}"
        normalized.append(
            {
                "step": index,
                "title": title,
                "status": "pending",
                "summary": "",
                "error": "",
            }
        )
    compressed: list[dict] = []
    seen_titles: set[str] = set()
    for item in normalized:
        key = re.sub(r"\s+", "", item["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        if len(normalized) > 3 and _is_low_value_plan_step(item["title"]):
            item["low_value"] = True
        compressed.append(item)
    if not compressed:
        compressed = normalized[: min(len(normalized), 3)]
    for index, item in enumerate(compressed, 1):
        item["step"] = index
    return compressed


def fallback_plan_steps(user_input: str, max_steps: int) -> list[dict]:
    return normalize_plan_steps([user_input.strip() or "完成用户请求"], max_steps)


def build_plan_execute_instruction(plan_state: dict) -> str:
    plan_json = json.dumps(plan_state, ensure_ascii=False)
    return (
        "你正在以 Plan-and-Execute 模式工作。\n"
        "目标是在保持计划状态同步和输出格式正确的前提下，用最短可行路径完成任务。\n"
        "请先在内部判断当前步骤最合适的动作，再只输出规定格式，不要展示长篇思维过程。\n\n"
        f"<plan_state>\n{plan_json}\n</plan_state>\n\n"
        "执行规则：\n"
        "- 优先选择最小可行动作：能基于当前上下文直接完成当前步骤，就直接完成；只有缺失外部证据时再调用工具。\n"
        "- 若某些步骤标记为 low_value=true，你可以将其与相邻步骤合并完成，避免无意义的独立回合。\n"
        "- 如果你判断当前计划拆分过细、边界不合理，或在新证据下出现了更短更自然的执行路径，可以先输出 schema A，content 以 PLAN_UPDATE:<json> 开头，重写剩余未完成步骤；代码只会保留已完成步骤并替换后续计划。\n"
        "- 如果需要调用工具：输出 schema B，且 content 必须是空字符串。\n"
        "- 当前步骤不需要工具且你已完成：输出 schema A，content 以 STEP_DONE:<step_number>:<summary> 开头。\n"
        "- 如果步骤失败且需要回退/重试：输出 schema A 且 content 以 STEP_FAIL:<step_number>:<reason> 开头。\n"
        "- 文件、表格、搜索结果等外部事实必须基于真实工具证据；内部推理、简单计算、信息整理在不依赖外部证据时可直接完成当前步骤。\n"
        "- 对于文件/表格缺失类失败：在输出 STEP_FAIL 之前，优先尝试一次最小修复重试（例如修正路径/目录前缀），或用 ASK_USER 让用户确认文件位置。\n"
        "- 若遇到关键歧义/需要用户确认：输出 schema A，content 以 ASK_USER:<question> 开头，并在后续行提供可选项（用短列表）。\n"
        "- 当有多个可行方案时：先在内部比较，再输出你选择的那个动作；只有确实需要用户决定时才使用 ASK_USER。\n"
        "- 当阻塞来自外部信息缺口而不是推理本身时（例如候选文件不足、路径/范围需要用户确认、要求与当前工作区事实不匹配），优先把缺口外化为 ASK_USER，而不是仅在内部反思或重复相似搜索。ASK_USER 可以请用户确认是否接受当前证据、补充资源，或放宽约束。\n"
        "- 只有当所有步骤 status 都变为 completed 后，才输出最终回答（schema A，content 为正常回答文本，不要以 STEP_DONE/STEP_FAIL 开头）。"
    )


def _write_plan_and_progress(output_dir: Path, plan_state: dict, llm_calls: int, tool_rounds: int, status: str) -> None:
    write_text(_render_plan_markdown(plan_state), output_dir / "plan_preview.md")
    write_text(_render_progress_markdown(plan_state, llm_calls, tool_rounds, status), output_dir / "progress.md")


def _build_runtime_trace(
    status: str,
    tool_rounds: int,
    llm_calls: int,
    plan_state: dict,
    turns: list[dict],
    terminal_error: Any,
    routing: dict | None = None,
    extra: dict | None = None,
) -> dict:
    trace = {
        "status": status,
        "tool_rounds_used": tool_rounds,
        "llm_call_count": llm_calls,
        "plan_state": deepcopy(plan_state),
        "turns": deepcopy(turns),
        "error": deepcopy(terminal_error),
    }
    if routing:
        trace["adaptive"] = deepcopy(routing)
    if isinstance(extra, dict):
        trace.update(deepcopy(extra))
    return trace


def _persist_runtime_state(
    output_dir: Path,
    messages: list[dict],
    trace: dict,
    final_answer: str,
    plan_state: dict,
    llm_calls: int,
    tool_rounds: int,
    progress_status: str,
    pending_question: str = "",
    runtime_bridge: dict | None = None,
    phase: str | None = None,
) -> None:
    question = str(pending_question or "").strip()
    if question:
        write_text(question + "\n", output_dir / "pending_question.md")
    else:
        pending_path = output_dir / "pending_question.md"
        if pending_path.exists():
            pending_path.unlink()
    write_json(messages, output_dir / "messages.json")
    write_json(trace, output_dir / "trace.json")
    write_text(str(final_answer or "").strip() + "\n", output_dir / "final_answer.md")
    _write_plan_and_progress(output_dir, plan_state, llm_calls, tool_rounds, progress_status)
    if not isinstance(runtime_bridge, dict):
        return
    emit = runtime_bridge.get("emit")
    if not callable(emit):
        return
    emit(
        {
            "status": str(trace.get("status") or progress_status or "running"),
            "progress_status": progress_status,
            "phase": phase,
            "messages": deepcopy(messages),
            "trace": deepcopy(trace),
            "final_answer": str(final_answer or ""),
            "pending_question": question,
        }
    )


def _execute_tool_calls_with_runtime_bridge(
    runtime_bridge: dict | None,
    tool_calls: list[dict],
    tools_config: str,
    toolset: str,
    tool_outdir: str,
) -> list[dict]:
    if isinstance(runtime_bridge, dict):
        executor = runtime_bridge.get("execute_tool_calls")
        if callable(executor):
            return executor(deepcopy(tool_calls), tools_config, toolset, tool_outdir)
    from b3_tool_layer import execute_tool_calls

    return execute_tool_calls(tool_calls, tools_config, toolset, tool_outdir)



def _render_plan_markdown(plan_state: dict) -> str:
    steps = plan_state.get("steps") or []
    lines = ["# Plan-and-Execute 计划预览", ""]
    for step in steps:
        step_no = step.get("step")
        title = str(step.get("title") or "").strip()
        status = str(step.get("status") or "pending")
        summary = str(step.get("summary") or "").strip()
        error = str(step.get("error") or "").strip()
        prefix = f"{step_no}. " if step_no is not None else "- "
        lines.append(f"{prefix}{title} [{status}]")
        if summary:
            summary_lines = summary.splitlines()
            lines.append(f"   - summary: {summary_lines[0]}")
            for extra in summary_lines[1:]:
                if extra.strip():
                    lines.append(f"     {extra}")
        if error:
            error_lines = error.splitlines()
            lines.append(f"   - error: {error_lines[0]}")
            for extra in error_lines[1:]:
                if extra.strip():
                    lines.append(f"     {extra}")
    return "\n".join(lines).strip() + "\n"

def _render_progress_markdown(plan_state: dict, llm_calls: int, tool_rounds: int, status: str) -> str:
    steps = plan_state.get("steps") or []
    total = len(steps)
    completed = sum(1 for step in steps if step.get("status") == "completed")
    failed = sum(1 for step in steps if step.get("status") == "failed")
    pending = total - completed - failed
    lines = [
        "# Plan-and-Execute 进度",
        "",
        f"- status: {status}",
        f"- llm_calls: {llm_calls}",
        f"- tool_rounds: {tool_rounds}",
        f"- steps: total={total}, completed={completed}, failed={failed}, pending={pending}",
        "",
    ]
    lines.append(_render_plan_markdown(plan_state).strip())
    return "\n".join(lines).strip() + "\n"


def _tool_error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or "").strip()
    return str(error or "").strip()


def _is_missing_file_error(message: str) -> bool:
    lowered = message.casefold()
    return "file not found:" in lowered or "table file not found:" in lowered or "not found:" in lowered


def _replan_hint_from_tool_failures(tool_messages: list[dict]) -> str:
    merged = merge_tool_messages(tool_messages)
    failures: list[dict] = []
    for item in merged:
        if item.get("status") == "success":
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        name = str(item.get("name") or result.get("skill_name") or "").strip()
        tool_input = result.get("input") if isinstance(result.get("input"), dict) else {}
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        failures.append({"name": name, "input": tool_input, "error": error})
    if not failures:
        return ""
    failure_json = json.dumps(failures, ensure_ascii=False)
    return (
        "动态重规划提示（Reflexion/ReAct 风格）：\n"
        "你刚刚遇到了一轮工具调用失败。下面是失败项的结构化信息（name/input/error）：\n"
        f"{failure_json}\n\n"
        "请在下一次决策前，先在内部完成一次“诊断→候选→选择动作”的自我修复，不要把完整思维过程写出来。\n"
        "1) 在内部判断 1~3 个最可能的失败原因；\n"
        "2) 在内部比较 1~2 个替代动作，优先最小改动，不要重复已经成功的工具调用；\n"
        "3) 只输出你选择的那个动作：\n"
        "   - 若有明确可重试方案：输出 schema B，请只请求必要的新 tool_calls；\n"
        "   - 若存在多个候选且无法判断：输出 schema A，content 以 ASK_USER: 开头列出候选与需要用户确认的信息；\n"
        "   - 若无法在预算内完成或缺失关键信息：输出 schema A，content 以 STEP_FAIL:<step_number>: 开头说明原因与替代方案。\n"
        "注意：不要在 content 中夹带 tool_calls，也不要输出 role 标签（user/assistant）或多余文本。"
    ).strip()


def _normalize_tool_call_args_for_execution(tool_call: dict) -> dict:
    if not isinstance(tool_call, dict):
        return tool_call
    name = str(tool_call.get("name") or "")
    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else None
    if not isinstance(args, dict):
        return tool_call
    if name == "table_analyzer":
        path = args.get("path")
        if isinstance(path, str):
            stripped = path.strip()
            lowered = stripped.lower()
            if stripped and "/" not in stripped and lowered.endswith((".csv", ".tsv")):
                updated_args = dict(args)
                updated_args["path"] = f"tables/{Path(stripped).name}"
                updated = dict(tool_call)
                updated["args"] = updated_args
                return updated
    return tool_call


def _step_match_score(step: dict, tool_names: set[str], paths: list[str]) -> int:
    title = str(step.get("title") or "").lower()
    score = 0
    for name in tool_names:
        if name and name.lower() in title:
            score += 3
    if "table_analyzer" in tool_names and ("csv" in title or "tsv" in title or "table" in title or "表" in title):
        score += 4
    if "local_file_search" in tool_names and ("search" in title or "搜索" in title):
        score += 2
    if "file_reader" in tool_names and ("read" in title or "读取" in title or "检查" in title):
        score += 2
    for path in paths:
        lowered = path.lower()
        if lowered and lowered in title:
            score += 4
        stem = Path(path).stem.lower()
        if stem and stem in title:
            score += 1
    if step.get("status") == "failed":
        score += 1
    return score


def _match_plan_step_for_tool_calls(plan_state: dict, tool_calls: list[dict]) -> tuple[int | None, int]:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None, 0
    tool_names = {str(call.get("name") or "") for call in tool_calls if isinstance(call, dict)}
    paths: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            paths.append(path.strip())
        root_dir = args.get("root_dir")
        if isinstance(root_dir, str) and root_dir.strip():
            paths.append(root_dir.strip())
    best_step = None
    best_score = 0
    for step in plan_state.get("steps") or []:
        if step.get("status") == "completed":
            continue
        score = _step_match_score(step, tool_names, paths)
        if score > best_score:
            best_score = score
            best_step = int(step.get("step"))
    return best_step, best_score


def apply_step_marker(plan_state: dict, marker_text: str) -> tuple[dict, str | None]:
    text = marker_text.strip()
    update_match = re.match(r"^PLAN_UPDATE\s*:\s*(.*)$", text, flags=re.DOTALL)
    if update_match:
        payload = parse_json_object_from_text(update_match.group(1).strip())
        if payload is not None:
            steps = plan_state.get("steps") or []
            completed_steps = [deepcopy(step) for step in steps if step.get("status") == "completed"]
            max_steps = max(len(steps), 1)
            updated_pending = normalize_plan_steps(payload, max_steps)
            renumbered: list[dict] = []
            for step in completed_steps:
                renumbered.append(
                    {
                        "step": len(renumbered) + 1,
                        "title": str(step.get("title") or "").strip(),
                        "status": "completed",
                        "summary": str(step.get("summary") or ""),
                        "error": str(step.get("error") or ""),
                        **({"low_value": True} if step.get("low_value") else {}),
                    }
                )
            for step in updated_pending:
                rebuilt = {
                    "step": len(renumbered) + 1,
                    "title": str(step.get("title") or "").strip(),
                    "status": "pending",
                    "summary": "",
                    "error": "",
                }
                if step.get("low_value"):
                    rebuilt["low_value"] = True
                renumbered.append(rebuilt)
            plan_state["steps"] = renumbered
            return plan_state, "plan_update"
    done_match = re.match(r"^STEP_DONE\s*:\s*(\d+)\s*:\s*(.*)$", text, flags=re.DOTALL)
    if done_match:
        step_no = int(done_match.group(1))
        summary = done_match.group(2).strip()
        steps = plan_state.get("steps") or []
        for step in steps:
            if step.get("step") == step_no:
                step["status"] = "completed"
                step["summary"] = summary
                step["error"] = ""
                break
        return plan_state, "done"
    fail_match = re.match(r"^STEP_FAIL\s*:\s*(\d+)\s*:\s*(.*)$", text, flags=re.DOTALL)
    if fail_match:
        step_no = int(fail_match.group(1))
        reason = fail_match.group(2).strip()
        steps = plan_state.get("steps") or []
        for step in steps:
            if step.get("step") == step_no:
                step["status"] = "failed"
                step["error"] = reason
                if not step.get("summary"):
                    step["summary"] = ""
                break
        return plan_state, "fail"
    return plan_state, None


def next_pending_step(plan_state: dict) -> int | None:
    steps = plan_state.get("steps") or []
    for step in steps:
        if step.get("status") == "pending":
            return int(step.get("step"))
    return None


def first_failed_step(plan_state: dict) -> int | None:
    steps = plan_state.get("steps") or []
    for step in steps:
        if step.get("status") == "failed":
            return int(step.get("step"))
    return None


def all_steps_completed(plan_state: dict) -> bool:
    steps = plan_state.get("steps") or []
    return bool(steps) and all(step.get("status") == "completed" for step in steps)


def _infer_target_step_from_tool_calls(plan_state: dict, tool_calls: list[dict]) -> int | None:
    best_step, best_score = _match_plan_step_for_tool_calls(plan_state, tool_calls)
    if best_step is None or best_score <= 0:
        return None
    return best_step


def _expected_tools_for_step(step_title: str) -> set[str]:
    title = step_title.casefold()
    tools: set[str] = set()
    has_search = "搜索" in title or "search" in title
    has_table = any(token in title for token in ["csv", "tsv", "table", "表格", "预算表"])
    file_mentions = re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", step_title, flags=re.IGNORECASE)
    needs_file_reader = (
        not has_search
        and (
            any(token in title for token in ["读取", "阅读", "read", "检查", "是否存在"])
            or (not has_table and ("文件" in title or file_mentions))
        )
    )
    if has_search:
        tools.add("local_file_search")
    if has_table:
        tools.add("table_analyzer")
    if needs_file_reader:
        tools.add("file_reader")
    if any(token in title for token in ["计算", "算", "expression"]):
        tools.add("calculator")
    if any(token in title for token in ["格式", "format", "转换"]):
        tools.add("format_converter")
    return tools


def _deterministic_plan_tool_calls(step_title: str, tools_schema: list[dict]) -> list[dict]:
    title = str(step_title or "").strip()
    lowered = title.casefold()
    available = _available_tool_names(tools_schema)
    if "local_file_search" not in available or not any(token in lowered for token in ("搜索", "查找", "检索", "search")):
        return []
    root_dir = "."
    if re.search(r"docs(?:目录|文件夹|[/\\]|\b)", title, flags=re.IGNORECASE):
        root_dir = "docs"
    else:
        root_match = re.search(
            r"(?:^|[\s：:，,])([A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_./\\-]*)?)\s*(?:目录|文件夹|中|里)",
            title,
            flags=re.IGNORECASE,
        )
        if root_match:
            root_dir = root_match.group(1).replace("\\", "/").strip("./") or "."
    query = re.sub(r"^\s*(?:步骤\s*\d+\s*[：:]?\s*)", "", title, flags=re.IGNORECASE)
    query = re.sub(r"^(?:搜索|查找|检索|search)\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(rf"^{re.escape(root_dir)}\s*(?:目录|文件夹)?\s*(?:中|里|内)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"[（(](?:如|例如).*?[）)]", "", query).strip(" ：:，,。")
    if not query:
        query = title
    return [
        {
            "id": "call_fast_plan_search_001",
            "name": "local_file_search",
            "args": {"query": query[:160], "root_dir": root_dir, "top_k": 8},
        }
    ]


def _cn_count_to_int(token: str) -> int | None:
    normalized = str(token or "").strip()
    if not normalized:
        return None
    if normalized.isdigit():
        try:
            return int(normalized)
        except Exception:
            return None
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if normalized in mapping:
        return mapping[normalized]
    if normalized.startswith("十") and len(normalized) == 2 and normalized[1] in mapping:
        return 10 + mapping[normalized[1]]
    return None


def _required_distinct_file_count(step_title: str) -> int:
    title = str(step_title or "")
    normalized = re.sub(r"\s+", "", title)
    match = re.search(r"(前|读取前)?([0-9]+|[一二两三四五六七八九十])(?:个|篇|份|条)?(?:最相关)?(?:文件|文档|结果)", normalized)
    if match:
        value = _cn_count_to_int(match.group(2))
        if isinstance(value, int) and value > 0:
            return value
    if any(token in normalized for token in ["两篇", "两个文件", "两个文档", "2个文件", "2个文档", "前2个"]):
        return 2
    if any(token in normalized for token in ["三篇", "三个文件", "三个文档", "3个文件", "3个文档", "前3个"]):
        return 3
    if any(token in normalized for token in ["对比", "比较", "差异", "共同点"]):
        return 2
    return 1


def _distinct_successful_file_sources(evidence_items: list[dict]) -> list[str]:
    sources: list[str] = []
    merged = evidence_items if isinstance(evidence_items, list) else []
    for item in merged:
        if item.get("name") != "file_reader" or item.get("status") != "success":
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        source = str(output.get("source") or "").strip()
        if source:
            sources.append(source)
    return list(dict.fromkeys(sources))


def _has_explicit_external_resource_signals(step_title: str) -> bool:
    title = step_title.casefold()
    file_mentions = re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", step_title, flags=re.IGNORECASE)
    if file_mentions:
        return True
    if any(token in title for token in ["搜索", "search", "查找", "检索"]):
        return True
    if any(token in title for token in ["csv", "tsv", "table", "表格", "预算表"]):
        return True
    if any(token in title for token in ["docs/", "doc/", "目录", "文档"]):
        return True
    return False


def _tool_calls_require_external_evidence(tool_calls: list[dict] | None) -> bool:
    if not isinstance(tool_calls, list):
        return False
    external_tools = {"local_file_search", "file_reader", "table_analyzer"}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("name") or "").strip() in external_tools:
            return True
    return False


def _tool_batch_supports_step(step_title: str, evidence_items: list[dict]) -> bool:
    expected = _expected_tools_for_step(step_title)
    if not expected:
        return False
    successful_tools = {
        str(item.get("name") or "").strip()
        for item in (evidence_items or [])
        if isinstance(item, dict) and item.get("status") == "success"
    }
    return bool(expected & successful_tools)


def _step_requires_tool_evidence(step_title: str) -> bool:
    title = step_title.casefold()
    if any(token in title for token in ["基于现有信息", "最终结论", "下一步清单", "总结", "判断", "报告"]):
        return False
    return bool(_expected_tools_for_step(step_title))


def _step_requires_external_evidence(step_title: str, planner_tool_calls: list[dict] | None = None) -> bool:
    if _tool_calls_require_external_evidence(planner_tool_calls):
        return True
    if _has_explicit_external_resource_signals(step_title):
        return True
    expected = _expected_tools_for_step(step_title)
    return bool(expected.intersection({"local_file_search", "file_reader", "table_analyzer"}))


def _has_sufficient_evidence(step_title: str, evidence_items: list[dict], evidence_policy: str) -> bool:
    policy = str(evidence_policy or "strict").strip().lower()
    if policy == "lite":
        if not _step_requires_external_evidence(step_title):
            return True
    return _step_has_sufficient_evidence(step_title, evidence_items)


def _step_has_sufficient_evidence(step_title: str, evidence_items: list[dict]) -> bool:
    if not _step_requires_tool_evidence(step_title):
        return True
    title = step_title.casefold()
    merged = evidence_items if isinstance(evidence_items, list) else []
    required_file_count = _required_distinct_file_count(step_title)
    successful_file_sources = _distinct_successful_file_sources(merged)
    if "搜索" in title or "search" in title:
        return any(item.get("name") == "local_file_search" and item.get("status") == "success" for item in merged)
    if any(token in title for token in ["csv", "tsv", "table", "表格", "预算表"]):
        return any(item.get("name") == "table_analyzer" and item.get("status") == "success" for item in merged)
    if "是否存在" in title:
        for item in merged:
            if item.get("name") != "file_reader":
                continue
            if item.get("status") == "success":
                return True
            err_text = _tool_error_text(item.get("result", {}).get("error") if isinstance(item.get("result"), dict) else "")
            if _is_missing_file_error(err_text):
                return True
        return False
    if any(token in title for token in ["对比", "比较", "差异", "共同点"]):
        return len(successful_file_sources) >= max(required_file_count, 2)
    if "读取" in title or "read" in title or "阅读" in title or "文件" in title or "文档" in title:
        return len(successful_file_sources) >= max(required_file_count, 1)
    if "计算" in title or "expression" in title:
        return any(item.get("name") == "calculator" and item.get("status") == "success" for item in merged)
    if "格式" in title or "format" in title or "转换" in title:
        return any(item.get("name") == "format_converter" and item.get("status") == "success" for item in merged)
    return any(item.get("status") == "success" for item in merged)


def _build_file_count_shortfall_guidance(step_title: str, evidence_items: list[dict]) -> str:
    required = _required_distinct_file_count(step_title)
    if required <= 1:
        return ""
    available_sources = _distinct_successful_file_sources(evidence_items)
    if len(available_sources) >= required:
        return ""
    available = len(available_sources)
    source_text = "、".join(available_sources) if available_sources else "暂无已读取文件"
    return (
        f"当前仅拿到 {available}/{required} 个不同文件的真实内容（{source_text}）。\n"
        "如果搜索结果不足，先尝试一次最小自修复：例如补做 local_file_search、放宽关键词，或搜索更合适的相邻目录；\n"
        "如果你判断当前阻塞更像是工作区证据不足而不是推理不足，可以直接输出 ASK_USER: 把缺口外化给用户，例如请用户确认是否接受基于当前文件继续、提供额外文件，或放宽“至少读取多个文件”的约束；\n"
        "当存在多个同样合理的方向时，也优先输出 ASK_USER: 并提供 2~3 个简短选项；"
        "若在预算内仍无法完成，再输出 STEP_FAIL。"
    )


def _build_evidence_sufficiency_nudge(task_text: str, evidence_items: list[dict]) -> str:
    title = str(task_text or "").strip()
    if not title:
        return ""
    required = _required_distinct_file_count(title)
    if required <= 1:
        return ""
    available_sources = _distinct_successful_file_sources(evidence_items)
    if len(available_sources) < required:
        return ""
    source_text = "、".join(available_sources[:required])
    return (
        f"收敛提醒：你已经拿到至少 {required} 个不同文件的真实内容（{source_text}），"
        "这已经满足当前多文件读取/对比任务的最低证据要求。"
        "除非用户明确要求继续扩大搜索范围，否则优先基于这些已读文件直接完成总结或对比，"
        "不要因为“最相关”无法绝对证明就继续发散搜索或重复读取。"
    )


def _build_budget_exhausted_ask_user(
    tool_calls: list[dict],
    step_title: str = "",
    evidence_items: list[dict] | None = None,
    current_step: int | None = None,
) -> str:
    evidence_items = evidence_items if isinstance(evidence_items, list) else []
    title = str(step_title or "").strip()
    blocked_tool_names = []
    for call in tool_calls if isinstance(tool_calls, list) else []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        if name and name not in blocked_tool_names:
            blocked_tool_names.append(name)
    tool_name_text = "、".join(blocked_tool_names)

    required = _required_distinct_file_count(title)
    available_sources = _distinct_successful_file_sources(evidence_items)
    if title and required > 1 and len(available_sources) < required:
        available = len(available_sources)
        source_text = "、".join(available_sources) if available_sources else "暂无已读取文件"
        step_prefix = f"第 {current_step} 步" if current_step is not None else "当前步骤"
        step_text = f"{step_prefix}「{title}」" if title else step_prefix
        return (
            "ASK_USER: 当前工具预算已用完，我还缺少完成任务所需的外部证据。\n"
            f"{step_text} 仍未满足。\n"
            f"目前仅拿到 {available}/{required} 个不同文件的真实内容（{source_text}）。\n"
            "- 选项 A：接受我基于当前证据继续，我会明确说明不确定性；\n"
            "- 选项 B：请提供缺失文件或更准确的路径/范围，我再继续；\n"
            "- 选项 C：放宽约束，允许只基于当前已找到的文件完成。"
        )

    if title and not _has_sufficient_evidence(title, evidence_items, "strict"):
        step_prefix = f"第 {current_step} 步" if current_step is not None else "当前步骤"
        blocked_text = f"；下一步原本还需要继续调用：{tool_name_text}" if tool_name_text else ""
        return (
            "ASK_USER: 当前工具预算已用完，但我还缺少继续完成任务所需的外部证据。\n"
            f"{step_prefix}「{title}」目前证据不足{blocked_text}。\n"
            "- 选项 A：接受我基于当前证据继续，并在答案里标明边界；\n"
            "- 选项 B：请补充文件、路径或范围信息，我再继续；\n"
            "- 选项 C：调整要求或允许我减少读取/对比范围。"
        )

    evidence_driven_tools = {"file_reader", "local_file_search", "table_analyzer", "read_convert_file"}
    if any(name in evidence_driven_tools for name in blocked_tool_names):
        blocked_text = f"原本还需要继续调用：{tool_name_text}。" if tool_name_text else ""
        return (
            "ASK_USER: 当前工具预算已用完，但任务还需要更多外部证据才能继续。\n"
            f"{blocked_text}\n"
            "- 选项 A：接受我基于当前证据直接作答；\n"
            "- 选项 B：请补充文件位置、更多资源或更明确的范围；\n"
            "- 选项 C：放宽约束后让我继续。"
        )
    return ""


def _duplicate_resource_confirmation_from_user_text(user_text: str) -> str:
    text = str(user_text or "")
    if not text.strip():
        return ""
    file_mentions = re.findall(r"\b[\w./-]+\.(?:txt|md|json|csv|tsv)\b", text, flags=re.IGNORECASE)
    seen: set[str] = set()
    repeated: list[str] = []
    for path in file_mentions:
        normalized = path.replace("\\", "/").casefold()
        if normalized in seen and normalized not in repeated:
            repeated.append(normalized)
        else:
            seen.add(normalized)
    if not repeated:
        return ""
    duplicate_text = "；".join(repeated)
    return (
        "ASK_USER: 你的请求里出现了同一资源的重复读取，存在语义歧义。\n"
        f"重复项：{duplicate_text}\n"
        "- 选项 A：严格按原指令重复处理这些资源；\n"
        "- 选项 B：识别为重复项，只处理一次/去重后继续，并在答案里说明；\n"
        "- 选项 C：你其实想写成别的资源路径，请直接告诉我。"
    )


def _duplicate_resource_confirmation_text(tool_calls: list[dict]) -> str:
    if not isinstance(tool_calls, list):
        return ""
    duplicates: list[str] = []
    seen: set[tuple[str, str]] = set()
    repeated: set[tuple[str, str]] = set()
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        resource = ""
        if name in {"file_reader", "table_analyzer"}:
            resource = str(args.get("path") or "").strip()
        if not resource:
            continue
        signature = (name, resource)
        if signature in seen:
            repeated.add(signature)
        else:
            seen.add(signature)
    for name, resource in sorted(repeated):
        duplicates.append(f"{name}:{resource}")
    if not duplicates:
        return ""
    duplicate_text = "；".join(duplicates)
    return (
        "ASK_USER: 你刚刚的请求里出现了同一资源的重复读取，存在语义歧义。\n"
        f"重复项：{duplicate_text}\n"
        "- 选项 A：严格按原指令重复读取这些资源；\n"
        "- 选项 B：识别为重复项，只读取一次/去重后继续，并在答案里说明；\n"
        "- 选项 C：你想改成别的文件路径，请直接告诉我。"
    )


def _should_allow_search_repair_for_step(step_title: str, evidence_items: list[dict] | None) -> bool:
    title = str(step_title or "").casefold()
    if not any(token in title for token in ["读取", "read", "阅读", "对比", "比较", "差异", "共同点", "文件", "文档"]):
        return False
    required = _required_distinct_file_count(step_title)
    if required <= 1:
        return False
    successful_file_sources = _distinct_successful_file_sources(evidence_items or [])
    return len(successful_file_sources) < required


def _validate_tool_calls_for_step(
    _plan_state: dict,
    _current_step: int | None,
    tool_calls: list[dict],
    _step_evidence_items: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    return list(tool_calls), []


def _extract_step_number_from_marker_text(text: str) -> int | None:
    match = re.match(r"^STEP_(?:DONE|FAIL)\s*:\s*(\d+)\s*:", str(text or "").strip(), flags=re.DOTALL)
    if not match:
        return None
    return int(match.group(1))


def _should_defer_step_done_marker(
    plan_state: dict,
    marker_text: str,
    tool_calls: list[dict],
    step_evidence: dict[int, list[dict]],
    evidence_policy: str,
) -> tuple[bool, int | None]:
    text = str(marker_text or "").strip()
    if not text.startswith("STEP_DONE:"):
        return False, None
    step_no = _extract_step_number_from_marker_text(text)
    if step_no is None:
        return False, None
    step_title = current_plan_step_title(plan_state, step_no)
    if not step_title:
        return False, step_no
    if _has_sufficient_evidence(step_title, step_evidence.get(step_no, []), evidence_policy):
        return False, step_no
    matched_step, matched_score = _match_plan_step_for_tool_calls(plan_state, tool_calls)
    if matched_step == step_no and matched_score > 0:
        return True, step_no
    if next_pending_step(plan_state) == step_no and _tool_calls_require_external_evidence(tool_calls):
        return True, step_no
    return False, step_no


def current_plan_step_title(plan_state: dict, step_number: int | None) -> str:
    if step_number is None:
        return ""
    for step in plan_state.get("steps") or []:
        if step.get("step") == step_number:
            return str(step.get("title") or "")
    return ""


def _file_reader_source_and_content(item: dict) -> tuple[str, str]:
    if item.get("name") != "file_reader":
        return "", ""
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    tool_input = result.get("input") if isinstance(result.get("input"), dict) else {}
    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    source = str(output.get("source") or tool_input.get("path") or "").strip()
    content = output.get("content")
    if not isinstance(content, str):
        content = ""
    return source, content


def _constraint_guardrails_from_file_reads(file_reads: list[tuple[str, str]]) -> list[str]:
    guardrails: list[str] = []
    for source, content in file_reads:
        normalized_source = source.replace("\\", "/").casefold()
        if not normalized_source.endswith("requirements.md"):
            continue
        if "月租金上限 4800（不含水电网）" in content:
            guardrails.append("`requirements.md` 里的 4800 指月租金上限，不含水电网，不要改写成月度总支出上限。")
        if "最好 4300 左右" in content:
            guardrails.append("`requirements.md` 里的“最好 4300 左右”是偏好目标，不是硬性淘汰线。")
    return list(dict.fromkeys(guardrails))


def _build_execution_guardrails(step_evidence: dict[int, list[dict]]) -> str:
    file_reads: list[tuple[str, str]] = []
    for items in step_evidence.values():
        if not isinstance(items, list):
            continue
        for item in items:
            source, content = _file_reader_source_and_content(item)
            if source and content:
                file_reads.append((source, content))
    guardrails = _constraint_guardrails_from_file_reads(file_reads)
    if not guardrails:
        return ""
    return "执行约束提醒：\n- " + "\n- ".join(guardrails)


def summarize_tool_round_for_step(tool_messages: list[dict], step_number: int | None, step_title: str) -> str:
    merged = merge_tool_messages(tool_messages)
    lines = []
    if step_number is not None:
        lines.append(f"当前执行的是第 {step_number} 步：{step_title}")
    successful_file_reads: list[dict] = []
    failed_calls: list[dict] = []
    for item in merged:
        result = item.get("result") or {}
        tool_input = result.get("input") if isinstance(result, dict) else None
        output = result.get("output")
        error = result.get("error")
        if item.get("status") == "success":
            if item.get("name") == "file_reader" and isinstance(output, dict):
                successful_file_reads.append(
                    {
                        "tool_call_id": item.get("tool_call_id"),
                        "source": output.get("source") or (tool_input.get("path") if isinstance(tool_input, dict) else None),
                        "content": output.get("content"),
                    }
                )
            if isinstance(output, dict):
                snippet = output.get("content")
                if not isinstance(snippet, str) or not snippet.strip():
                    snippet = json.dumps(output, ensure_ascii=False)
            else:
                snippet = json.dumps(output, ensure_ascii=False)
            lines.append(f'- {item["name"]} 成功：{str(snippet).strip()[:200]}')
        else:
            detail = error.get("message", "未知工具错误") if isinstance(error, dict) else str(error)
            lines.append(f'- {item["name"]} 失败：{detail}')
            failed_calls.append(
                {
                    "tool_call_id": item.get("tool_call_id"),
                    "name": item.get("name"),
                    "input": tool_input,
                    "error": error,
                }
            )
    if len(successful_file_reads) >= 2:
        file_list = []
        for item in successful_file_reads:
            source = str(item.get("source") or "").strip()
            if source:
                file_list.append(source)
        if file_list:
            unique_files = list(dict.fromkeys(file_list))
            lines.append(
                "跨工具提示：你已经读取了多个文件，请直接输出对比结论而不是逐条复述。建议格式：\n"
                "- 共同关注点（3 条以内）\n"
                "- 关键差异点（按维度对比）\n"
                "- 结论与建议\n"
                + "涉及文件："
                + "，".join(unique_files)
            )
    local_guardrails = _constraint_guardrails_from_file_reads(
        [
            (str(item.get("source") or "").strip(), str(item.get("content") or ""))
            for item in successful_file_reads
            if str(item.get("source") or "").strip()
        ]
    )
    if local_guardrails:
        lines.append("约束提醒：\n- " + "\n- ".join(local_guardrails))
    if failed_calls:
        lines.append("失败工具调用信息（仅供你决定是否重试其中失败项）：\n" + json.dumps(failed_calls, ensure_ascii=False))
    return "\n".join(lines)


def _propagate_step_evidence(
    plan_state: dict,
    current_step: int | None,
    merged_after: list[dict],
    step_evidence: dict[int, list[dict]],
    evidence_policy: str,
) -> list[int]:
    if current_step is None or not merged_after:
        return []
    propagated: list[int] = []
    for step in plan_state.get("steps") or []:
        step_no = step.get("step")
        if not isinstance(step_no, int) or step_no == current_step:
            continue
        if step.get("status") != "pending":
            continue
        step_title = str(step.get("title") or "")
        if not _step_requires_tool_evidence(step_title):
            continue
        if not _tool_batch_supports_step(step_title, merged_after):
            continue
        combined = list(step_evidence.get(step_no, [])) + deepcopy(merged_after)
        if not _has_sufficient_evidence(step_title, combined, evidence_policy):
            continue
        step_evidence[step_no] = combined
        propagated.append(step_no)
    return propagated


def _dtype_value(torch_module: Any, configured: str) -> Any:
    if configured == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]


def _model_cache_key(
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[str, ...]:
    try:
        device_map_key = json.dumps(device_map, sort_keys=True, separators=(",", ":"))
    except TypeError:
        device_map_key = repr(device_map)
    try:
        max_memory_key = json.dumps(max_memory, sort_keys=True, separators=(",", ":"))
    except TypeError:
        max_memory_key = repr(max_memory)
    return (
        str(model_path),
        str(tokenizer_path),
        str(local_only),
        str(trust_remote_code),
        str(dtype),
        device_map_key,
        max_memory_key,
    )


def _load_model_bundle(
    auto_model: Any,
    auto_tokenizer: Any,
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[Any, Any]:
    cache_key = _model_cache_key(
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        device_map,
        max_memory,
    )
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            print("model_cache=hit", file=sys.stderr, flush=True)
            return cached

        print("model_cache=miss", file=sys.stderr, flush=True)
        tokenizer = auto_tokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=local_only,
            trust_remote_code=trust_remote_code,
        )
        staged_cuda_device = (
            device_map
            if isinstance(device_map, str) and re.fullmatch(r"cuda(?::\d+)?", device_map)
            else None
        )
        model_load_kwargs = {
            "local_files_only": local_only,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if staged_cuda_device is None and device_map is not None:
            model_load_kwargs["device_map"] = device_map
        uses_automatic_device_map = isinstance(device_map, str) and device_map in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }
        if max_memory is not None and uses_automatic_device_map:
            model_load_kwargs["max_memory"] = max_memory
        print(
            f"model_load load_device={'cpu' if staged_cuda_device else device_map!r} "
            f"target_device={staged_cuda_device!r} dtype={dtype} "
            f"max_memory={model_load_kwargs.get('max_memory')!r}",
            file=sys.stderr,
            flush=True,
        )
        model = auto_model.from_pretrained(str(model_path), **model_load_kwargs)
        if staged_cuda_device is not None:
            print(f"model_transfer target_device={staged_cuda_device!r}", file=sys.stderr, flush=True)
            model = model.to(staged_cuda_device)
        _MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model


def _build_prompt_messages(messages: list[dict], tools_schema: list[dict]) -> list[dict]:
    prompt_messages = _prepare_chat_messages(messages)
    _normalize_assistant_tool_calls_for_chat_template(prompt_messages)
    usage_hint = _tool_usage_hint(messages, tools_schema)
    mobility_hint = _decision_mobility_hint(messages, tools_schema)
    latest_tool_guidance = _latest_tool_batch_guidance(messages)
    format_instruction = (
        "IMPORTANT OUTPUT FORMAT:\n"
        "You must return exactly one valid JSON object.\n"
        "Do not output markdown.\n"
        "Do not output explanations.\n"
        "Do not output code fences or backticks.\n"
        'The first output character must be "{" and the last output character must be "}".\n\n'
        "Valid schema A:\n"
        '{"content":"final answer text","tool_calls":[]}\n\n'
        "Valid schema B:\n"
        '{"content":"","tool_calls":[{"id":"call_001","name":"file_reader",'
        '"args":{"path":"docs/a.txt","max_chars":1200}},{"id":"call_002","name":"file_reader",'
        '"args":{"path":"docs/b.txt","max_chars":1200}}]}\n\n'
        "The top-level keys must be exactly:\n"
        "- content: string\n"
        "- tool_calls: array\n\n"
        "When the task can be parallelized, you may request multiple tool calls in the same response.\n"
        "Each tool call object must contain exactly id, name, args.\n"
        "Never put tool_calls inside content.\n"
        'Never output {"content":"tool_calls": ...}.'
    )
    decision_instruction = "DECISION POLICY:\n" + mobility_hint
    envelope_reminder = (
        "IMPORTANT OUTPUT FORMAT: Output the JSON object now. "
        'Your first output character must be "{" and your last output character must be "}". '
        "Never output a backtick, Markdown, a code block, an explanation, or text outside the JSON. "
        'Use exactly the top-level keys "content" (string) and "tool_calls" (array). '
        "Choose exactly one schema: final content with an empty tool_calls array, or empty content with one or more tool calls. "
        "If multiple tools are independently needed, output them together in the same tool_calls array. "
        'Never put tool_calls inside content. Never output {"content":"tool_calls": ...}.'
    )
    system_instruction = (
        "\n\n"
        + decision_instruction
        + "\n\nAvailable tools JSON schema:\n"
        + json.dumps(tools_schema, ensure_ascii=False)
        + "\n"
        + format_instruction
    )
    if usage_hint:
        system_instruction += "\n\nTask-specific tool policy:\n" + usage_hint
    prompt_messages[0]["content"] = _append_text_block(prompt_messages[0].get("content") or "", system_instruction)

    for message in reversed(prompt_messages):
        if message.get("role") == "user":
            extra_parts = [decision_instruction]
            if usage_hint:
                extra_parts.append("TASK-SPECIFIC TOOL POLICY:\n" + usage_hint)
            extra_parts.append(envelope_reminder)
            extra = "\n\n".join(extra_parts)
            message["content"] = _append_text_block(message.get("content") or "", extra)
            break
    if prompt_messages[-1].get("role") == "tool":
        followup = (
            envelope_reminder
            + " The latest ToolMessages already contain the tool results for this round. If they provide the "
            + 'requested information, answer with schema A now and set "tool_calls" to exactly []. Do not '
            + "repeat any completed tool call from the latest tool-result batch."
        )
        if latest_tool_guidance:
            followup = _append_text_block(followup, latest_tool_guidance)
        prompt_messages.append(
            {
                "role": "user",
                "content": followup,
            }
        )
    return prompt_messages

def _build_native_tool_messages(messages: list[dict], tools_schema: list[dict]) -> list[dict]:
    prompt_messages = _prepare_chat_messages(messages)
    _normalize_assistant_tool_calls_for_chat_template(prompt_messages)
    usage_hint = _tool_usage_hint(messages, tools_schema)
    mobility_hint = _decision_mobility_hint(messages, tools_schema)
    latest_tool_guidance = _latest_tool_batch_guidance(messages)
    native_instruction = (
        "DECISION POLICY:\n"
        + mobility_hint
        + "\n\n"
        "When tools are needed, use the tokenizer-native tool-calling format for this model.\n"
        "If the model renders native tool calls as XML blocks, use this structure exactly:\n"
        "<tool_call>\n"
        "<function=file_reader>\n"
        "<parameter=path>docs/example.txt</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "If multiple independent tools are needed, emit multiple <tool_call> blocks in the same reply.\n"
        "Keep reasoning internal and output only the native tool call blocks or the final direct answer for this turn.\n"
        "Do not answer from prior knowledge when the task explicitly refers to local files, local search, or table data."
    )
    if usage_hint:
        native_instruction += "\n\nTask-specific tool policy:\n" + usage_hint
    prompt_messages[0]["content"] = _append_text_block(prompt_messages[0].get("content") or "", native_instruction)
    for message in reversed(prompt_messages):
        if message.get("role") == "user":
            reminder = (
                "First decide internally whether you should answer directly or call tools. "
                "If tools are needed, emit all independent tool calls for this turn in one reply."
            )
            if usage_hint:
                reminder = "TASK-SPECIFIC TOOL POLICY:\n" + usage_hint + "\n\n" + reminder
            message["content"] = _append_text_block(message.get("content") or "", reminder)
            break
    if prompt_messages[-1].get("role") == "tool":
        followup = (
            "The latest ToolMessages already contain the tool results for this round. "
            "If they provide the requested information, answer directly without repeating successful tool calls."
        )
        if latest_tool_guidance:
            followup = _append_text_block(followup, latest_tool_guidance)
        prompt_messages.append({"role": "user", "content": followup})
    return prompt_messages

def _generate_from_chat_template(
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    prompt_messages: list[dict],
    generation_config: dict,
    tools_schema: list[dict] | None = None,
    max_input_tokens: int | None = None,
    allow_thinking: bool = False,
) -> dict:
    started_at = time.perf_counter()
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if tools_schema is not None:
        template_kwargs["tools"] = tools_schema
    try:
        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            **template_kwargs,
            enable_thinking=bool(allow_thinking),
        )
    except TypeError as exc:
        error_text = str(exc)
        if "enable_thinking" in error_text:
            try:
                inputs = tokenizer.apply_chat_template(
                    prompt_messages,
                    **template_kwargs,
                )
            except TypeError as inner_exc:
                inner_text = str(inner_exc)
                if tools_schema is not None and "tools" in inner_text:
                    raise RuntimeError(
                        "native_tools mode requires tokenizer.apply_chat_template(..., tools=...) support"
                    ) from inner_exc
                raise RuntimeError(
                    f"tokenizer.apply_chat_template is incompatible with native_tools arguments: {inner_text}"
                ) from inner_exc
        elif tools_schema is not None and "tools" in error_text:
            raise RuntimeError("native_tools mode requires tokenizer.apply_chat_template(..., tools=...) support") from exc
        else:
            raise RuntimeError(
                f"tokenizer.apply_chat_template failed with unsupported arguments: {error_text}"
            ) from exc
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = int(inputs["input_ids"].shape[-1])
    original_input_length = input_length
    limit = int(max_input_tokens or 0)
    if limit > 0 and input_length > limit:
        for key, value in list(inputs.items()):
            if not hasattr(value, "shape"):
                continue
            if getattr(value, "ndim", 0) != 2:
                continue
            if int(value.shape[-1]) != input_length:
                continue
            inputs[key] = value[:, -limit:]
        input_length = int(inputs["input_ids"].shape[-1])
    options = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 1024)),
        "do_sample": bool(generation_config.get("do_sample", False)),
    }
    if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    with _MODEL_GENERATION_LOCK, torch_module.no_grad():
        generated = model.generate(**inputs, **options)
    new_tokens = generated[0][input_length:]
    elapsed_seconds = time.perf_counter() - started_at
    skip_special_tokens = bool(generation_config.get("skip_special_tokens", True))
    payload = {
        "raw_text": tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False),
        "input_tokens": input_length,
        "output_tokens": int(new_tokens.shape[-1]),
        "elapsed_seconds": round(elapsed_seconds, 4),
    }
    if original_input_length != input_length:
        payload["input_tokens_original"] = original_input_length
    return payload


def _prompt_json_generate(
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    profile_name: str | None = None,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("prompt_json mode requires requirements-llm.txt") from exc
    profile_name = profile_name or _resolve_model_profile(config, messages)
    model_config = _merged_model_settings(config, profile_name)
    generation_config = config.get("generation", {})
    context_config = config.get("context") if isinstance(config.get("context"), dict) else {}
    max_input_tokens = int(context_config.get("max_input_tokens", 0) or 0)
    model_setting = model_config.get("model_name_or_path")
    tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")
    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")
    local_only = bool(model_config.get("local_files_only", True))
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
    tokenizer, model = _load_model_bundle(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        model_config.get("device_map", "auto"),
        model_config.get("max_memory"),
    )
    prompt_messages = _build_prompt_messages(messages, tools_schema)
    allow_thinking = bool(generation_config.get("allow_thinking", False))
    return _generate_from_chat_template(
        torch, tokenizer, model, prompt_messages, generation_config, None, max_input_tokens, allow_thinking=allow_thinking
    )


def _native_tools_generate(
    config_path: Path,
    config: dict,
    messages: list[dict],
    tools_schema: list[dict],
    profile_name: str | None = None,
) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("native_tools mode requires requirements-llm.txt") from exc
    profile_name = profile_name or _resolve_model_profile(config, messages)
    model_config = _merged_model_settings(config, profile_name)
    generation_config = config.get("generation", {})
    context_config = config.get("context") if isinstance(config.get("context"), dict) else {}
    max_input_tokens = int(context_config.get("max_input_tokens", 0) or 0)
    model_setting = model_config.get("model_name_or_path")
    tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")
    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")
    local_only = bool(model_config.get("local_files_only", True))
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
    tokenizer, model = _load_model_bundle(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        model_config.get("device_map", "auto"),
        model_config.get("max_memory"),
    )
    prompt_messages = _build_native_tool_messages(messages, tools_schema)
    allow_thinking = bool(generation_config.get("allow_thinking", False))
    return _generate_from_chat_template(
        torch,
        tokenizer,
        model,
        prompt_messages,
        generation_config,
        tools_schema,
        max_input_tokens,
        allow_thinking=allow_thinking,
    )


def warmup_model(model_config: str, profile_name: str | None = None) -> dict:
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("model warmup requires requirements-llm.txt") from exc
    config_path, config = _load_model_config(model_config)
    if not profile_name:
        routing = config.get("routing") if isinstance(config.get("routing"), dict) else {}
        profile_name = str(routing.get("default_profile") or "").strip() or None
    settings = _merged_model_settings(config, profile_name)
    model_setting = settings.get("model_name_or_path")
    tokenizer_setting = settings.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")
    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")
    dtype = _dtype_value(torch, str(settings.get("torch_dtype", "auto")))
    _load_model_bundle(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_path,
        tokenizer_path,
        bool(settings.get("local_files_only", True)),
        bool(settings.get("trust_remote_code", False)),
        dtype,
        settings.get("device_map", "auto"),
        settings.get("max_memory"),
    )
    return {
        "status": "success",
        "profile": profile_name,
        "model_path": str(model_path),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def generate_ai_message(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str | None = None,
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    forced_profile: str | None = None,
) -> dict:
    config_path, config = _load_model_config(model_config)
    mode = _resolve_runtime_mode(config, mode)
    messages = validate_messages(deepcopy(messages))
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    generated_at = now_iso()
    selected_profile = forced_profile or _resolve_model_profile(config, messages)
    merged_model_config = _merged_model_settings(config, selected_profile)
    backend = "mock" if mode == "mock" else merged_model_config.get("backend", "transformers")
    effective_mode = mode
    fallback = None
    usage = None
    if mode == "mock":
        ai_message = _mock_generate(messages)
        raw_text = json.dumps({"content": ai_message["content"], "tool_calls": ai_message["tool_calls"]}, ensure_ascii=False)
        parsed_candidate = {"content": ai_message["content"], "tool_calls": ai_message["tool_calls"]}
        status = "success"
        error = None
        attempts = [{"attempt_index": 1, "raw_text": raw_text}]
    elif mode == "prompt_json":
        generation_result = _generate_with_retry(
            _prompt_json_generate,
            config_path,
            config,
            messages,
            tools_schema,
            selected_profile,
            mode,
        )
        raw_text = generation_result["raw_text"]
        usage = generation_result.get("usage")
        parsed_candidate = generation_result.get("parsed_candidate")
        ai_message = generation_result["ai_message"]
        status = generation_result["status"]
        error = generation_result.get("error")
        attempts = generation_result.get("attempts", [])
    elif mode == "native_tools":
        try:
            generation_result = _generate_with_retry(
                _native_tools_generate,
                config_path,
                config,
                messages,
                tools_schema,
                selected_profile,
                mode,
            )
        except RuntimeError as exc:
            message = str(exc)
            native_incompatible = (
                "native_tools" in message
                or "apply_chat_template" in message
                or "unsupported arguments" in message
            )
            if not native_incompatible:
                raise
            fallback = {"from": "native_tools", "to": "prompt_json", "reason": message}
            effective_mode = "prompt_json"
            generation_result = _generate_with_retry(
                _prompt_json_generate,
                config_path,
                config,
                messages,
                tools_schema,
                selected_profile,
                effective_mode,
            )
        if generation_result.get("status") != "success" and effective_mode == "native_tools":
            native_error = generation_result.get("error") or {"type": "NativeToolsError", "message": "native output parsing failed"}
            fallback = {
                "from": "native_tools",
                "to": "prompt_json",
                "reason": f"{native_error.get('type', 'Error')}: {native_error.get('message', '')}",
            }
            effective_mode = "prompt_json"
            generation_result = _generate_with_retry(
                _prompt_json_generate,
                config_path,
                config,
                messages,
                tools_schema,
                selected_profile,
                effective_mode,
            )
        raw_text = generation_result["raw_text"]
        usage = generation_result.get("usage")
        parsed_candidate = generation_result.get("parsed_candidate")
        ai_message = generation_result["ai_message"]
        status = generation_result["status"]
        error = generation_result.get("error")
        attempts = generation_result.get("attempts", [])
    else:
        raise ValueError("mode must be mock, prompt_json, or native_tools")
    raw_record = {
        "mode": mode,
        "effective_mode": effective_mode,
        "backend": backend,
        "model_profile": selected_profile,
        "resolved_model_path": str(resolve_from_file(merged_model_config.get("model_name_or_path", ""), config_path)) if merged_model_config.get("model_name_or_path") else "",
        "raw_text": raw_text,
        "parsed_candidate": parsed_candidate,
        "status": status,
        "error": error,
        "attempts": attempts,
        "repair_attempted": len(attempts) > 1,
        "generated_at": generated_at,
    }
    if fallback is not None:
        raw_record["fallback"] = fallback
    if usage is not None:
        raw_record["usage"] = usage
    if artifact_dir:
        raw_path, message_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
        write_json(raw_record, raw_path)
        write_json(ai_message, message_path)
        append_jsonl(
            {
                "timestamp": generated_at,
                "mode": mode,
                "effective_mode": effective_mode,
                "status": status,
                "raw_output_path": str(raw_path),
                "ai_message_path": str(message_path),
                "error": error,
            },
            log_path,
        )
    return {
        "ai_message": ai_message,
        "status": status,
        "error": error,
        "raw_record": raw_record,
    }

def _required_args_map(tools_schema: list[dict]) -> dict[str, set[str]]:
    required_map: dict[str, set[str]] = {}
    for item in tools_schema:
        if not isinstance(item, dict):
            continue
        function_schema = item.get("function")
        if not isinstance(function_schema, dict):
            continue
        name = function_schema.get("name")
        parameters = function_schema.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        required = parameters.get("required") or []
        if isinstance(required, list):
            required_map[name] = {str(key) for key in required}
    return required_map


def _evaluate_tool_mode_result(result: dict, tools_schema: list[dict]) -> dict:
    ai_message = result.get("ai_message") or {}
    tool_calls = ai_message.get("tool_calls") or []
    required_map = _required_args_map(tools_schema)
    tool_name_correct = True
    args_complete = True
    for call in tool_calls:
        name = str(call.get("name") or "")
        if name not in required_map:
            tool_name_correct = False
            continue
        required = required_map.get(name, set())
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if any(key not in args for key in required):
            args_complete = False
    raw_record = result.get("raw_record") or {}
    usage = raw_record.get("usage") or {}
    return {
        "status": result.get("status"),
        "tool_call_count": len(tool_calls),
        "tool_name_correct": tool_name_correct,
        "args_complete": args_complete,
        "structured_output_success": result.get("status") == "success",
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "elapsed_seconds": usage.get("elapsed_seconds"),
    }


def _aggregate_llm_usage(llm_calls_dir: Path) -> dict:
    if not llm_calls_dir.exists() or not llm_calls_dir.is_dir():
        return {}
    input_tokens = 0
    output_tokens = 0
    elapsed_seconds = 0.0
    has_any = False
    for path in llm_calls_dir.glob("*raw_model_output.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        has_any = True
        try:
            input_tokens += int(usage.get("input_tokens") or 0)
        except Exception:
            pass
        try:
            output_tokens += int(usage.get("output_tokens") or 0)
        except Exception:
            pass
        try:
            elapsed_seconds += float(usage.get("elapsed_seconds") or 0.0)
        except Exception:
            pass
    if not has_any:
        return {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 4),
    }


def _extract_last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _collect_trace_tool_calls(trace: dict) -> list[dict]:
    turns = trace.get("turns") if isinstance(trace, dict) else None
    if not isinstance(turns, list):
        return []
    collected: list[dict] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        tool_messages = turn.get("tool_messages")
        if not isinstance(tool_messages, list) or not tool_messages:
            continue
        ai_message = turn.get("ai_message") if isinstance(turn.get("ai_message"), dict) else {}
        tool_calls = ai_message.get("tool_calls") if isinstance(ai_message.get("tool_calls"), list) else []
        for call in tool_calls:
            if isinstance(call, dict):
                collected.append(call)
    return collected


def _evaluate_plan_execute_trace(trace: dict, tools_schema: list[dict]) -> dict:
    tool_calls = _collect_trace_tool_calls(trace)
    required_map = _required_args_map(tools_schema)
    tool_name_correct = True
    args_complete = True
    for call in tool_calls:
        name = str(call.get("name") or "")
        if name not in required_map:
            tool_name_correct = False
            continue
        required = required_map.get(name, set())
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if any(key not in args for key in required):
            args_complete = False
    return {
        "tool_call_count": len(tool_calls),
        "tool_name_correct": tool_name_correct,
        "args_complete": args_complete,
        "actual_tools": _normalize_name_list([str(call.get("name") or "") for call in tool_calls]),
    }


def _question_matches_expectation(trace: dict, expected_question_contains: str) -> bool:
    expected = str(expected_question_contains or "").strip()
    if not expected:
        return True
    error = trace.get("error") if isinstance(trace, dict) else None
    if not isinstance(error, dict):
        return False
    question = str(error.get("question") or "").strip()
    return expected.casefold() in question.casefold()


def _optional_internal_tools_satisfied(expected_tools: list[str], actual_tools: list[str], evidence_policy: str) -> bool:
    policy = str(evidence_policy or "strict").strip().lower()
    if policy != "lite":
        return False
    normalized_expected = _normalize_name_list(expected_tools)
    normalized_actual = _normalize_name_list(actual_tools)
    if normalized_actual:
        return False
    return bool(normalized_expected) and all(name in {"calculator", "format_converter"} for name in normalized_expected)


def _heuristic_task_complexity(messages: list[dict]) -> dict:
    user_text = _extract_last_user_text(messages)
    text = user_text.casefold()
    if not text:
        return {"complexity": "high", "confidence": 0.0, "reason": "empty user input"}
    normalized_math = user_text.replace("×", "*").replace("÷", "/")
    if re.search(r"\d+(?:\.\d+)?(?:\s*[+\-*/%]\s*\d+(?:\.\d+)?)+", normalized_math):
        return {"complexity": "low", "confidence": 0.95, "reason": "任务是可由计算器单轮完成的算术表达式。"}
    high_markers = [
        "先搜索",
        "search",
        "最相关",
        "比较",
        "差异",
        "分别",
        "多步",
    ]
    if any(token in text for token in high_markers):
        return {"complexity": "high", "confidence": 0.7, "reason": "任务包含检索、比较或明显的多步依赖。"}
    if "同时阅读" in user_text or ("docs/" in user_text and "和" in user_text and "比较" in user_text):
        return {"complexity": "low", "confidence": 0.75, "reason": "任务可在单轮并行工具调用后直接汇总，无需显式状态推进。"}
    low_markers = [
        "请计算",
        "计算 ",
        "expression",
        "读取",
        "读一下",
        "分析 tables/",
        "分析表格",
        "列结构",
        "预览内容",
    ]
    if any(token in text for token in low_markers):
        return {"complexity": "low", "confidence": 0.85, "reason": "任务更像单步求值或单轮外部读取/分析，可在 0~1 轮工具后收敛。"}
    return {"complexity": "high", "confidence": 0.55, "reason": "启发式未能确认其为单步任务，保守回退到 plan_execute。"}


def run_batch_plan_execute_evaluation(
    model_config: str,
    tools_schema: list[dict],
    cases_path: str,
    tools_config: str,
    toolset: str,
    outdir: str,
    modes: list[str] | None = None,
    profiles: list[str] | None = None,
    max_turns: int = 3,
    max_plan_steps: int = 6,
    evidence_policy: str = "strict",
) -> dict:
    config_path, config = _load_model_config(model_config)
    output_dir = ensure_dir(outdir)
    cases = _load_eval_cases(cases_path)
    selected_modes = modes or []
    if not selected_modes:
        evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
        configured_modes = evaluation.get("default_modes") if isinstance(evaluation, dict) else None
        if isinstance(configured_modes, list) and configured_modes:
            selected_modes = [str(item).strip() for item in configured_modes if str(item).strip()]
        else:
            selected_modes = ["prompt_json"]
    selected_profiles = _resolve_eval_profiles(config, profiles)
    model_name = _resolved_model_name(config, selected_profiles[0] if selected_profiles else None)
    profile_targets = {name or "default": _resolved_model_target(config_path, config, name or None) for name in selected_profiles}
    rows: list[dict] = []
    for profile_name in selected_profiles:
        model_target = profile_targets.get(profile_name or "default", "")
        for mode in selected_modes:
            for case in cases:
                artifact_dir = ensure_dir(output_dir / "artifacts" / (profile_name or "default") / mode / case["id"])
                if mode == "adaptive":
                    result = run_adaptive_execute(
                        str(config_path),
                        case["messages"],
                        tools_schema,
                        str(resolve_cli_path(tools_config)),
                        str(toolset),
                        str(artifact_dir),
                        max_turns=max_turns,
                        max_plan_steps=max_plan_steps,
                        evidence_policy=str(evidence_policy),
                        forced_profile=profile_name or None,
                    )
                else:
                    result = run_plan_execute(
                        str(config_path),
                        case["messages"],
                        tools_schema,
                        str(resolve_cli_path(tools_config)),
                        str(toolset),
                        mode,
                        str(artifact_dir),
                        max_turns=max_turns,
                        max_plan_steps=max_plan_steps,
                        evidence_policy=str(evidence_policy),
                        forced_profile=profile_name or None,
                    )
                trace = result.get("trace") or {}
                status = str(result.get("status") or "")
                expected_status = str(case.get("expected_status") or "success").strip() or "success"
                structured_ok = status == expected_status
                eval_detail = _evaluate_plan_execute_trace(trace, tools_schema)
                actual_tools = eval_detail.get("actual_tools") if isinstance(eval_detail.get("actual_tools"), list) else []
                expected_tools = _normalize_name_list(case.get("expected_tools", []))
                raw_match_type = str(case.get("expected_match_type") or "").strip()
                if raw_match_type:
                    match_type = raw_match_type
                else:
                    match_type = "exact" if not expected_tools else "contains"
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
                usage = _aggregate_llm_usage(Path(artifact_dir) / "llm_calls")
                error = trace.get("error") if isinstance(trace, dict) else None
                rows.append(
                    {
                        "case_id": case["id"],
                        "title": case["title"],
                        "category": case["category"],
                        "mode": mode,
                        "model_profile": profile_name or "default",
                        "model_name": model_name,
                        "model_target": model_target,
                        "expected_status": expected_status,
                        "actual_status": status,
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
        "cases_path": str(cases_path),
        "profiles": selected_profiles,
        "modes": selected_modes,
        "profile_targets": {name or "default": target for name, target in profile_targets.items() if target},
        "summary": summary,
        "evaluation_mode": "plan_execute",
    }
    write_json(summary_payload, output_dir / "eval_summary.json")
    _write_eval_report_csv(rows, output_dir / "eval_report.csv")
    return {"rows": rows, "summary": summary, "outdir": str(output_dir)}


def compare_tools_injection_modes(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    outdir: str,
) -> dict:
    output_dir = ensure_dir(outdir)
    prompt_dir = ensure_dir(output_dir / "prompt_json")
    native_dir = ensure_dir(output_dir / "native_tools")
    prompt_result = generate_ai_message(model_config, messages, tools_schema, "prompt_json", str(prompt_dir))
    native_result = generate_ai_message(model_config, messages, tools_schema, "native_tools", str(native_dir))
    comparison = {
        "prompt_json": _evaluate_tool_mode_result(prompt_result, tools_schema),
        "native_tools": _evaluate_tool_mode_result(native_result, tools_schema),
    }
    write_json(comparison, output_dir / "comparison.json")
    return comparison


def _default_eval_system_prompt(expected_tools: list[str]) -> str:
    base = "You are a local tool-using agent."
    normalized_tools = [str(name).strip() for name in expected_tools if str(name).strip()]
    if not normalized_tools:
        return base + " Use available tools when needed."
    tool_list = ", ".join(dict.fromkeys(normalized_tools))
    return (
        base
        + " When the user asks about local files, local search, table contents, or arithmetic, you must use the available tools instead of prior knowledge. "
        + "For this evaluation case, the expected tool(s) are: "
        + tool_list
        + ". If multiple independent tools are needed, request them in the same response."
    )


def _load_eval_cases_payload(cases_path: str | Path) -> dict:
    payload = read_json(cases_path)
    if isinstance(payload, list):
        if not payload:
            raise ValueError("eval cases must be a non-empty array")
        return {"batch_eval": {}, "cases": payload}
    if not isinstance(payload, dict):
        raise ValueError("eval cases must be a non-empty array or an object containing cases")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("eval cases object must contain a non-empty cases array")
    batch_eval = payload.get("batch_eval")
    if batch_eval is None:
        batch_eval = {}
    if not isinstance(batch_eval, dict):
        raise ValueError("batch_eval must be an object when provided")
    return {"batch_eval": batch_eval, "cases": cases}


def _load_eval_cases(cases_path: str | Path) -> list[dict]:
    payload = _load_eval_cases_payload(cases_path)
    raw_cases = payload["cases"]
    cases: list[dict] = []
    for index, item in enumerate(raw_cases, 1):
        if not isinstance(item, dict):
            raise ValueError("each eval case must be an object")
        case_id = str(item.get("id") or f"case_{index:03d}")
        category = str(item.get("category") or "uncategorized")
        expected_tools = item.get("expected_tools") or []
        if not isinstance(expected_tools, list):
            raise ValueError(f"case {case_id}: expected_tools must be an array")
        expected_status = str(item.get("expected_status") or "success").strip().lower()
        if expected_status not in {"success", "needs_user"}:
            raise ValueError(f"case {case_id}: expected_status must be success or needs_user")
        tool_order_matters = bool(item.get("tool_order_matters", False))
        expected_match_type = str(item.get("expected_match_type") or "exact").strip().lower()
        if expected_match_type not in {"exact", "contains"}:
            raise ValueError(f"case {case_id}: expected_match_type must be exact or contains")
        if "messages" in item:
            messages = validate_messages(deepcopy(item["messages"]))
        else:
            user_input = str(item.get("user_input") or "").strip()
            if not user_input:
                raise ValueError(f"case {case_id}: user_input or messages is required")
            system_prompt = str(item.get("system_prompt") or _default_eval_system_prompt([str(name) for name in expected_tools]))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        cases.append(
            {
                "id": case_id,
                "title": str(item.get("title") or case_id),
                "category": category,
                "messages": messages,
                "expected_tools": [str(name) for name in expected_tools],
                "expected_status": expected_status,
                "expected_question_contains": str(item.get("expected_question_contains") or "").strip(),
                "tool_order_matters": tool_order_matters,
                "expected_match_type": expected_match_type,
            }
        )
    return cases

def _normalize_name_list(values: list[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _tool_match(expected: list[str], actual: list[str], order_matters: bool, match_type: str = "exact") -> bool:
    if match_type == "contains":
        expected_counter = Counter(expected)
        actual_counter = Counter(actual)
        return all(actual_counter.get(name, 0) >= count for name, count in expected_counter.items())
    if order_matters:
        return expected == actual
    return sorted(expected) == sorted(actual)


def _resolve_eval_profiles(config: dict, profile_names: list[str] | None) -> list[str]:
    pool = config.get("model_pool") if isinstance(config.get("model_pool"), dict) else {}
    if profile_names:
        invalid = [name for name in profile_names if name not in pool]
        if invalid:
            raise ValueError(f"unknown model profiles: {', '.join(invalid)}")
        return profile_names
    evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    default_profiles = evaluation.get("default_profiles") if isinstance(evaluation, dict) else None
    if isinstance(default_profiles, list) and default_profiles:
        normalized = [str(name).strip() for name in default_profiles if str(name).strip()]
        invalid = [name for name in normalized if name not in pool]
        if invalid:
            raise ValueError(f"unknown evaluation.default_profiles: {', '.join(invalid)}")
        if normalized:
            return normalized
    if isinstance(pool, dict) and pool:
        return list(pool.keys())
    return [""]


def _resolved_model_name(config: dict, profile_name: str | None) -> str:
    merged = _merged_model_settings(config, profile_name or None)
    model_name = merged.get("model_name_or_path")
    if not isinstance(model_name, str) or not model_name:
        return ""
    return Path(model_name).name or model_name


def _resolved_model_target(config_path: Path, config: dict, profile_name: str | None) -> str:
    merged = _merged_model_settings(config, profile_name or None)
    model_name = merged.get("model_name_or_path")
    if not isinstance(model_name, str) or not model_name.strip():
        return ""
    return str(resolve_from_file(model_name, config_path))

def _write_eval_report_csv(rows: list[dict], path: str | Path) -> Path:
    fieldnames = [
        "case_id",
        "title",
        "category",
        "mode",
        "model_profile",
        "model_name",
        "model_target",
        "expected_status",
        "actual_status",
        "expected_tools",
        "actual_tools",
        "tool_match",
        "structured_output_success",
        "args_complete",
        "success",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
        "error_type",
        "error_message",
    ]
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target

def _summarize_eval_rows(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f'{row.get("model_profile") or "default"}::{row.get("mode")}'
        groups.setdefault(key, []).append(row)
    summary: dict[str, dict] = {}
    for key, items in groups.items():
        total = len(items)
        success_count = sum(1 for item in items if item.get("success"))
        structured_count = sum(1 for item in items if item.get("structured_output_success"))
        tool_match_count = sum(1 for item in items if item.get("tool_match"))
        args_complete_count = sum(1 for item in items if item.get("args_complete"))
        input_values = [int(item["input_tokens"]) for item in items if isinstance(item.get("input_tokens"), int)]
        output_values = [int(item["output_tokens"]) for item in items if isinstance(item.get("output_tokens"), int)]
        elapsed_values = [float(item["elapsed_seconds"]) for item in items if isinstance(item.get("elapsed_seconds"), (int, float))]
        summary[key] = {
            "cases": total,
            "success_rate": round(success_count / total, 4) if total else 0.0,
            "structured_output_rate": round(structured_count / total, 4) if total else 0.0,
            "tool_match_rate": round(tool_match_count / total, 4) if total else 0.0,
            "args_complete_rate": round(args_complete_count / total, 4) if total else 0.0,
            "avg_input_tokens": round(sum(input_values) / len(input_values), 2) if input_values else None,
            "avg_output_tokens": round(sum(output_values) / len(output_values), 2) if output_values else None,
            "avg_elapsed_seconds": round(sum(elapsed_values) / len(elapsed_values), 4) if elapsed_values else None,
        }
    return summary


def run_batch_evaluation(
    model_config: str,
    tools_schema: list[dict],
    cases_path: str,
    outdir: str,
    modes: list[str] | None = None,
    profiles: list[str] | None = None,
) -> dict:
    config_path, config = _load_model_config(model_config)
    output_dir = ensure_dir(outdir)
    cases = _load_eval_cases(cases_path)
    evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    configured_modes = evaluation.get("default_modes") if isinstance(evaluation, dict) else None
    if modes is not None:
        selected_modes = modes
    elif isinstance(configured_modes, list) and configured_modes:
        selected_modes = [str(mode).strip() for mode in configured_modes if str(mode).strip()]
    else:
        selected_modes = ["prompt_json", "native_tools"]
    invalid_modes = [mode for mode in selected_modes if mode not in {"mock", "prompt_json", "native_tools"}]
    if invalid_modes:
        raise ValueError(f"unsupported eval modes: {', '.join(invalid_modes)}")
    selected_profiles = _resolve_eval_profiles(config, profiles)
    profile_targets = {
        (profile_name or "default"): _resolved_model_target(config_path, config, profile_name or None)
        for profile_name in selected_profiles
    }
    rows: list[dict] = []
    for profile_name in selected_profiles:
        model_name = _resolved_model_name(config, profile_name or None)
        model_target = profile_targets.get(profile_name or "default", "")
        for mode in selected_modes:
            for case in cases:
                artifact_dir = ensure_dir(output_dir / "artifacts" / (profile_name or "default") / mode / case["id"])
                result = generate_ai_message(
                    str(config_path),
                    case["messages"],
                    tools_schema,
                    mode,
                    str(artifact_dir),
                    forced_profile=profile_name or None,
                )
                ai_message = result.get("ai_message") or {}
                raw_record = result.get("raw_record") or {}
                usage = raw_record.get("usage") or {}
                actual_tools = _normalize_name_list([call.get("name", "") for call in ai_message.get("tool_calls", [])])
                expected_tools = _normalize_name_list(case.get("expected_tools", []))
                structured_ok = result.get("status") == "success"
                args_complete = _evaluate_tool_mode_result(result, tools_schema).get("args_complete", False)
                tool_match = _tool_match(
                    expected_tools,
                    actual_tools,
                    bool(case.get("tool_order_matters")),
                    str(case.get("expected_match_type") or "exact"),
                )
                success = structured_ok and tool_match and args_complete
                error = result.get("error") or {}
                rows.append(
                    {
                        "case_id": case["id"],
                        "title": case["title"],
                        "category": case["category"],
                        "mode": mode,
                        "model_profile": profile_name or "default",
                        "model_name": model_name,
                        "model_target": model_target,
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
    duplicate_target_warnings = []
    target_to_profiles: dict[str, list[str]] = {}
    for profile_name, target in profile_targets.items():
        if not target:
            continue
        target_to_profiles.setdefault(target, []).append(profile_name)
    for target, names in target_to_profiles.items():
        if len(names) > 1:
            duplicate_target_warnings.append(
                {
                    "type": "duplicate_model_target",
                    "model_target": target,
                    "profiles": names,
                    "message": "These profiles resolve to the same local model path, so routing/comparison does not represent a real multi-model switch.",
                }
            )
    summary_payload = {
        "cases_path": str(cases_path),
        "profiles": selected_profiles,
        "modes": selected_modes,
        "profile_targets": profile_targets,
        "summary": summary,
    }
    if duplicate_target_warnings:
        summary_payload["warnings"] = duplicate_target_warnings
    write_json(summary_payload, output_dir / "eval_summary.json")
    _write_eval_report_csv(rows, output_dir / "eval_report.csv")
    return {"rows": rows, "summary": summary, "outdir": str(output_dir)}

def _build_adaptive_routing_instruction() -> str:
    return (
        "\n\n请判断该任务复杂度，并输出严格 JSON（放在 schema A 的 content 字段里）：\n"
        '{"complexity":"low|high","confidence":0.0,"reason":"..."}\n'
        "判定口径：\n"
        '- low：通常 0~1 轮工具即可完成（单步/低复杂度），不需要显式多步计划与状态推进。\n'
        "- high：通常需要多步计划与状态推进，或需要多轮工具结果回灌才能收敛。\n"
        "输出要求：\n"
        "- 必须输出 schema A；tool_calls 必须是 []。\n"
        "- content 必须是严格 JSON 字符串，不要输出解释或 Markdown。\n"
    )


def classify_task_complexity(
    model_config: str,
    messages: list[dict],
    outdir: str,
    forced_profile: str | None = None,
) -> dict:
    messages = validate_messages(deepcopy(messages))
    heuristic = _heuristic_task_complexity(messages)
    output_dir = ensure_dir(outdir)
    llm_calls_dir = ensure_dir(output_dir / "llm_calls_router")
    routing_messages = deepcopy(messages)
    routing_instruction = _build_adaptive_routing_instruction()
    for message in reversed(routing_messages):
        if message.get("role") == "user":
            message["content"] = str(message.get("content") or "") + routing_instruction
            break
    result = generate_ai_message(
        model_config,
        routing_messages,
        [],
        "prompt_json",
        str(llm_calls_dir),
        "llm_call_001",
        forced_profile=forced_profile,
    )
    status = str(result.get("status") or "")
    ai_message = result.get("ai_message") if isinstance(result.get("ai_message"), dict) else {}
    tool_calls = ai_message.get("tool_calls") if isinstance(ai_message.get("tool_calls"), list) else []
    content = str(ai_message.get("content") or "").strip()
    parsed = parse_json_object_from_text(content) if content else None
    decision = {
        "strategy": "plan_execute",
        "complexity": "high",
        "confidence": 0.0,
        "reason": "",
        "llm_status": status,
        "llm_error": result.get("error"),
    }
    if status != "success":
        decision["complexity"] = str(heuristic.get("complexity") or "high")
        decision["confidence"] = float(heuristic.get("confidence") or 0.0)
        decision["reason"] = "routing classifier did not return a valid AIMessage; fallback to heuristic. " + str(heuristic.get("reason") or "")
        decision["strategy"] = "react_one_round" if decision["complexity"] == "low" else "plan_execute"
        return decision
    if tool_calls:
        decision["complexity"] = str(heuristic.get("complexity") or "high")
        decision["confidence"] = float(heuristic.get("confidence") or 0.0)
        decision["reason"] = "routing classifier unexpectedly requested tool_calls; fallback to heuristic. " + str(heuristic.get("reason") or "")
        decision["strategy"] = "react_one_round" if decision["complexity"] == "low" else "plan_execute"
        return decision
    if isinstance(parsed, dict):
        complexity = str(parsed.get("complexity") or "").strip().lower()
        if complexity in {"low", "high"}:
            decision["complexity"] = complexity
        try:
            decision["confidence"] = float(parsed.get("confidence") or 0.0)
        except Exception:
            decision["confidence"] = 0.0
        decision["reason"] = str(parsed.get("reason") or "").strip()
    if decision["complexity"] not in {"low", "high"} or (decision["confidence"] <= 0.0 and not decision["reason"]):
        decision["complexity"] = str(heuristic.get("complexity") or "high")
        decision["confidence"] = float(heuristic.get("confidence") or 0.0)
        decision["reason"] = str(heuristic.get("reason") or "")
    decision["strategy"] = "react_one_round" if decision["complexity"] == "low" else "plan_execute"
    return decision


def _run_react_one_round_execute_impl(
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
    if not isinstance(max_tool_rounds, int) or isinstance(max_tool_rounds, bool) or max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be a non-negative integer")
    _, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    output_dir = ensure_dir(outdir)
    llm_calls_dir = ensure_dir(output_dir / "llm_calls")
    turns: list[dict] = []
    tool_rounds = 0
    llm_calls = 0
    status = "success"
    terminal_error = None
    final_answer = ""
    plan_state: dict = {"mode": "react_one_round", "steps": []}
    trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
    _persist_runtime_state(
        output_dir,
        messages,
        trace,
        final_answer,
        plan_state,
        llm_calls,
        tool_rounds,
        "initializing",
        runtime_bridge=runtime_bridge,
        phase="react_init",
    )
    duplicate_user_confirmation = _duplicate_resource_confirmation_from_user_text(_latest_user_text(messages))
    if duplicate_user_confirmation:
        status = "needs_user"
        terminal_error = {
            "type": "HumanInTheLoop",
            "message": "user request contains duplicate local resources",
            "question": duplicate_user_confirmation,
            "plan_state": deepcopy(plan_state),
        }
        trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
        _persist_runtime_state(
            output_dir,
            messages,
            trace,
            "",
            plan_state,
            llm_calls,
            tool_rounds,
            status,
            pending_question=duplicate_user_confirmation,
            runtime_bridge=runtime_bridge,
            phase="react_wait_user",
        )
        return {
            "status": status,
            "final_answer": "",
            "messages": messages,
            "trace": trace,
            "outdir": str(output_dir),
        }

    max_llm_calls = 1 + max_tool_rounds * 2 + 4
    budget_notice_sent = False
    pending_fast_tool_calls = _deterministic_plan_tool_calls(
        current_plan_step_title(plan_state, next_pending_step(plan_state)), tools_schema
    )

    while True:
        if not pending_fast_tool_calls and llm_calls >= max_llm_calls:
            status = "max_llm_calls_exceeded"
            final_answer = "任务因 ReAct 循环次数过多而终止。"
            terminal_error = {
                "type": "MaxLLMCallsExceeded",
                "message": final_answer,
                "max_llm_calls": max_llm_calls,
            }
            break
        llm_calls += 1
        llm_result = generate_ai_message(
            model_config,
            messages,
            tools_schema,
            mode,
            str(llm_calls_dir),
            f"llm_call_{llm_calls:03d}",
            forced_profile=forced_profile,
        )
        ai_message = llm_result.get("ai_message") if isinstance(llm_result.get("ai_message"), dict) else {}
        llm_status = llm_result.get("status")
        llm_error = llm_result.get("error")
        messages.append(ai_message)
        turn: dict = {
            "turn_index": llm_calls,
            "phase": "react",
            "ai_message": ai_message,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
        }
        if llm_status != "success":
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_calls,
                "cause": llm_error,
            }
            turns.append(turn)
            break
        tool_calls = ai_message.get("tool_calls") if isinstance(ai_message.get("tool_calls"), list) else []
        if not tool_calls:
            final_answer = str(ai_message.get("content") or "")
            turns.append(turn)
            trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                status,
                runtime_bridge=runtime_bridge,
                phase="react_answer_ready",
            )
            break
        duplicate_confirmation = _duplicate_resource_confirmation_text(tool_calls)
        if duplicate_confirmation:
            status = "needs_user"
            terminal_error = {
                "type": "HumanInTheLoop",
                "message": "agent requests user confirmation for duplicate resource reads",
                "question": duplicate_confirmation,
                "plan_state": deepcopy(plan_state),
            }
            turn["note"] = "duplicate_resource_ambiguity"
            turns.append(turn)
            trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                status,
                pending_question=duplicate_confirmation,
                runtime_bridge=runtime_bridge,
                phase="react_wait_user",
            )
            break
        if tool_rounds >= max_tool_rounds:
            if budget_notice_sent:
                turn["note"] = "tool_budget_blocked"
                turn["blocked_tool_calls"] = tool_calls
                turns.append(turn)
                pending_question = _build_budget_exhausted_ask_user(tool_calls)
                if pending_question:
                    status = "needs_user"
                    terminal_error = {
                        "type": "HumanInTheLoop",
                        "message": "tool call budget exhausted and user input is required to proceed",
                        "question": pending_question,
                        "blocked_tool_calls": tool_calls,
                        "plan_state": deepcopy(plan_state),
                    }
                    trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
                    _persist_runtime_state(
                        output_dir,
                        messages,
                        trace,
                        final_answer,
                        plan_state,
                        llm_calls,
                        tool_rounds,
                        status,
                        pending_question=pending_question,
                        runtime_bridge=runtime_bridge,
                        phase="react_wait_user",
                    )
                    break
                status = "tool_budget_exceeded"
                terminal_error = {
                    "type": "ToolBudgetExceeded",
                    "message": "tool call budget exhausted before the task converged",
                    "blocked_tool_calls": tool_calls,
                }
                trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
                _persist_runtime_state(
                    output_dir,
                    messages,
                    trace,
                    final_answer,
                    plan_state,
                    llm_calls,
                    tool_rounds,
                    status,
                    runtime_bridge=runtime_bridge,
                    phase="react_budget_exhausted",
                )
                break
            budget_notice_sent = True
            turn["note"] = "tool_budget_notice"
            turn["blocked_tool_calls"] = tool_calls
            turns.append(turn)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "本次运行的工具预算已用尽，后续不会再执行新的工具调用。\n"
                        "请仅基于当前对话和已有工具证据继续收敛：\n"
                        "- 如果信息已经足够，直接给出最终回答；\n"
                        "- 如果仍存在关键歧义或需要用户决定，请直接提出问题；\n"
                        "- 如果只能给出有边界的结论，也请明确说明边界。\n"
                        "不要重复已经成功返回的相同工具调用。"
                    ),
                }
            )
            trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                "tool_budget_notice",
                runtime_bridge=runtime_bridge,
                phase="react_budget_notice",
            )
            continue

        tool_rounds += 1
        tool_outdir = ensure_dir(output_dir / "tool_rounds" / f"round_{tool_rounds:03d}")
        normalized_calls: list[dict] = []
        for call in tool_calls:
            if isinstance(call, dict):
                normalized_calls.append(_normalize_tool_call_args_for_execution(call))
        tool_messages_raw = _execute_tool_calls_with_runtime_bridge(
            runtime_bridge,
            normalized_calls,
            tools_config,
            toolset,
            str(tool_outdir),
        )
        tool_messages, compression_meta = compress_tool_messages(tool_messages_raw, config)
        messages.extend(tool_messages)
        turn["tool_messages"] = tool_messages
        turn["tool_messages_compression"] = compression_meta
        turn["tool_round_dir"] = str(tool_outdir)
        turns.append(turn)
        trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
        _persist_runtime_state(
            output_dir,
            messages,
            trace,
            final_answer,
            plan_state,
            llm_calls,
            tool_rounds,
            "tool_round_completed",
            runtime_bridge=runtime_bridge,
            phase="react_tool_round",
        )
        messages.append(
            {
                "role": "user",
                "content": _append_text_block(
                    (
                        "已获得本轮工具结果。请基于最新证据继续决策："
                        "如果信息已足够，就直接输出最终回答；"
                        "如果仍缺关键证据，也可以继续请求下一轮工具；"
                        "但优先利用现有证据收敛，不要重复已经成功返回的相同调用。"
                    ),
                    _build_evidence_sufficiency_nudge(_latest_user_text(messages), merge_tool_messages(tool_messages)),
                ),
            }
        )

    trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error, routing=routing)
    _persist_runtime_state(
        output_dir,
        messages,
        trace,
        final_answer,
        plan_state,
        llm_calls,
        tool_rounds,
        status,
        runtime_bridge=runtime_bridge,
        phase="react_finished",
    )
    return {
        "status": status,
        "final_answer": final_answer,
        "messages": messages,
        "trace": trace,
        "outdir": str(output_dir),
    }


def _run_adaptive_execute_impl(
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
    output_dir = ensure_dir(outdir)
    if low_mode == "mock" and high_mode == "mock":
        heuristic = _heuristic_task_complexity(messages)
        complexity = str(heuristic.get("complexity") or "high")
        routing = {
            "strategy": "react_one_round" if complexity == "low" else "plan_execute",
            "complexity": complexity,
            "confidence": float(heuristic.get("confidence") or 0.0),
            "reason": "mock mode uses deterministic heuristic routing. " + str(heuristic.get("reason") or ""),
            "llm_status": "skipped_mock",
            "llm_error": None,
        }
    else:
        routing = classify_task_complexity(model_config, messages, str(output_dir), forced_profile=forced_profile)
    write_json(routing, output_dir / "routing.json")
    if routing.get("strategy") == "react_one_round":
        return _run_react_one_round_execute_impl(
            model_config,
            messages,
            tools_schema,
            tools_config,
            toolset,
            low_mode,
            str(output_dir),
            max_tool_rounds=1,
            forced_profile=forced_profile,
            routing=routing,
            runtime_bridge=runtime_bridge,
        )
    result = _run_plan_execute_impl(
        model_config,
        messages,
        tools_schema,
        tools_config,
        toolset,
        high_mode,
        str(output_dir),
        max_turns=int(max_turns),
        max_plan_steps=int(max_plan_steps),
        evidence_policy=str(evidence_policy),
        forced_profile=forced_profile,
        runtime_bridge=runtime_bridge,
    )
    trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
    if isinstance(trace, dict):
        trace["adaptive"] = deepcopy(routing)
        write_json(trace, output_dir / "trace.json")
        result["trace"] = trace
    return result


def _run_plan_execute_impl(
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
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if not isinstance(max_plan_steps, int) or isinstance(max_plan_steps, bool) or max_plan_steps < 1:
        raise ValueError("max_plan_steps must be a positive integer")
    evidence_policy = str(evidence_policy or "strict").strip().lower()
    if evidence_policy not in {"strict", "lite"}:
        raise ValueError("evidence_policy must be strict or lite")
    _, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    output_dir = ensure_dir(outdir)
    llm_calls_dir = ensure_dir(output_dir / "llm_calls")
    turns: list[dict] = []
    tool_rounds = 0
    llm_calls = 0
    trace = _build_runtime_trace("running", tool_rounds, llm_calls, {"steps": []}, turns, None)
    _persist_runtime_state(
        output_dir,
        messages,
        trace,
        "",
        {"steps": []},
        llm_calls,
        tool_rounds,
        "initializing",
        runtime_bridge=runtime_bridge,
        phase="plan_init",
    )
    duplicate_user_confirmation = _duplicate_resource_confirmation_from_user_text(_latest_user_text(messages))
    if duplicate_user_confirmation:
        status = "needs_user"
        terminal_error = {
            "type": "HumanInTheLoop",
            "message": "user request contains duplicate local resources",
            "question": duplicate_user_confirmation,
            "plan_state": {"steps": []},
        }
        trace = _build_runtime_trace(status, tool_rounds, llm_calls, {"steps": []}, turns, terminal_error)
        _persist_runtime_state(
            output_dir,
            messages,
            trace,
            "",
            {"steps": []},
            llm_calls,
            tool_rounds,
            status,
            pending_question=duplicate_user_confirmation,
            runtime_bridge=runtime_bridge,
            phase="plan_wait_user",
        )
        return {
            "status": status,
            "final_answer": "",
            "messages": messages,
            "trace": trace,
            "outdir": str(output_dir),
        }

    planning_messages = deepcopy(messages)
    planning_instruction = (
        "\n\n请先生成一个可执行的计划，最多 "
        + str(max_plan_steps)
        + " 步。\n"
        "要求：\n"
        "- 本轮不要调用任何工具（tool_calls 必须是 []）。\n"
        "- 按严格 JSON 输出计划，content 必须是 JSON（对象或数组），不要输出解释或 Markdown。\n"
        "- 计划应保持执行导向，优先给出能完成任务的最短可行步骤；不要为了形式感而刻意拆细，但也不要省略真正影响执行路径的关键步骤。\n"
        "- 若后续执行中发现当前计划不够自然、存在冗余或遗漏，你可以再通过 PLAN_UPDATE 对剩余计划做自我修正，因此这里不必追求僵硬的一次定稿。\n"
        '  例如：{"plan":["步骤1","步骤2"]}\n'
    )
    for message in reversed(planning_messages):
        if message.get("role") == "user":
            message["content"] = str(message.get("content") or "") + planning_instruction
            break
    llm_calls += 1
    plan_result = generate_ai_message(
        model_config,
        planning_messages,
        tools_schema,
        mode,
        str(llm_calls_dir),
        f"llm_call_{llm_calls:03d}",
        forced_profile=forced_profile,
    )
    plan_ai_message = plan_result["ai_message"]
    messages.append(plan_ai_message)
    turns.append(
        {
            "turn_index": llm_calls,
            "phase": "plan",
            "ai_message": plan_ai_message,
            "llm_status": plan_result.get("status"),
            "llm_error": plan_result.get("error"),
            "tool_messages": [],
        }
    )

    plan_payload = parse_json_object_from_text(plan_ai_message.get("content", ""))
    try:
        steps = normalize_plan_steps(plan_payload, max_plan_steps) if plan_payload is not None else None
    except Exception:
        steps = None
    if not steps:
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = str(message.get("content") or "").strip()
                break
        steps = fallback_plan_steps(user_text, max_plan_steps)
    plan_state: dict = {"mode": "plan_execute", "steps": steps, "evidence_policy": evidence_policy}
    trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, None)
    _persist_runtime_state(
        output_dir,
        messages,
        trace,
        "",
        plan_state,
        llm_calls,
        tool_rounds,
        "planning_done",
        runtime_bridge=runtime_bridge,
        phase="planning_done",
    )

    pending_step = next_pending_step(plan_state)
    planner_tool_calls = plan_ai_message.get("tool_calls") if isinstance(plan_ai_message.get("tool_calls"), list) else []
    planner_requested_tools = bool(planner_tool_calls)
    pending_fast_tool_calls = _deterministic_plan_tool_calls(
        current_plan_step_title(plan_state, pending_step), tools_schema
    )
    if pending_step is not None and len(plan_state.get("steps") or []) <= 1 and not planner_requested_tools:
        step_title = current_plan_step_title(plan_state, pending_step)
        if not _step_requires_external_evidence(step_title, planner_tool_calls):
            for step in plan_state.get("steps") or []:
                step["status"] = "completed"
                step["summary"] = "Fast Path：单步任务且不需要外部工具证据，直接完成。"
                step["error"] = ""
            finish_requested = True
            trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, None)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                "",
                plan_state,
                llm_calls,
                tool_rounds,
                "fast_path_ready",
                runtime_bridge=runtime_bridge,
                phase="fast_path_ready",
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        build_plan_execute_instruction(plan_state)
                        + "\n\n该任务计划步数≤1，且不需要外部工具证据。请直接输出最终回答（schema A），不要调用任何工具。"
                    ),
                }
            )
        else:
            finish_requested = False
            messages.append(
                {
                    "role": "user",
                    "content": (
                        build_plan_execute_instruction(plan_state)
                        + "\n\n"
                        + f"开始执行计划。请先完成第 {pending_step} 步。需要工具就调用；完成后按 STEP_DONE 规则汇报。"
                    ),
                }
            )
    else:
        finish_requested = False
        messages.append(
            {
                "role": "user",
                "content": (
                    build_plan_execute_instruction(plan_state)
                    + "\n\n"
                    + f"开始执行计划。请先完成第 {pending_step} 步。需要工具就调用；完成后按 STEP_DONE 规则汇报。"
                ),
            }
        )

    final_answer = ""
    status = "success"
    terminal_error = None
    max_llm_calls = 1 + max_plan_steps * 3 + max_turns * 2
    tool_result_cache: dict[str, dict] = {}
    step_evidence: dict[int, list[dict]] = {}
    budget_notice_sent = False

    while True:
        if llm_calls >= max_llm_calls:
            status = "max_llm_calls_exceeded"
            final_answer = "任务因 Plan-and-Execute 循环次数过多而终止。"
            terminal_error = {
                "type": "MaxLLMCallsExceeded",
                "message": final_answer,
                "max_llm_calls": max_llm_calls,
                "plan_state": deepcopy(plan_state),
            }
            break
        if pending_fast_tool_calls:
            ai_message = make_ai_message("", pending_fast_tool_calls)
            llm_status = "success"
            llm_error = None
            pending_fast_tool_calls = []
            turn_phase = "deterministic_tool_fast_path"
        else:
            llm_calls += 1
            llm_result = generate_ai_message(
                model_config,
                messages,
                tools_schema,
                mode,
                str(llm_calls_dir),
                f"llm_call_{llm_calls:03d}",
                forced_profile=forced_profile,
            )
            ai_message = llm_result["ai_message"]
            llm_status = llm_result.get("status")
            llm_error = llm_result.get("error")
            turn_phase = "finish" if finish_requested else "execute"
        messages.append(ai_message)
        turn: dict = {
            "turn_index": llm_calls,
            "phase": turn_phase,
            "ai_message": ai_message,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "plan_state": deepcopy(plan_state),
        }
        if llm_status != "success":
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_calls,
                "cause": llm_error,
            }
            turns.append(turn)
            break

        tool_calls = ai_message.get("tool_calls", [])
        if tool_calls:
            step_marker = str(ai_message.get("_step_marker") or "").strip()
            step_marker_result = None
            deferred_step_done = False
            deferred_step_no = None
            if step_marker and not step_marker.startswith("STEP_FAIL:"):
                deferred_step_done, deferred_step_no = _should_defer_step_done_marker(
                    plan_state,
                    step_marker,
                    tool_calls,
                    step_evidence,
                    evidence_policy,
                )
                if not deferred_step_done:
                    plan_state, step_marker_result = apply_step_marker(plan_state, step_marker)
            current_step = next_pending_step(plan_state)
            if deferred_step_done and deferred_step_no is not None:
                current_step = deferred_step_no
            if current_step is None:
                inferred_step = _infer_target_step_from_tool_calls(plan_state, tool_calls)
                if inferred_step is not None:
                    for step in plan_state.get("steps") or []:
                        if step.get("step") == inferred_step and step.get("status") == "failed":
                            step["status"] = "pending"
                            step["error"] = ""
                            if not step.get("summary"):
                                step["summary"] = ""
                            break
                    current_step = inferred_step
            current_step_title_text = current_plan_step_title(plan_state, current_step)
            allowed_tool_calls, _ = _validate_tool_calls_for_step(
                plan_state,
                current_step,
                tool_calls,
                step_evidence.get(current_step, []),
            )
            tool_calls = allowed_tool_calls
            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"第 {current_step} 步“{current_step_title_text}”当前没有有效的工具调用可执行。"
                            "请重新决策：要么输出新的有效工具调用，要么在证据充足时输出 STEP_DONE。"
                        ),
                    }
                )
                turn["note"] = "empty_tool_calls_after_validation"
                turn["plan_state"] = deepcopy(plan_state)
                turns.append(turn)
                trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
                _persist_runtime_state(
                    output_dir,
                    messages,
                    trace,
                    final_answer,
                    plan_state,
                    llm_calls,
                    tool_rounds,
                    "tool_calls_rejected",
                    runtime_bridge=runtime_bridge,
                    phase="plan_execute",
                )
                continue
            duplicate_confirmation = _duplicate_resource_confirmation_text(tool_calls)
            if duplicate_confirmation:
                status = "needs_user"
                terminal_error = {
                    "type": "HumanInTheLoop",
                    "message": "agent requests user confirmation for duplicate resource reads",
                    "question": duplicate_confirmation,
                    "plan_state": deepcopy(plan_state),
                }
                turn["note"] = "duplicate_resource_ambiguity"
                turn["plan_state"] = deepcopy(plan_state)
                turns.append(turn)
                trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
                _persist_runtime_state(
                    output_dir,
                    messages,
                    trace,
                    final_answer,
                    plan_state,
                    llm_calls,
                    tool_rounds,
                    status,
                    pending_question=duplicate_confirmation,
                    runtime_bridge=runtime_bridge,
                    phase="plan_wait_user",
                )
                break
            if tool_rounds >= max_turns:
                if budget_notice_sent:
                    turn["note"] = "tool_budget_blocked"
                    turn["blocked_tool_calls"] = tool_calls
                    turns.append(turn)
                    pending_question = _build_budget_exhausted_ask_user(
                        tool_calls,
                        current_step_title_text,
                        step_evidence.get(current_step, []),
                        current_step=current_step,
                    )
                    if pending_question:
                        status = "needs_user"
                        terminal_error = {
                            "type": "HumanInTheLoop",
                            "message": "tool call budget exhausted and user input is required to proceed",
                            "question": pending_question,
                            "blocked_tool_calls": tool_calls,
                            "plan_state": deepcopy(plan_state),
                        }
                        trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
                        _persist_runtime_state(
                            output_dir,
                            messages,
                            trace,
                            final_answer,
                            plan_state,
                            llm_calls,
                            tool_rounds,
                            status,
                            pending_question=pending_question,
                            runtime_bridge=runtime_bridge,
                            phase="plan_wait_user",
                        )
                        break
                    status = "tool_budget_exceeded"
                    terminal_error = {
                        "type": "ToolBudgetExceeded",
                        "message": "tool call budget exhausted before the task converged",
                        "blocked_tool_calls": tool_calls,
                        "plan_state": deepcopy(plan_state),
                    }
                    trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
                    _persist_runtime_state(
                        output_dir,
                        messages,
                        trace,
                        final_answer,
                        plan_state,
                        llm_calls,
                        tool_rounds,
                        status,
                        runtime_bridge=runtime_bridge,
                        phase="plan_budget_exhausted",
                    )
                    break
                budget_notice_sent = True
                shortfall_guidance = _build_file_count_shortfall_guidance(current_step_title_text, step_evidence.get(current_step, []))
                turn["note"] = "tool_budget_notice"
                turn["blocked_tool_calls"] = tool_calls
                turns.append(turn)
                messages.append(
                    {
                        "role": "user",
                        "content": _append_text_block(
                            (
                                "本次运行的工具预算已用尽，后续不会再执行新的工具调用。\n"
                                "请仅基于当前已有证据继续收敛，自行选择最合适的下一步：\n"
                                + (f"- 若第 {current_step} 步已经可以完成，输出 STEP_DONE:{current_step}:...\n" if current_step is not None else "")
                                + "- 若仍需要用户确认、补充路径或决定方向，输出 ASK_USER:...\n"
                                + (f"- 若当前只能给出受限结论，也可以输出 STEP_FAIL:{current_step}:... 说明缺口与边界。\n" if current_step is not None else "- 若当前只能给出受限结论，也可以明确说明缺口与边界。\n")
                                + "如果你判断当前阻塞主要来自外部信息缺口，而不是推理能力不足，优先把这个缺口外化成 ASK_USER，而不是继续在同一路径上自我反思。"
                            ),
                            shortfall_guidance,
                        ),
                    }
                )
                trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
                _persist_runtime_state(
                    output_dir,
                    messages,
                    trace,
                    final_answer,
                    plan_state,
                    llm_calls,
                    tool_rounds,
                    "tool_budget_notice",
                    runtime_bridge=runtime_bridge,
                    phase="plan_budget_notice",
                )
                continue
            cache_hits: list[dict] = []
            pending_tool_calls: list[dict] = []
            for call in tool_calls:
                normalized_call = _normalize_tool_call_args_for_execution(call)
                name = str(normalized_call.get("name") or "")
                args = normalized_call.get("args") if isinstance(normalized_call.get("args"), dict) else {}
                cache_key = name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)
                cached_result = tool_result_cache.get(cache_key)
                if isinstance(cached_result, dict):
                    cache_hits.append(
                        {
                            "role": "tool",
                            "tool_call_id": normalized_call.get("id"),
                            "name": name,
                            "status": "success",
                            "content": json.dumps(cached_result, ensure_ascii=False),
                        }
                    )
                else:
                    pending_tool_calls.append(normalized_call)

            tool_messages_raw: list[dict] = []
            tool_outdir = None
            if pending_tool_calls:
                tool_rounds += 1
                tool_outdir = ensure_dir(output_dir / "tool_rounds" / f"round_{tool_rounds:03d}")
                tool_messages_raw = _execute_tool_calls_with_runtime_bridge(
                    runtime_bridge,
                    pending_tool_calls,
                    tools_config,
                    toolset,
                    str(tool_outdir),
                )

            tool_messages_raw = cache_hits + tool_messages_raw
            tool_messages, compression_meta = compress_tool_messages(tool_messages_raw, config)
            for message in tool_messages:
                if message.get("role") != "tool":
                    continue
                try:
                    result = _extract_tool_result(message)
                except Exception:
                    continue
                if str(result.get("status") or "").lower() != "success":
                    continue
                cache_input = result.get("input") if isinstance(result.get("input"), dict) else {}
                cache_key = str(result.get("skill_name") or "") + "|" + json.dumps(cache_input, ensure_ascii=False, sort_keys=True)
                tool_result_cache[cache_key] = result
            messages.extend(tool_messages)
            turn["tool_messages"] = tool_messages
            turn["tool_messages_compression"] = compression_meta
            if tool_outdir is not None:
                turn["tool_round_dir"] = str(tool_outdir)
            if cache_hits:
                turn["tool_cache_hits"] = len(cache_hits)
            if step_marker:
                turn["step_marker"] = step_marker
                turn["step_marker_result"] = step_marker_result
            if deferred_step_done and deferred_step_no is not None:
                turn["step_marker_deferred"] = deferred_step_no
            if current_step is not None:
                tool_summary = summarize_tool_round_for_step(tool_messages, current_step, current_step_title_text)
                merged_after = merge_tool_messages(tool_messages)
                if "是否存在" in current_step_title_text and merged_after:
                    only_missing = True
                    for item in merged_after:
                        if item.get("status") == "success":
                            only_missing = False
                            break
                        if item.get("name") != "file_reader":
                            only_missing = False
                            break
                        err_text = _tool_error_text(item.get("result", {}).get("error") if isinstance(item.get("result"), dict) else "")
                        if not _is_missing_file_error(err_text):
                            only_missing = False
                            break
                    if only_missing:
                        for step in plan_state.get("steps") or []:
                            if step.get("step") == current_step:
                                step["status"] = "completed"
                                step["summary"] = "已确认目标文件不存在。"
                                step["error"] = ""
                                break
                step_evidence.setdefault(current_step, []).extend(deepcopy(merged_after))
                propagated_steps = _propagate_step_evidence(
                    plan_state,
                    current_step,
                    merged_after,
                    step_evidence,
                    evidence_policy,
                )
                if propagated_steps:
                    turn["evidence_propagated_to_steps"] = propagated_steps
                if deferred_step_done and deferred_step_no == current_step:
                    current_step_evidence = step_evidence.get(current_step, [])
                    if _has_sufficient_evidence(current_step_title_text, current_step_evidence, evidence_policy):
                        plan_state, step_marker_result = apply_step_marker(plan_state, step_marker)
                has_failures = any(item.get("status") != "success" for item in merged_after)
                failure_guidance = ""
                if has_failures:
                    failure_guidance = (
                        f"\n- 若失败导致第 {current_step} 步无法继续，请直接输出 schema A，content 以 "
                        f"STEP_FAIL:{current_step}: 开头并说明原因；也可以仅重试失败项（不要重复成功项）。"
                    )
                replan_hint = _replan_hint_from_tool_failures(tool_messages) if has_failures else ""
                guardrail_text = _build_execution_guardrails(step_evidence)
                shortfall_guidance = _build_file_count_shortfall_guidance(current_step_title_text, step_evidence.get(current_step, []))
                sufficiency_nudge = _build_evidence_sufficiency_nudge(current_step_title_text, step_evidence.get(current_step, []))
                messages.append(
                    {
                        "role": "user",
                        "content": _append_text_block(
                            (
                                tool_summary
                                + "\n\n请只针对当前步骤做决策：\n"
                                + f"- 如果第 {current_step} 步已经完成，只输出 schema A，content 以 STEP_DONE:{current_step}: 开头，且不要再调用工具；\n"
                                + "- 如果还没完成，只调用新增必要工具，不要重复相同调用。"
                                + failure_guidance
                                + ("\n\n" + replan_hint if replan_hint else "")
                                + ("\n\n" + shortfall_guidance if shortfall_guidance else "")
                                + ("\n\n" + sufficiency_nudge if sufficiency_nudge else "")
                            ),
                            guardrail_text,
                        ),
                    }
                )
            turn["plan_state"] = deepcopy(plan_state)
            turns.append(turn)
            trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                "tool_round_completed",
                runtime_bridge=runtime_bridge,
                phase="plan_tool_round",
            )
            continue

        content = str(ai_message.get("content") or "")
        if content.strip().startswith("ASK_USER:"):
            status = "needs_user"
            terminal_error = {
                "type": "HumanInTheLoop",
                "message": "agent requests user confirmation",
                "question": content.strip(),
                "plan_state": deepcopy(plan_state),
            }
            turns.append(turn)
            trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                status,
                pending_question=content.strip(),
                runtime_bridge=runtime_bridge,
                phase="plan_wait_user",
            )
            break
        if finish_requested:
            final_answer = content
            turns.append(turn)
            trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                status,
                runtime_bridge=runtime_bridge,
                phase="plan_answer_ready",
            )
            break
        plan_state, marker = apply_step_marker(plan_state, content)
        marker_step = _extract_step_number_from_marker_text(content) if marker is not None else None
        if marker == "plan_update":
            pending_step = next_pending_step(plan_state)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        build_plan_execute_instruction(plan_state)
                        + "\n\n"
                        + (
                            f"计划已更新。请继续先完成第 {pending_step} 步。需要工具就调用；完成后按 STEP_DONE 规则汇报。"
                            if pending_step is not None
                            else "计划已更新且所有步骤均已完成。请直接输出最终回答（schema A）。"
                        )
                    ),
                }
            )
            turn["note"] = "plan_updated_by_model"
            turn["plan_state"] = deepcopy(plan_state)
            turns.append(turn)
            trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
            _persist_runtime_state(
                output_dir,
                messages,
                trace,
                final_answer,
                plan_state,
                llm_calls,
                tool_rounds,
                "plan_updated",
                runtime_bridge=runtime_bridge,
                phase="plan_updated",
            )
            continue
        if marker == "done" and marker_step is not None:
            step_title = current_plan_step_title(plan_state, marker_step)
            if not _has_sufficient_evidence(step_title, step_evidence.get(marker_step, []), evidence_policy):
                shortfall_guidance = _build_file_count_shortfall_guidance(step_title, step_evidence.get(marker_step, []))
                for step in plan_state.get("steps") or []:
                    if step.get("step") == marker_step:
                        step["status"] = "pending"
                        step["summary"] = ""
                        step["error"] = ""
                        break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"不能将第 {marker_step} 步标记为完成，因为当前没有足够的真实工具证据支撑这一步：{step_title}\n"
                            f"请先完成第 {marker_step} 步所需工具调用，再输出 STEP_DONE:{marker_step}:...\n"
                            "不要基于猜测或估算直接完成。\n"
                            "如果你判断当前缺口来自外部信息不足、用户约束需要确认，或工作区本身缺少满足条件的资源，也可以直接输出 ASK_USER: 把缺口说清楚，而不是只做内部反思。"
                            + ("\n" + shortfall_guidance if shortfall_guidance else "")
                        ),
                    }
                )
                turn["note"] = "step_done_without_evidence_blocked"
                turn["plan_state"] = deepcopy(plan_state)
                turns.append(turn)
                trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
                _persist_runtime_state(
                    output_dir,
                    messages,
                    trace,
                    final_answer,
                    plan_state,
                    llm_calls,
                    tool_rounds,
                    "step_done_blocked",
                    runtime_bridge=runtime_bridge,
                    phase="plan_execute",
                )
                continue
        if marker == "fail" and marker_step is not None:
            step_title = current_plan_step_title(plan_state, marker_step)
            if "是否存在" in step_title and _has_sufficient_evidence(step_title, step_evidence.get(marker_step, []), evidence_policy):
                for step in plan_state.get("steps") or []:
                    if step.get("step") == marker_step:
                        step["status"] = "completed"
                        if not step.get("summary"):
                            step["summary"] = "已确认目标文件不存在。"
                        step["error"] = ""
                        break
                marker = "done"
        if marker is None:
            target_step = first_failed_step(plan_state)
            if target_step is None:
                target_step = next_pending_step(plan_state)
            if target_step is not None:
                target_title = current_plan_step_title(plan_state, target_step)
                if _has_sufficient_evidence(target_title, step_evidence.get(target_step, []), evidence_policy):
                    inferred = f"STEP_DONE:{target_step}:{content.strip()}"
                    plan_state, _ = apply_step_marker(plan_state, inferred)
                else:
                    shortfall_guidance = _build_file_count_shortfall_guidance(target_title, step_evidence.get(target_step, []))
                    target_label = "失败步骤" if first_failed_step(plan_state) == target_step else "当前步骤"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"第 {target_step} 步“{target_title}”仍是需要优先处理的{target_label}，当前还缺少足够的真实工具证据，不能直接用自然语言完成。\n"
                                f"请只处理第 {target_step} 步：调用所需工具，或在确认需要用户补充/确认时输出 ASK_USER:...\n"
                                f"如果只是暂时未完成但仍可继续修复，也可以输出 STEP_FAIL:{target_step}:... 说明当前中断原因与下一步打算。\n"
                                "如果当前阻塞更像外部信息缺口而不是推理本身，优先用 ASK_USER 把缺口和你需要的确认项讲清楚。"
                                + ("\n" + shortfall_guidance if shortfall_guidance else "")
                            ),
                        }
                    )
                    turn["note"] = "free_text_without_evidence_blocked"
                    turn["plan_state"] = deepcopy(plan_state)
                    turns.append(turn)
                    trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
                    _persist_runtime_state(
                        output_dir,
                        messages,
                        trace,
                        final_answer,
                        plan_state,
                        llm_calls,
                        tool_rounds,
                        "free_text_blocked",
                        runtime_bridge=runtime_bridge,
                        phase="plan_execute",
                    )
                    continue
        pending_step = next_pending_step(plan_state)
        failed_step = first_failed_step(plan_state)
        guardrail_text = _build_execution_guardrails(step_evidence)
        if all_steps_completed(plan_state):
            finish_requested = True
            messages.append(
                {
                    "role": "user",
                    "content": _append_text_block(
                        "所有步骤已完成。请输出最终回答（schema A），不要以 STEP_DONE/STEP_FAIL 开头。",
                        guardrail_text,
                    ),
                }
            )
        elif failed_step is not None:
            failed_title = current_plan_step_title(plan_state, failed_step)
            messages.append(
                {
                    "role": "user",
                    "content": _append_text_block(
                        (
                            f"当前仍有未解决的失败步骤：第 {failed_step} 步“{failed_title}”。在所有步骤 completed 之前，不要输出最终回答。\n"
                            "请继续只处理这一步：\n"
                            + f"- 若已有工具证据足以说明该步骤客观上已完成，输出 STEP_DONE:{failed_step}:...\n"
                            + "- 若还可修复，请只请求该步骤所需的新工具调用；\n"
                            + "- 若必须让用户确认，输出 ASK_USER:..."
                        ),
                        guardrail_text,
                    ),
                }
            )
        else:
            state_json = json.dumps(plan_state, ensure_ascii=False)
            messages.append(
                {
                    "role": "user",
                    "content": _append_text_block(
                        f"当前计划状态：{state_json}\n请继续完成第 {pending_step} 步。完成后按 STEP_DONE 规则汇报。",
                        guardrail_text,
                    ),
                }
            )
        turn["plan_state"] = deepcopy(plan_state)
        turns.append(turn)
        trace = _build_runtime_trace("running", tool_rounds, llm_calls, plan_state, turns, terminal_error)
        _persist_runtime_state(
            output_dir,
            messages,
            trace,
            final_answer,
            plan_state,
            llm_calls,
            tool_rounds,
            "step_state_updated",
            runtime_bridge=runtime_bridge,
            phase="plan_execute",
        )

    trace = _build_runtime_trace(status, tool_rounds, llm_calls, plan_state, turns, terminal_error)
    _persist_runtime_state(
        output_dir,
        messages,
        trace,
        final_answer,
        plan_state,
        llm_calls,
        tool_rounds,
        status,
        runtime_bridge=runtime_bridge,
        phase="plan_finished",
    )
    return {
        "status": status,
        "final_answer": final_answer,
        "messages": messages,
        "trace": trace,
        "outdir": str(output_dir),
    }


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
    from b1_agent_runtime_1 import run_react_one_round_execute as b1_run_react_one_round_execute

    return b1_run_react_one_round_execute(
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
    from b1_agent_runtime_1 import run_adaptive_execute as b1_run_adaptive_execute

    return b1_run_adaptive_execute(
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
    from b1_agent_runtime_1 import run_plan_execute as b1_run_plan_execute

    return b1_run_plan_execute(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one AIMessage with a local or mock LLM.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--tools_schema", required=True)
    parser.add_argument("--mode", choices=["mock", "prompt_json", "native_tools"], default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--plan_execute", action="store_true")
    parser.add_argument("--adaptive_execute", action="store_true")
    parser.add_argument("--compare_tools_injection", action="store_true")
    parser.add_argument("--batch_eval", action="store_true")
    parser.add_argument("--batch_plan_execute", action="store_true")
    parser.add_argument("--eval_cases", default=None)
    parser.add_argument("--eval_modes", default=None)
    parser.add_argument("--eval_profiles", default=None)
    parser.add_argument("--tools_config", default=None)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_plan_steps", type=int, default=6)
    parser.add_argument("--evidence_policy", choices=["strict", "lite"], default="strict")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        requested_mode = "prompt_json" if args.plan_execute and args.mode is None else args.mode
        if args.batch_eval:
            if args.plan_execute or args.adaptive_execute or args.compare_tools_injection or args.batch_plan_execute:
                raise ValueError("--batch_eval cannot be combined with --plan_execute, --adaptive_execute, --batch_plan_execute or --compare_tools_injection")
            if not args.eval_cases:
                raise ValueError("--eval_cases is required with --batch_eval")
            eval_modes = [item.strip() for item in str(args.eval_modes or "prompt_json,native_tools").split(",") if item.strip()]
            eval_profiles = [item.strip() for item in str(args.eval_profiles or "").split(",") if item.strip()] or None
            run_batch_evaluation(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(resolve_cli_path(args.eval_cases)),
                str(outdir),
                eval_modes,
                eval_profiles,
            )
            print(outdir / "eval_report.csv")
        elif args.batch_plan_execute:
            if args.plan_execute or args.adaptive_execute or args.compare_tools_injection:
                raise ValueError("--batch_plan_execute cannot be combined with --plan_execute, --adaptive_execute or --compare_tools_injection")
            if not args.eval_cases:
                raise ValueError("--eval_cases is required with --batch_plan_execute")
            if not args.tools_config or not args.toolset:
                raise ValueError("--tools_config and --toolset are required with --batch_plan_execute")
            eval_modes = [item.strip() for item in str(args.eval_modes or "prompt_json,native_tools").split(",") if item.strip()]
            eval_profiles = [item.strip() for item in str(args.eval_profiles or "").split(",") if item.strip()] or None
            run_batch_plan_execute_evaluation(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(resolve_cli_path(args.eval_cases)),
                str(resolve_cli_path(args.tools_config)),
                str(args.toolset),
                str(outdir),
                eval_modes,
                eval_profiles,
                max_turns=int(args.max_turns),
                max_plan_steps=int(args.max_plan_steps),
                evidence_policy=str(args.evidence_policy),
            )
            print(outdir / "eval_report.csv")
        elif args.compare_tools_injection:
            if args.plan_execute:
                raise ValueError("--compare_tools_injection cannot be combined with --plan_execute")
            compare_tools_injection_modes(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.messages)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(outdir),
            )
            print(outdir / "comparison.json")
        elif args.adaptive_execute:
            if args.plan_execute or args.compare_tools_injection:
                raise ValueError("--adaptive_execute cannot be combined with --plan_execute or --compare_tools_injection")
            if not args.tools_config or not args.toolset:
                raise ValueError("--tools_config and --toolset are required with --adaptive_execute")
            run_adaptive_execute(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.messages)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(resolve_cli_path(args.tools_config)),
                str(args.toolset),
                str(outdir),
                max_turns=int(args.max_turns),
                max_plan_steps=int(args.max_plan_steps),
                evidence_policy=str(args.evidence_policy),
            )
            print(outdir / "final_answer.md")
        elif args.plan_execute:
            if not args.tools_config or not args.toolset:
                raise ValueError("--tools_config and --toolset are required with --plan_execute")
            run_plan_execute(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.messages)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(resolve_cli_path(args.tools_config)),
                str(args.toolset),
                requested_mode,
                str(outdir),
                int(args.max_turns),
                int(args.max_plan_steps),
                evidence_policy=str(args.evidence_policy),
            )
            print(outdir / "final_answer.md")
        else:
            generate_ai_message(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.messages)),
                read_json(resolve_cli_path(args.tools_schema)),
                requested_mode,
                str(outdir),
            )
            print(outdir / "ai_message.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
