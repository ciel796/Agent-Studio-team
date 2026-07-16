from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import re
import sys
import threading
import traceback
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


AI_WEB_DIR = Path(__file__).resolve().parent
AGENT_DIR = AI_WEB_DIR.parent
STATIC_DIR = AI_WEB_DIR / "static"
OUTPUTS_DIR = AGENT_DIR / "outputs" / "web_ui"
CHECKPOINT_FILENAME = "checkpoint.json"
TRACE_FILENAME = "trace.json"
MESSAGES_FILENAME = "messages.json"
FINAL_ANSWER_FILENAME = "final_answer.md"
PROGRESS_FILENAME = "progress.md"
STOP_MARKER_FILENAME = ".ui_job_stop.json"
WORKER_ERROR_FILENAME = "worker_error.json"

sys.path.insert(0, str(AGENT_DIR / "code"))


INTERNAL_RUNTIME_USER_PREFIXES = (
    "已获得本轮工具结果。请基于最新证据继续决策：",
    "本次运行的工具预算已用尽，后续不会再执行新的工具调用。",
    "你正在以 Plan-and-Execute 模式工作。",
)
INTERNAL_RUNTIME_USER_SNIPPETS = (
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
)
RESUME_CONTEXT_PREFIX = "【继续上一个待确认问题】"

JOB_REGISTRY: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
MODEL_WARMUP_STATE: dict[str, object] = {"status": "not_started"}


def _llm_runtime_capabilities() -> tuple[bool, list[str]]:
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    return not missing, missing


def _start_model_warmup() -> None:
    available, missing = _llm_runtime_capabilities()
    if not available:
        MODEL_WARMUP_STATE.update({"status": "skipped", "reason": f"missing: {', '.join(missing)}"})
        return

    def _warmup() -> None:
        MODEL_WARMUP_STATE.update({"status": "running", "started_at": datetime.now().isoformat()})
        try:
            from b4_local_agent_llm import warmup_model

            result = warmup_model(str(AGENT_DIR / "configs" / "model.yaml"))
            MODEL_WARMUP_STATE.update(result)
        except Exception as exc:
            MODEL_WARMUP_STATE.update(
                {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
            )

    threading.Thread(target=_warmup, name="agent-model-warmup", daemon=True).start()


def _is_internal_runtime_user_message(message: object) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in INTERNAL_RUNTIME_USER_PREFIXES):
        return True
    if any(snippet in text for snippet in INTERNAL_RUNTIME_USER_SNIPPETS):
        return True
    if "开始执行计划。请先完成第 " in text and "STEP_DONE" in text:
        return True
    if "该任务计划步数≤1，且不需要外部工具证据。" in text:
        return True
    if "请重新决策：要么输出新的有效工具调用，要么在证据充足时输出 STEP_DONE。" in text:
        return True
    if "计划已更新。请继续先完成第 " in text and "STEP_DONE" in text:
        return True
    if "当前计划状态：" in text and "请继续完成第 " in text and "STEP_DONE" in text:
        return True
    if "所有步骤已完成。请输出最终回答（schema A），不要以 STEP_DONE/STEP_FAIL 开头。" in text:
        return True
    return False


def _humanize_internal_runtime_user_message(content: object) -> dict[str, str]:
    text = "" if content is None else str(content).strip()
    if not text:
        return {"summary": "", "details": ""}

    summary = "正在执行当前步骤"
    details = summary

    step_match = re.search(r"当前执行的是第\s*(\d+)\s*步[:：]\s*(.+?)(?:\n|$)", text)
    if step_match:
        step_no = step_match.group(1).strip()
        step_title = step_match.group(2).strip()
        summary = f"正在执行第 {step_no} 步：{step_title}"
        details = summary

    blocked_match = re.search(r"不能将第\s*(\d+)\s*步标记为完成，因为(.+?)(?:\n|$)", text)
    if blocked_match:
        step_no = blocked_match.group(1).strip()
        if not step_match or step_match.group(1).strip() != step_no:
            summary = f"正在执行第 {step_no} 步"
        details = summary

    shortfall_match = re.search(r"当前仅拿到\s*([^\n。]+)", text)
    if shortfall_match:
        details = summary

    unresolved_match = re.search(r"当前仍有未解决的失败步骤[:：]\s*第\s*(\d+)\s*步[“\"]?([^”\"\n]*)", text)
    if unresolved_match:
        step_no = unresolved_match.group(1).strip()
        step_title = unresolved_match.group(2).strip(" ”\"")
        summary = f"正在执行第 {step_no} 步"
        if step_title:
            summary = f"{summary}：{step_title}"
        details = summary

    if details == "正在执行当前步骤":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        filtered = [
            line for line in lines
            if not line.startswith("请只针对当前步骤做决策")
            and not line.startswith("动态重规划提示")
            and not line.startswith("- ")
        ]
        if filtered:
            first_line = filtered[0].rstrip("。")
            if first_line.startswith("当前执行的是第 "):
                summary = re.sub(r"^当前执行的是第\s*(\d+)\s*步[:：]\s*(.+)$", r"正在执行第 \1 步：\2", first_line)
            elif re.match(r"^第\s*\d+\s*步[:：]", first_line):
                summary = re.sub(r"^第\s*(\d+)\s*步[:：]\s*(.+)$", r"正在执行第 \1 步：\2", first_line)
            else:
                summary = f"正在执行：{first_line}"
            details = summary

    return {"summary": summary, "details": details or summary}


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _describe_tool_call_for_ui(call: object) -> str:
    if not isinstance(call, dict):
        return "我先调用一个工具获取所需信息。"
    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        function = call.get("function")
        if isinstance(function, dict):
            fn_name = function.get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                name = fn_name.strip()
    name = str(name or "unknown_tool")
    args = call.get("args")
    if not isinstance(args, dict):
        function = call.get("function")
        raw_args = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else {}
            except Exception:
                args = {"arguments": raw_args}
        else:
            args = {}

    if name == "file_reader":
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            return f"我先读取文件《{path.strip()}》获取内容。"
        return "我先读取相关文件获取内容。"
    if name == "local_file_search":
        query = str(args.get("query") or "").strip()
        root_dir = str(args.get("root_dir") or "").strip()
        if query and root_dir:
            return f"我先在 {root_dir} 里搜索和“{query}”相关的文件。"
        if query:
            return f"我先搜索和“{query}”相关的本地文件。"
        return "我先搜索相关的本地文件。"
    if name == "calculator":
        expression = str(args.get("expression") or "").strip()
        if expression:
            return f"我先计算一下：{expression}。"
        return "我先做一个计算。"
    if name == "table_analyzer":
        table_path = str(args.get("path") or args.get("table_path") or "").strip()
        if table_path:
            return f"我先分析表格《{table_path}》里的数据。"
        return "我先分析一下表格数据。"
    if name == "format_converter":
        source = str(args.get("path") or args.get("input_path") or "").strip()
        target = str(args.get("target_format") or args.get("format") or "").strip()
        if source and target:
            return f"我先把《{source}》转换成 {target} 格式。"
        return "我先做格式转换。"
    if name == "read_convert_file":
        path = str(args.get("path") or "").strip()
        if path:
            return f"我先读取并转换文件《{path}》。"
        return "我先读取并转换相关文件。"
    return f"我先调用工具“{name}”获取更多信息。参数：{_compact_json(args)}"


def _humanize_assistant_content(content: object) -> str:
    text = "" if content is None else str(content).strip()
    if not text:
        return ""
    ask_match = re.match(r"^ASK_USER:\s*(.+)$", text, flags=re.DOTALL)
    if ask_match:
        return f"我需要你确认一下：{ask_match.group(1).strip()}"
    step_done = re.match(r"^STEP_DONE:(\d+):(.*)$", text, flags=re.DOTALL)
    if step_done:
        return f"第 {step_done.group(1)} 步已完成：{step_done.group(2).strip()}"
    step_fail = re.match(r"^STEP_FAIL:(\d+):(.*)$", text, flags=re.DOTALL)
    if step_fail:
        return f"第 {step_fail.group(1)} 步遇到问题：{step_fail.group(2).strip()}"
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, list) and payload and all(isinstance(item, str) for item in payload):
            lines = ["我打算按这几个步骤来执行："]
            lines.extend(f"{index}. {item.strip()}" for index, item in enumerate(payload, 1))
            return "\n".join(lines)
    return text


def _humanize_message_for_ui(message: dict) -> dict:
    normalized = dict(message)
    if normalized.get("role") != "assistant":
        if normalized.get("role") == "user":
            normalized["content"] = _display_user_message_for_ui(normalized.get("content"))
        return normalized
    tool_calls = normalized.get("tool_calls")
    readable_content = _humanize_assistant_content(normalized.get("content"))
    if isinstance(tool_calls, list) and tool_calls:
        lines = [_describe_tool_call_for_ui(call) for call in tool_calls]
        if readable_content:
            lines.append(readable_content)
        normalized["content"] = "\n".join(line for line in lines if line).strip()
        normalized["tool_calls"] = []
        return normalized
    normalized["content"] = readable_content
    return normalized


def _parse_tool_result_payload(message: dict) -> dict | None:
    if not isinstance(message, dict) or message.get("role") != "tool":
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_tool_message_for_ui(message: dict) -> str:
    if not isinstance(message, dict):
        return ""
    tool_name = str(message.get("name") or "工具")
    payload = _parse_tool_result_payload(message)
    if not payload:
        content = str(message.get("content") or "").strip()
        return f"{tool_name} 已返回结果。" if content else f"{tool_name} 已执行。"

    status = str(payload.get("status") or "").strip() or "unknown"
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}

    if status == "success":
        if tool_name == "file_reader":
            source = str(output.get("source") or tool_input.get("path") or "").strip()
            return f"已读取文件《{source}》。" if source else "已读取相关文件。"
        if tool_name == "local_file_search":
            items = output.get("items") if isinstance(output.get("items"), list) else []
            query = str(tool_input.get("query") or "").strip()
            if query:
                return f"已完成本地搜索，和“{query}”相关的结果共 {len(items)} 条。"
            return f"已完成本地搜索，共找到 {len(items)} 条结果。"
        if tool_name == "calculator":
            value = output.get("result")
            return f"已完成计算，结果是 {value}。"
        if tool_name == "table_analyzer":
            rows = output.get("row_count")
            cols = output.get("column_count")
            if rows is not None and cols is not None:
                return f"已分析表格，共 {rows} 行、{cols} 列。"
            return "已完成表格分析。"
        if tool_name == "format_converter":
            path = str(output.get("output_path") or "").strip()
            return f"已完成格式转换，输出文件为《{path}》。" if path else "已完成格式转换。"
        return f"{tool_name} 已执行完成。"

    if tool_name == "file_reader":
        target = str(tool_input.get("path") or "").strip()
        if error.get("type") == "FileNotFoundError":
            return f"读取文件《{target}》失败：文件不存在。"
        return f"读取文件《{target}》失败。"

    message_text = str(error.get("message") or "").strip()
    if message_text:
        return f"{tool_name} 执行失败：{message_text}"
    return f"{tool_name} 执行失败。"


def _is_protocol_style_assistant_content(content: object) -> bool:
    text = "" if content is None else str(content).strip()
    if not text:
        return False
    if re.match(r"^(ASK_USER|STEP_DONE|STEP_FAIL):", text):
        return True
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, list) and payload and all(isinstance(item, str) for item in payload):
            return True
    return False


def _extract_chat_messages_for_ui(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        return []
    sanitized: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system" or role == "tool":
            continue
        if _is_internal_runtime_user_message(message):
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                continue
            humanized = _humanize_assistant_content(message.get("content"))
            if not humanized:
                continue
            if _is_protocol_style_assistant_content(message.get("content")) and not humanized.startswith("我需要你确认一下："):
                continue
            sanitized.append({"role": "assistant", "content": humanized})
            continue
        content = str(message.get("content") or "")
        if role == "user":
            content = _display_user_message_for_ui(content)
        sanitized.append({"role": role, "content": content})
    return _dedupe_adjacent_ui_messages(_collapse_adjacent_assistant_messages(sanitized))


def _normalize_ui_message_text(content: object) -> str:
    return re.sub(r"\s+", " ", str(content or "").strip())


def _dedupe_adjacent_ui_messages(messages: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if deduped:
            prev = deduped[-1]
            if (
                prev.get("role") == item.get("role")
                and _normalize_ui_message_text(prev.get("content")) == _normalize_ui_message_text(item.get("content"))
            ):
                continue
        deduped.append(item)
    return deduped


def _collapse_adjacent_assistant_messages(messages: list[dict]) -> list[dict]:
    collapsed: list[dict] = []
    pending_assistant: dict | None = None
    for item in messages:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "assistant":
            pending_assistant = item
            continue
        if pending_assistant is not None:
            collapsed.append(pending_assistant)
            pending_assistant = None
        collapsed.append(item)
    if pending_assistant is not None:
        collapsed.append(pending_assistant)
    return collapsed


def _append_final_answer_for_ui(chat_messages: list[dict], final_answer: str, status: str) -> list[dict]:
    if str(status or "").strip() != "success":
        return chat_messages
    answer = str(final_answer or "").strip()
    if not answer:
        return chat_messages
    normalized_answer = _normalize_ui_message_text(answer)
    normalized_existing = [
        _normalize_ui_message_text(item.get("content"))
        for item in chat_messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    if normalized_answer in normalized_existing:
        return chat_messages
    merged = list(chat_messages)
    merged.append({"role": "assistant", "content": answer})
    return _dedupe_adjacent_ui_messages(merged)


def _extract_plan_titles(messages: object, trace: dict | None = None) -> list[str]:
    plan_state = trace.get("plan_state") if isinstance(trace, dict) else {}
    steps = plan_state.get("steps") if isinstance(plan_state, dict) else []
    titles: list[str] = []
    if isinstance(steps, list):
        ordered_steps: list[tuple[int, str]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            title = str(step.get("title") or "").strip()
            if not title:
                continue
            try:
                step_no = int(step.get("step"))
            except (TypeError, ValueError):
                continue
            ordered_steps.append((step_no, title))
        ordered_steps.sort(key=lambda item: item[0])
        titles = [title for _, title in ordered_steps]
        if titles:
            return titles

    if not isinstance(messages, list):
        return []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        humanized = _humanize_assistant_content(message.get("content"))
        if not humanized.startswith("我打算按这几个步骤来执行："):
            continue
        titles = []
        for line in humanized.splitlines()[1:]:
            match = re.match(r"^\d+\.\s*(.+)$", line.strip())
            if match:
                titles.append(match.group(1).strip())
        if titles:
            return titles
    return []


def _build_plan_card(plan_titles: list[str]) -> dict | None:
    if not plan_titles:
        return None
    lines = ["我打算按这几个步骤来执行："]
    lines.extend(f"{index}. {title}" for index, title in enumerate(plan_titles, 1))
    return {"kind": "plan", "text": "\n".join(lines)}


def _build_process_updates_from_trace(trace: dict | None, plan_titles: list[str]) -> list[dict]:
    if not isinstance(trace, dict):
        return []
    plan_state = trace.get("plan_state")
    if not isinstance(plan_state, dict):
        turns = trace.get("turns")
        if isinstance(turns, list):
            for turn in reversed(turns):
                if isinstance(turn, dict) and isinstance(turn.get("plan_state"), dict):
                    plan_state = turn.get("plan_state")
                    break
    steps = plan_state.get("steps") if isinstance(plan_state, dict) else []
    if not isinstance(steps, list) or not steps:
        return []

    normalized_steps: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            step_no = int(step.get("step"))
        except (TypeError, ValueError):
            continue
        title = str(step.get("title") or "").strip()
        if not title and 0 < step_no <= len(plan_titles):
            title = plan_titles[step_no - 1]
        normalized_steps.append(
            {
                "step": step_no,
                "title": title,
                "status": str(step.get("status") or "").strip().lower(),
                "summary": str(step.get("summary") or "").strip(),
                "error": str(step.get("error") or "").strip(),
            }
        )
    normalized_steps.sort(key=lambda item: item["step"])

    updates: list[dict] = []
    active_step: dict | None = None
    for step in normalized_steps:
        status = step["status"]
        title = step["title"]
        step_no = step["step"]
        if status == "completed":
            summary = step["summary"] or title
            text = f"第 {step_no} 步已完成：{summary}" if summary else f"第 {step_no} 步已完成。"
            updates.append({"kind": "completed", "text": text})
        elif status == "failed":
            message = step["error"] or title or f"第 {step_no} 步执行受阻。"
            updates.append({"kind": "error", "text": f"第 {step_no} 步执行受阻：{message}"})
        elif active_step is None:
            active_step = step

    trace_status = str(trace.get("status") or "").strip().lower()
    if active_step and trace_status not in {"success", "stopped", "error"}:
        title = active_step["title"]
        step_no = active_step["step"]
        text = f"正在执行第 {step_no} 步：{title}" if title else f"正在执行第 {step_no} 步"
        updates.append({"kind": "active", "text": text})
    return updates


def _build_process_updates_from_messages(messages: object, plan_titles: list[str]) -> list[dict]:
    if not isinstance(messages, list):
        return []
    updates: list[dict] = []
    completed_steps: set[int] = set()
    active_step: tuple[int, str] | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            humanized = _humanize_assistant_content(message.get("content"))
            done_match = re.match(r"^第\s*(\d+)\s*步已完成[:：]\s*(.+)$", humanized)
            fail_match = re.match(r"^第\s*(\d+)\s*步遇到问题[:：]\s*(.+)$", humanized)
            if done_match:
                step_no = int(done_match.group(1))
                completed_steps.add(step_no)
                updates.append({"kind": "completed", "text": f"第 {step_no} 步已完成：{done_match.group(2).strip()}"})
            elif fail_match:
                step_no = int(fail_match.group(1))
                updates.append({"kind": "error", "text": f"第 {step_no} 步执行受阻：{fail_match.group(2).strip()}"})
        elif role == "user" and _is_internal_runtime_user_message(message):
            humanized = _humanize_internal_runtime_user_message(message.get("content"))
            text = str(humanized.get("summary") or "").strip()
            match = re.match(r"^正在执行第\s*(\d+)\s*步[:：]?\s*(.*)$", text)
            if match:
                active_step = (int(match.group(1)), match.group(2).strip())

    if active_step and active_step[0] not in completed_steps:
        step_no, title = active_step
        text = f"正在执行第 {step_no} 步：{title}" if title else f"正在执行第 {step_no} 步"
        updates.append({"kind": "active", "text": text})
    return updates


def _dedupe_process_updates(updates: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if deduped and str(deduped[-1].get("text") or "").strip() == text:
            continue
        deduped.append({"kind": item.get("kind") or "active", "text": text})
    return deduped


def _build_process_updates_for_ui(messages: object, trace: dict | None = None) -> list[dict]:
    plan_titles = _extract_plan_titles(messages, trace)
    updates: list[dict] = []
    plan_card = _build_plan_card(plan_titles)
    if plan_card:
        updates.append(plan_card)
    trace_updates = _build_process_updates_from_trace(trace, plan_titles)
    if trace_updates:
        updates.extend(trace_updates)
    else:
        updates.extend(_build_process_updates_from_messages(messages, plan_titles))
    return _dedupe_process_updates(updates)


def _build_plan_transition_updates_from_trace(trace: dict | None, existing_updates: list[dict]) -> list[dict]:
    if not isinstance(trace, dict):
        return []
    turns = trace.get("turns")
    if not isinstance(turns, list):
        return []

    explicit_done_steps: set[int] = set()
    explicit_failed_steps: set[int] = set()
    for item in existing_updates:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        done_match = re.match(r"^第\s*(\d+)\s*步已完成(?:[:：].*)?$", text)
        fail_match = re.match(r"^第\s*(\d+)\s*步遇到问题(?:[:：].*)?$", text)
        if done_match:
            explicit_done_steps.add(int(done_match.group(1)))
        if fail_match:
            explicit_failed_steps.add(int(fail_match.group(1)))

    previous_statuses: dict[int, str] = {}
    synthesized: list[dict] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        plan_state = turn.get("plan_state")
        steps = plan_state.get("steps") if isinstance(plan_state, dict) else []
        if not isinstance(steps, list):
            continue
        current_statuses: dict[int, str] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_no_raw = step.get("step")
            if isinstance(step_no_raw, bool):
                continue
            try:
                step_no = int(step_no_raw)
            except (TypeError, ValueError):
                continue
            status = str(step.get("status") or "").strip().lower()
            current_statuses[step_no] = status
            previous_status = previous_statuses.get(step_no)
            if status == "completed" and previous_status != "completed" and step_no not in explicit_done_steps:
                summary = str(step.get("summary") or "").strip()
                synthesized.append({"kind": "thinking", "text": f"第 {step_no} 步已完成：{summary}" if summary else f"第 {step_no} 步已完成。"})
                explicit_done_steps.add(step_no)
            elif status == "failed" and previous_status != "failed" and step_no not in explicit_failed_steps:
                error = str(step.get("error") or "").strip()
                synthesized.append({"kind": "thinking", "text": f"第 {step_no} 步遇到问题：{error}" if error else f"第 {step_no} 步遇到问题。"})
                explicit_failed_steps.add(step_no)
        previous_statuses = current_statuses
    return synthesized


def _display_user_message_for_ui(content: object) -> str:
    text = "" if content is None else str(content)
    if not text.startswith(RESUME_CONTEXT_PREFIX):
        return text
    answer_match = re.search(r"^我的补充回复：\s*(.+)$", text, flags=re.MULTILINE)
    if answer_match:
        return answer_match.group(1).strip()
    return text


def _sanitize_messages_for_ui(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        return []
    sanitized: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if _is_internal_runtime_user_message(message):
            continue
        sanitized.append(_humanize_message_for_ui(message))
    return sanitized


def _read_json_file_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_b4_eval_summary_path() -> Path | None:
    for root in (AGENT_DIR / "outputs" / "B4_llm", AGENT_DIR / "outputs" / "B4_compat"):
        if not root.is_dir():
            continue
        candidates = list(root.rglob("eval_summary.json"))
        if candidates:
            return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return None


def _read_text_file_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_progress_markdown(text: str) -> dict:
    content = str(text or "")
    parsed = {
        "status": "",
        "llm_calls": None,
        "tool_rounds": None,
        "steps_total": None,
        "steps_completed": None,
        "steps_failed": None,
        "steps_pending": None,
    }
    if not content.strip():
        return parsed
    status_match = re.search(r"^- status:\s*(.+)$", content, flags=re.MULTILINE)
    llm_match = re.search(r"^- llm_calls:\s*(\d+)$", content, flags=re.MULTILINE)
    tool_match = re.search(r"^- tool_rounds:\s*(\d+)$", content, flags=re.MULTILINE)
    steps_match = re.search(
        r"^- steps:\s*total=(\d+),\s*completed=(\d+),\s*failed=(\d+),\s*pending=(\d+)$",
        content,
        flags=re.MULTILINE,
    )
    if status_match:
        parsed["status"] = status_match.group(1).strip()
    if llm_match:
        parsed["llm_calls"] = int(llm_match.group(1))
    if tool_match:
        parsed["tool_rounds"] = int(tool_match.group(1))
    if steps_match:
        parsed["steps_total"] = int(steps_match.group(1))
        parsed["steps_completed"] = int(steps_match.group(2))
        parsed["steps_failed"] = int(steps_match.group(3))
        parsed["steps_pending"] = int(steps_match.group(4))
    return parsed


def _current_plan_progress(trace: dict) -> dict:
    plan_state = trace.get("plan_state") if isinstance(trace, dict) else {}
    steps = plan_state.get("steps") if isinstance(plan_state, dict) else []
    if not isinstance(steps, list) or not steps:
        return {"total": 0, "completed": 0, "failed": 0, "pending": 0, "current_step": None, "current_title": ""}
    total = len(steps)
    completed = sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "completed")
    failed = sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "failed")
    pending = total - completed - failed
    current_step = None
    current_title = ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("status") == "pending":
            current_step = step.get("step")
            current_title = str(step.get("title") or "").strip()
            break
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "current_step": current_step,
        "current_title": current_title,
    }


def _phase_from_progress_status(progress_status: str) -> tuple[str, str]:
    status = str(progress_status or "").strip().lower()
    mapping = {
        "initializing": ("preparing", "正在准备上下文"),
        "planning_done": ("planning_done", "已完成规划，准备开始执行"),
        "fast_path_ready": ("drafting", "正在直接整理答案"),
        "tool_round_completed": ("tool_result_ready", "已拿到工具结果，正在整理"),
        "tool_budget_notice": ("budget_notice", "工具预算已用完，正在基于现有证据收敛"),
        "tool_budget_exceeded": ("budget_exhausted", "工具预算已耗尽"),
        "tool_calls_rejected": ("replanning", "正在调整工具调用方案"),
        "step_done_blocked": ("evidence_blocked", "证据不足，正在修正执行步骤"),
        "free_text_blocked": ("evidence_blocked", "证据不足，正在修正执行步骤"),
        "plan_updated": ("replanning", "计划已更新，正在继续执行"),
        "step_state_updated": ("executing", "当前步骤进行中"),
        "success": ("completed", "已完成"),
        "needs_user": ("needs_user", "等待你的确认"),
        "stopped": ("stopped", "已中断"),
        "error": ("error", "执行出错"),
    }
    return mapping.get(status, ("running", "正在处理中"))


def _infer_ui_phase(trace: dict, checkpoint: object, run_dir: Path) -> dict:
    progress = _parse_progress_markdown(_read_text_file_if_exists(run_dir / PROGRESS_FILENAME))
    status = str(trace.get("status") or "running").strip().lower()
    final_answer_ready = bool(_read_text_file_if_exists(run_dir / FINAL_ANSWER_FILENAME).strip())
    turns = trace.get("turns") if isinstance(trace.get("turns"), list) else []
    last_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    last_phase = str(last_turn.get("phase") or "").strip().lower()
    plan_progress = _current_plan_progress(trace)
    code = "running"
    label = "正在处理中"

    if status == "success":
        code, label = "completed", "已完成"
    elif status == "needs_user":
        code, label = "needs_user", "等待你的确认"
    elif status == "stopped":
        code, label = "stopped", "已中断"
    elif status == "error":
        code, label = "error", "执行出错"
    elif progress.get("status"):
        code, label = _phase_from_progress_status(progress.get("status"))
    elif last_phase == "plan":
        code, label = "planning", "正在规划步骤"
    elif last_phase == "execute":
        code, label = "executing", "当前步骤进行中"
    elif last_phase == "react":
        code, label = "tool_calling", "正在调用工具"
    elif last_phase == "finish":
        code, label = "drafting", "正在生成最终答案"

    if code in {"executing", "tool_calling", "running"}:
        tool_messages = last_turn.get("tool_messages") if isinstance(last_turn, dict) else []
        ai_message = last_turn.get("ai_message") if isinstance(last_turn, dict) else {}
        tool_calls = ai_message.get("tool_calls") if isinstance(ai_message, dict) and isinstance(ai_message.get("tool_calls"), list) else []
        if isinstance(tool_messages, list) and tool_messages:
            code, label = "tool_result_ready", "已拿到工具结果，正在整理"
        elif tool_calls:
            code, label = "tool_calling", "正在调用工具"
        elif last_phase == "finish":
            code, label = "drafting", "正在生成最终答案"

    detail_parts: list[str] = []
    current_step = plan_progress.get("current_step")
    current_title = str(plan_progress.get("current_title") or "").strip()
    total = int(plan_progress.get("total") or progress.get("steps_total") or 0)
    completed = int(plan_progress.get("completed") or progress.get("steps_completed") or 0)
    all_steps_completed = total > 0 and completed >= total

    if code in {"running", "executing", "tool_calling", "tool_result_ready", "drafting"} and (
        final_answer_ready or all_steps_completed
    ):
        code, label = "finalizing", "正在收尾"

    if total > 0:
        detail_parts.append(f"已完成 {completed}/{total} 步")
    if current_step:
        if current_title:
            detail_parts.append(f"当前第 {current_step} 步：{current_title}")
        else:
            detail_parts.append(f"当前第 {current_step} 步")
    elif code in {"drafting", "finalizing"} and total > 0:
        detail_parts.append("正在汇总并生成最终回答")

    last_updated = None
    if isinstance(checkpoint, dict):
        updated_at = checkpoint.get("updated_at")
        if isinstance(updated_at, str) and updated_at.strip():
            last_updated = updated_at.strip()

    return {
        "code": code,
        "label": label,
        "detail": "；".join(detail_parts),
        "plan_total": total,
        "plan_completed": completed,
        "current_step": current_step,
        "current_step_title": current_title,
        "last_turn_phase": last_phase or None,
        "progress_status": progress.get("status") or None,
        "last_updated": last_updated,
    }


def _build_trace_summary(raw_trace: object, checkpoint: object, run_dir: Path) -> dict:
    trace = raw_trace if isinstance(raw_trace, dict) else {}
    state = checkpoint if isinstance(checkpoint, dict) else {}
    summary = dict(trace)
    if not summary:
        summary = {
            "conversation_id": (state.get("runtime") or {}).get("conversation_id") if isinstance(state.get("runtime"), dict) else None,
            "execution_mode": state.get("execution_mode"),
            "agent_mode": (state.get("runtime") or {}).get("agent_mode") if isinstance(state.get("runtime"), dict) else None,
            "status": state.get("status", "running"),
            "toolset": (state.get("runtime") or {}).get("toolset") if isinstance(state.get("runtime"), dict) else None,
            "max_turns": (state.get("runtime") or {}).get("max_turns") if isinstance(state.get("runtime"), dict) else None,
            "user_turn_count": len(state.get("user_turns") or []),
            "user_turns": state.get("user_turns") or [],
            "tool_rounds_used": state.get("tool_rounds"),
            "llm_call_count": state.get("llm_calls"),
            "pending_question": state.get("pending_question"),
            "error": state.get("terminal_error"),
            "warnings": state.get("warnings") or [],
        }
    summary.setdefault("run_dir", str(run_dir))
    summary["ui_phase"] = _infer_ui_phase(summary, checkpoint, run_dir)
    return summary


def _mark_job_finished(run_dir: Path) -> dict | None:
    key = str(run_dir.resolve())
    with JOB_LOCK:
        job = JOB_REGISTRY.get(key)
        if not job:
            return None
        process = job.get("process")
        thread = job.get("thread")
        if process is not None:
            try:
                job["returncode"] = process.poll()
            except Exception:
                job["returncode"] = None
        elif thread is not None and not thread.is_alive():
            job.setdefault("returncode", 0)
        if job.get("stdout_handle"):
            try:
                job["stdout_handle"].close()
            except Exception:
                pass
            job["stdout_handle"] = None
        if job.get("stderr_handle"):
            try:
                job["stderr_handle"].close()
            except Exception:
                pass
            job["stderr_handle"] = None
        return dict(job)


def _load_ui_run_payload(run_dir: Path) -> dict:
    checkpoint = _read_json_file_if_exists(run_dir / CHECKPOINT_FILENAME)
    runtime_input = _read_json_file_if_exists(_runtime_input_path(run_dir))
    raw_messages = _read_json_file_if_exists(run_dir / MESSAGES_FILENAME)
    if not isinstance(raw_messages, list) and isinstance(checkpoint, dict):
        raw_messages = checkpoint.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    raw_trace = _read_json_file_if_exists(run_dir / TRACE_FILENAME)
    trace = _build_trace_summary(raw_trace, checkpoint, run_dir)

    stopped = _read_json_file_if_exists(run_dir / STOP_MARKER_FILENAME)
    worker_error = _read_json_file_if_exists(run_dir / WORKER_ERROR_FILENAME)
    key = str(run_dir.resolve())
    with JOB_LOCK:
        job = JOB_REGISTRY.get(key)
        process = job.get("process") if isinstance(job, dict) else None
        thread = job.get("thread") if isinstance(job, dict) else None
        job_status = str(job.get("status") or "") if isinstance(job, dict) else ""
        job_returncode = job.get("returncode") if isinstance(job, dict) else None
    process_running = bool(process and process.poll() is None)
    thread_running = bool(thread and thread.is_alive())
    is_running = (process_running or thread_running) and job_status != "stopped"
    if (process or thread) and not process_running and not thread_running:
        finished_job = _mark_job_finished(run_dir)
        if isinstance(finished_job, dict):
            job_returncode = finished_job.get("returncode")

    status = str(trace.get("status") or "running")
    if is_running:
        status = "running"
    elif isinstance(stopped, dict):
        status = "stopped"
        trace["status"] = "stopped"
        trace["error"] = stopped.get("error")
    elif job_status == "stopped":
        status = "stopped"
        trace["status"] = "stopped"

    final_answer = _read_text_file_if_exists(run_dir / FINAL_ANSWER_FILENAME).strip()
    if not final_answer and isinstance(raw_trace, dict):
        final_answer = str(raw_trace.get("final_answer") or "").strip()

    # If the worker process has already exited but the persisted trace still says
    # "running" and there is no final answer, surface this as an incomplete run
    # instead of leaving the UI stuck in a fake in-progress state.
    if isinstance(worker_error, dict):
        status = "error"
        trace["status"] = "error"
        trace["error"] = worker_error
    elif (
        not is_running
        and status == "running"
        and not final_answer
        and not isinstance(stopped, dict)
    ):
        status = "error"
        trace["status"] = "error"
        message = "后台任务已结束，但没有产出最终答案。"
        if job_returncode not in (None, 0):
            message = f"后台任务异常退出（returncode={job_returncode}），且没有产出最终答案。"

    chat_messages = _extract_chat_messages_for_ui(raw_messages)
    chat_messages = _append_final_answer_for_ui(chat_messages, final_answer, status)
    process_updates = _build_process_updates_for_ui(raw_messages, trace)
    stderr_tail = _read_text_file_if_exists(run_dir / "web_stderr.log").strip()
    if not is_running and not isinstance(raw_trace, dict) and not isinstance(checkpoint, dict):
        if job_returncode not in (None, 0) or stderr_tail:
            status = "error"
            trace["status"] = "error"
            message = "后台任务启动失败。"
            if stderr_tail:
                message = stderr_tail.splitlines()[-1].strip() or message
            trace["error"] = {"type": "BackgroundJobStartError", "message": message}
    if status in {"error", "stopped"} and stderr_tail:
        tail_lines = stderr_tail.splitlines()[-6:]
        trace.setdefault("stderr_tail", "\n".join(tail_lines))
        if status == "error" and not trace.get("error"):
            message = tail_lines[-1].strip() if tail_lines else ""
            trace["error"] = {
                "type": "IncompleteRunError",
                "message": message or "后台任务已结束，但没有产出最终答案。",
            }
    elif status == "error" and not trace.get("error"):
        trace["error"] = {
            "type": "IncompleteRunError",
            "message": "后台任务已结束，但没有产出最终答案。",
        }
    trace["status"] = status
    if status == "error" and isinstance(trace.get("error"), dict):
        error = trace["error"]
        error_text = f"执行失败：{error.get('type', 'Error')}: {error.get('message', '未知错误')}"
        if not any(message.get("role") == "assistant" and message.get("content") == error_text for message in chat_messages):
            chat_messages.append({"role": "assistant", "content": error_text})
    runtime_options = runtime_input.get("agent_options") if isinstance(runtime_input, dict) and isinstance(runtime_input.get("agent_options"), dict) else {}
    model_profile = str(trace.get("model_profile") or runtime_options.get("forced_profile") or "").strip() or None
    selected_memory = _read_json_file_if_exists(run_dir / "selected_memory.json")
    saved_memory = _read_json_file_if_exists(run_dir / "saved_memory.json")

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "conversation_id": trace.get("conversation_id"),
        "execution_mode": trace.get("execution_mode"),
        "agent_mode": trace.get("agent_mode"),
        "model_profile": model_profile,
        "messages": chat_messages,
        "process_updates": process_updates,
        "trace": trace,
        "final_answer": final_answer,
        "is_running": is_running,
        "can_stop": is_running,
        "pending_question_path": str(run_dir / "pending_question.md") if (run_dir / "pending_question.md").exists() else None,
        "selected_memory": selected_memory,
        "saved_memory": saved_memory,
        "memory_summary_mode": bool(runtime_input.get("memory_summary_mode")) if isinstance(runtime_input, dict) else False,
    }


def _write_stop_marker(run_dir: Path, reason: str) -> None:
    payload = {
        "status": "stopped",
        "stopped_at": _now_tag(),
        "error": {"type": "UserInterrupted", "message": reason},
    }
    (run_dir / STOP_MARKER_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_stop_marker(run_dir: Path) -> None:
    marker = run_dir / STOP_MARKER_FILENAME
    if marker.exists():
        marker.unlink()


def _build_chat_execution(request: dict) -> tuple[Path, Path, str]:
    user_input = request["user_input"]
    conversation_id = request["conversation_id"]
    toolset = request["toolset"]
    llm_mode = request["llm_mode"]
    agent_mode = request["agent_mode"]
    max_turns = request["max_turns"]
    save_memory = request["save_memory"]
    selected_memory_ids = request["selected_memory_ids"]
    use_global_memory = request["use_global_memory"]
    resume = request["resume"]
    run_dir_value = request["run_dir"]

    if resume:
        run_dir = Path(run_dir_value).resolve()
        input_path = _prepare_resume_input(run_dir, user_input)
        _clear_stop_marker(run_dir)
    else:
        run_dir = OUTPUTS_DIR / conversation_id / _now_tag()
        run_dir.mkdir(parents=True, exist_ok=True)
        runtime_input = _build_runtime_input(
            conversation_id=conversation_id,
            user_input=user_input,
            toolset=toolset,
            max_turns=max_turns,
            save_memory=save_memory,
            selected_memory_ids=selected_memory_ids,
            use_global_memory=use_global_memory,
            llm_mode=llm_mode,
            agent_mode=agent_mode,
            model_profile=request.get("model_profile"),
            history_compression=request["history_compression"],
            system_prompt_events=request["system_prompt_events"],
            tool_cache_enabled=request["tool_cache_enabled"],
            memory_summary_mode=request["memory_summary_mode"],
            requested_llm_mode=request["requested_llm_mode"],
            conversation_history=request["conversation_history"],
        )
        input_path = _runtime_input_path(run_dir)
        input_path.write_text(json.dumps(runtime_input, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, input_path, conversation_id


def _parse_chat_request(body: dict) -> dict:
    user_input = (body.get("user_input") or "").strip()
    raw_conversation_id = body.get("conversation_id")
    conversation_id = (raw_conversation_id or f"conv_web_{_now_tag()}").strip()
    toolset = (body.get("toolset") or "basic_tools").strip()
    llm_mode = body.get("llm_mode")
    requested_llm_mode = llm_mode
    if llm_mode is not None and llm_mode not in {"mock", "prompt_json", "native_tools", "adaptive"}:
        raise ValueError("llm_mode must be one of: mock, prompt_json, native_tools, adaptive")
    agent_mode = str(body.get("agent_mode") or "adaptive_execute").strip()
    if agent_mode not in {"integrated", "react_one_round", "plan_execute", "adaptive_execute"}:
        raise ValueError("agent_mode must be one of: integrated, react_one_round, plan_execute, adaptive_execute")
    if llm_mode == "adaptive":
        llm_mode = {
            "integrated": "native_tools",
            "react_one_round": "native_tools",
            "plan_execute": "prompt_json",
            "adaptive_execute": "adaptive",
        }[agent_mode]
    llm_runtime_available, _ = _llm_runtime_capabilities()
    if not llm_runtime_available and llm_mode != "mock":
        llm_mode = "mock"

    selected_memory_ids = body.get("selected_memory_ids") or []
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")

    use_global_memory = body.get("use_global_memory")
    if use_global_memory is None:
        use_global_memory = True
    if not isinstance(use_global_memory, bool):
        raise ValueError("use_global_memory must be boolean")

    max_turns = body.get("max_turns")
    if max_turns is None:
        max_turns = 3
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")

    save_memory = body.get("save_memory")
    if save_memory is None:
        save_memory = "none"
    if save_memory not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")

    memory_summary_mode = body.get("memory_summary_mode", False)
    if not isinstance(memory_summary_mode, bool):
        raise ValueError("memory_summary_mode must be boolean")
    if memory_summary_mode:
        save_memory = "conversation"
        use_global_memory = False
        from b5_memory import _memory_paths, _read_index

        memory_paths = _memory_paths(str(AGENT_DIR / "configs" / "memory.yaml"))
        memory_index = _read_index(memory_paths["index"])
        conversation_memory_id = f"mem_conversation_{conversation_id}"
        if conversation_memory_id in memory_index and conversation_memory_id not in selected_memory_ids:
            selected_memory_ids.append(conversation_memory_id)

    model_profile = body.get("model_profile")
    if model_profile is not None:
        model_profile = str(model_profile).strip()
        if not model_profile:
            model_profile = None
    if model_profile:
        from common.io_utils import read_yaml

        model_config = read_yaml(AGENT_DIR / "configs" / "model.yaml") or {}
        available_profiles = {
            str(item.get("id"))
            for item in (_model_profile_catalog(model_config).get("profiles") or [])
            if isinstance(item, dict) and item.get("id")
        }
        if model_profile not in available_profiles:
            raise ValueError(f"unknown model_profile: {model_profile}")

    history_compression = body.get("history_compression", {"enabled": False})
    if isinstance(history_compression, bool):
        history_compression = {"enabled": history_compression}
    if not isinstance(history_compression, dict):
        raise ValueError("history_compression must be an object or boolean")
    system_prompt_events = body.get("system_prompt_events") or []
    if not isinstance(system_prompt_events, list):
        raise ValueError("system_prompt_events must be a list")
    tool_cache_enabled = body.get("tool_cache_enabled", True)
    if not isinstance(tool_cache_enabled, bool):
        raise ValueError("tool_cache_enabled must be boolean")
    conversation_history = body.get("conversation_history") or []
    if not isinstance(conversation_history, list):
        raise ValueError("conversation_history must be a list")
    conversation_history = [
        {"role": item.get("role"), "content": str(item.get("content") or "")}
        for item in conversation_history[-200:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
    ]

    resume = bool(body.get("resume"))
    run_dir_value = body.get("run_dir")
    if resume and (not isinstance(run_dir_value, str) or not run_dir_value.strip()):
        raise ValueError("resume requires run_dir")
    if not user_input and not resume:
        raise ValueError("user_input is required")
    if resume and not user_input:
        run_dir = Path(run_dir_value).resolve()
        pending_question = _load_pending_question_for_resume(run_dir)
        if pending_question:
            raise ValueError("resume for needs_user requires user_input")
        if not raw_conversation_id:
            runtime_payload = _load_runtime_input(run_dir)
            conversation_id = str(runtime_payload.get("conversation_id") or conversation_id).strip() or conversation_id

    return {
        "user_input": user_input,
        "conversation_id": conversation_id,
        "toolset": toolset,
        "llm_mode": llm_mode,
        "requested_llm_mode": requested_llm_mode,
        "agent_mode": agent_mode,
        "selected_memory_ids": selected_memory_ids,
        "use_global_memory": use_global_memory,
        "max_turns": max_turns,
        "save_memory": save_memory,
        "model_profile": model_profile,
        "history_compression": history_compression,
        "system_prompt_events": system_prompt_events,
        "tool_cache_enabled": tool_cache_enabled,
        "memory_summary_mode": memory_summary_mode,
        "conversation_history": conversation_history,
        "resume": resume,
        "run_dir": run_dir_value,
    }


def _start_background_job(run_dir: Path, input_path: Path, llm_mode: str | None, resume: bool, conversation_id: str) -> None:
    key = str(run_dir.resolve())

    def _worker() -> None:
        returncode = 0
        try:
            from b1_agent_runtime_1 import run_agent

            run_agent(
                str(input_path),
                str(AGENT_DIR / "configs" / "tools.yaml"),
                str(AGENT_DIR / "configs" / "memory.yaml"),
                str(AGENT_DIR / "configs" / "model.yaml"),
                str(run_dir),
                llm_mode,
                resume,
            )
        except Exception as exc:
            returncode = 1
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            (run_dir / WORKER_ERROR_FILENAME).write_text(
                json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            with JOB_LOCK:
                job = JOB_REGISTRY.get(key)
                if isinstance(job, dict):
                    job["returncode"] = returncode
                    if job.get("status") != "stopped":
                        job["status"] = "finished" if returncode == 0 else "error"

    thread = threading.Thread(target=_worker, name=f"agent-job-{conversation_id}", daemon=True)
    with JOB_LOCK:
        JOB_REGISTRY[key] = {
            "thread": thread,
            "conversation_id": conversation_id,
            "status": "running",
        }
    thread.start()


def _json_response(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error(handler: SimpleHTTPRequestHandler, status: int, message: str, *, details: object | None = None) -> None:
    payload: dict[str, object] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    _json_response(handler, payload, status=status)


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _model_profile_target(profile: dict) -> str:
    if not isinstance(profile, dict):
        return ""
    return str(profile.get("model_name_or_path") or profile.get("tokenizer_name_or_path") or "").strip()


def _model_profile_catalog(model_config: dict) -> dict[str, object]:
    pool = model_config.get("model_pool") if isinstance(model_config, dict) else {}
    if not isinstance(pool, dict):
        return {"profiles": [], "default_profile": None}

    reserved_aliases = {"default", "planner", "execute", "executor"}
    deduped_by_target: dict[str, dict] = {}
    ordered_profiles: list[dict] = []

    for name, settings in pool.items():
        if not isinstance(settings, dict):
            continue
        profile_id = str(name).strip()
        if not profile_id:
            continue
        target = _model_profile_target(settings)
        model_name = Path(target.replace("\\", "/")).name if target else profile_id
        option = {
            "id": profile_id,
            "label": f"{profile_id} ({model_name})" if model_name and model_name != profile_id else profile_id,
            "model_name": model_name or profile_id,
            "target": target,
        }
        dedupe_key = target or profile_id
        existing = deduped_by_target.get(dedupe_key)
        if existing is None:
            deduped_by_target[dedupe_key] = option
            ordered_profiles.append(option)
            continue
        if existing["id"] in reserved_aliases and profile_id not in reserved_aliases:
            index = ordered_profiles.index(existing)
            ordered_profiles[index] = option
            deduped_by_target[dedupe_key] = option

    routing = model_config.get("routing") if isinstance(model_config, dict) else {}
    evaluation = model_config.get("evaluation") if isinstance(model_config, dict) else {}
    raw_default = None
    if isinstance(routing, dict):
        raw_default = routing.get("default_profile")
    if not raw_default and isinstance(evaluation, dict):
        default_profiles = evaluation.get("default_profiles")
        if isinstance(default_profiles, list) and default_profiles:
            raw_default = default_profiles[0]
    if not raw_default and ordered_profiles:
        raw_default = ordered_profiles[0]["id"]

    default_profile = None
    if isinstance(raw_default, str) and raw_default.strip():
        raw_profile_settings = pool.get(raw_default)
        raw_target = _model_profile_target(raw_profile_settings) if isinstance(raw_profile_settings, dict) else ""
        if raw_target:
            for option in ordered_profiles:
                if option.get("target") == raw_target:
                    default_profile = option["id"]
                    break
        if not default_profile and any(option["id"] == raw_default for option in ordered_profiles):
            default_profile = raw_default
    if not default_profile and ordered_profiles:
        default_profile = ordered_profiles[0]["id"]

    return {"profiles": ordered_profiles, "default_profile": default_profile}


def _latest_trace_dir() -> Path | None:
    marker = OUTPUTS_DIR / ".latest_run.json"
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    path = payload.get("path") if isinstance(payload, dict) else None
    if not isinstance(path, str):
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def _set_latest_trace_dir(path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / ".latest_run.json").write_text(
        json.dumps({"path": str(path)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _runtime_input_path(run_dir: Path) -> Path:
    return run_dir / "runtime_input.json"


def _load_runtime_input(run_dir: Path) -> dict:
    input_path = _runtime_input_path(run_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"runtime_input.json not found in run_dir: {run_dir}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime_input.json must contain an object")
    return payload


def _load_pending_question_for_resume(run_dir: Path) -> str:
    checkpoint = _read_json_file_if_exists(run_dir / CHECKPOINT_FILENAME)
    if isinstance(checkpoint, dict):
        pending = checkpoint.get("pending_question") if isinstance(checkpoint.get("pending_question"), dict) else {}
        question = str(pending.get("question") or "").strip()
        if question:
            return question
    return _read_text_file_if_exists(run_dir / "pending_question.md").strip()


def _build_resume_user_input(run_dir: Path, user_input: str) -> str:
    pending_question = _load_pending_question_for_resume(run_dir)
    text = str(user_input or "").strip()
    if not pending_question:
        return text
    return (
        f"{RESUME_CONTEXT_PREFIX}\n"
        f"待确认问题：{pending_question}\n"
        f"我的补充回复：{text}"
    )


def _append_resume_user_input(run_dir: Path, user_input: str) -> Path:
    payload = _load_runtime_input(run_dir)
    enriched_input = _build_resume_user_input(run_dir, user_input)
    user_inputs = payload.get("user_inputs")
    if isinstance(user_inputs, list):
        payload["user_inputs"] = [str(item) for item in user_inputs] + [enriched_input]
    else:
        existing = payload.get("user_input")
        payload["user_inputs"] = [str(existing)] if isinstance(existing, str) and existing.strip() else []
        payload["user_inputs"].append(enriched_input)
    payload["user_input"] = payload["user_inputs"][0]
    input_path = _runtime_input_path(run_dir)
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return input_path


def _prepare_resume_input(run_dir: Path, user_input: str) -> Path:
    text = str(user_input or "").strip()
    if text:
        return _append_resume_user_input(run_dir, text)
    pending_question = _load_pending_question_for_resume(run_dir)
    if pending_question:
        raise ValueError("resume for needs_user requires user_input")
    return _runtime_input_path(run_dir)


def _build_runtime_input(
    *,
    conversation_id: str,
    user_input: str,
    toolset: str,
    max_turns: int,
    save_memory: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    llm_mode: str | None,
    agent_mode: str,
    model_profile: str | None,
    history_compression: dict,
    system_prompt_events: list[dict],
    tool_cache_enabled: bool,
    memory_summary_mode: bool,
    requested_llm_mode: str | None,
    conversation_history: list[dict],
) -> dict:
    runtime_input = {
        "conversation_id": conversation_id,
        "execution_mode": "integrated",
        "agent_mode": agent_mode,
        "user_input": user_input,
        "system_prompt_path": str((AGENT_DIR / "prompts" / "local_tool_agent.txt").resolve()),
        "selected_memory_ids": selected_memory_ids,
        "use_global_memory": use_global_memory,
        "toolset": toolset,
        "max_turns": max_turns,
        "save_memory": save_memory,
        "history_compression": history_compression,
        "system_prompt_events": system_prompt_events,
        "tool_cache_enabled": tool_cache_enabled,
        "memory_context_mode": "summary" if memory_summary_mode else "full",
        "memory_summary_mode": memory_summary_mode,
        "requested_llm_mode": requested_llm_mode,
        "effective_llm_mode": llm_mode,
        "conversation_history": conversation_history,
    }
    agent_options: dict[str, object] = {}
    if model_profile:
        agent_options["forced_profile"] = model_profile
    if agent_mode != "integrated":
        fallback_mock = llm_mode == "mock"
        agent_options.update(
            {
                "llm_mode": llm_mode or "prompt_json",
                "max_plan_steps": 6,
                "evidence_policy": "lite",
                "low_mode": "mock" if fallback_mock else "native_tools",
                "high_mode": "mock" if fallback_mock else "prompt_json",
            }
        )
    if agent_options:
        runtime_input["agent_options"] = agent_options
    return runtime_input


def _enrich_batch_result(result: dict) -> dict:
    enriched = dict(result)
    enriched_tasks = []
    for item in result.get("tasks") or []:
        task = dict(item)
        output_dir = Path(str(task.get("outdir") or ""))
        runtime = _read_json_file_if_exists(output_dir / "runtime_input.json") or {}
        trace = _read_json_file_if_exists(output_dir / TRACE_FILENAME) or {}
        stats = _read_json_file_if_exists(output_dir / "tool_call_stats.json") or {}
        final_answer = _read_text_file_if_exists(output_dir / FINAL_ANSWER_FILENAME).strip()
        user_inputs = runtime.get("user_inputs") if isinstance(runtime, dict) else None
        if not isinstance(user_inputs, list):
            user_input = runtime.get("user_input") if isinstance(runtime, dict) else None
            user_inputs = [user_input] if isinstance(user_input, str) and user_input.strip() else []
        cache_hits = 0
        by_tool = stats.get("by_tool") if isinstance(stats, dict) else None
        if isinstance(by_tool, dict):
            cache_hits = sum(
                int(tool_stats.get("cache_hits") or 0)
                for tool_stats in by_tool.values()
                if isinstance(tool_stats, dict)
            )
        task.update(
            {
                "user_inputs": user_inputs,
                "conversation_id": runtime.get("conversation_id") if isinstance(runtime, dict) else None,
                "execution_mode": runtime.get("execution_mode", "integrated") if isinstance(runtime, dict) else None,
                "agent_mode": runtime.get("agent_mode", "integrated") if isinstance(runtime, dict) else None,
                "max_turns": runtime.get("max_turns") if isinstance(runtime, dict) else None,
                "history_compression": runtime.get("history_compression") if isinstance(runtime, dict) else None,
                "system_prompt_events": runtime.get("system_prompt_events") if isinstance(runtime, dict) else None,
                "tool_cache_enabled": runtime.get("tool_cache_enabled", True) if isinstance(runtime, dict) else None,
                "final_answer": final_answer,
                "runtime_status": trace.get("status", task.get("status")) if isinstance(trace, dict) else task.get("status"),
                "llm_call_count": trace.get("llm_call_count") if isinstance(trace, dict) else None,
                "tool_rounds_used": trace.get("tool_rounds_used") if isinstance(trace, dict) else None,
                "user_turn_count": trace.get("user_turn_count") if isinstance(trace, dict) else None,
                "history_compression_count": len(trace.get("history_compressions") or []) if isinstance(trace, dict) else 0,
                "system_prompt_event_count": len(trace.get("system_prompt_events_applied") or []) if isinstance(trace, dict) else 0,
                "cache_hits": cache_hits,
            }
        )
        enriched_tasks.append(task)
    enriched["tasks"] = enriched_tasks
    return enriched


class AgentStudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        if parsed.path == "/":
            self.path = "/index_1.html"
        elif parsed.path == "/index.html":
            self.path = "/index_1.html"
        elif parsed.path.startswith("/app.js"):
            suffix = ""
            if "?" in self.path:
                suffix = "?" + self.path.split("?", 1)[1]
            self.path = f"/app_1.js{suffix}"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404)
            return
        self._handle_api_post(parsed)

    def _handle_api_get(self, parsed) -> None:
        query = parse_qs(parsed.query or "")
        try:
            if parsed.path == "/api/status":
                self._api_status()
                return
            if parsed.path == "/api/tools":
                toolset = (query.get("toolset") or ["basic_tools"])[0]
                auto_from_code = (query.get("auto_from_code") or ["0"])[0] in {"1", "true", "yes"}
                self._api_tools(toolset, auto_from_code)
                return
            if parsed.path == "/api/memory/index":
                self._api_memory_index()
                return
            if parsed.path == "/api/trace/latest":
                self._api_trace_latest()
                return
            if parsed.path == "/api/config/model":
                self._api_model_config()
                return
            if parsed.path == "/api/eval/b4/latest":
                self._api_eval_b4_latest()
                return
            if parsed.path == "/api/chat/progress":
                self._api_chat_progress(query)
                return
        except Exception as exc:
            _error(self, 500, f"{type(exc).__name__}: {exc}")
            return
        _error(self, 404, "unknown api endpoint")

    def _handle_api_post(self, parsed) -> None:
        try:
            if parsed.path == "/api/chat":
                self._api_chat()
                return
            if parsed.path == "/api/chat/start":
                self._api_chat_start()
                return
            if parsed.path == "/api/chat/stop":
                self._api_chat_stop()
                return
            if parsed.path == "/api/memory/search":
                self._api_memory_search()
                return
            if parsed.path == "/api/memory/update":
                self._api_memory_update()
                return
            if parsed.path == "/api/memory/impact":
                self._api_memory_impact()
                return
            if parsed.path == "/api/batch/run":
                self._api_batch_run()
                return
        except Exception as exc:
            _error(self, 500, f"{type(exc).__name__}: {exc}")
            return
        _error(self, 404, "unknown api endpoint")

    def _api_status(self) -> None:
        from common.io_utils import read_yaml

        tools_config = read_yaml(AGENT_DIR / "configs" / "tools.yaml") or {}
        model_config = read_yaml(AGENT_DIR / "configs" / "model.yaml") or {}
        toolsets = list((tools_config.get("toolsets") or {}).keys()) if isinstance(tools_config, dict) else []
        profile_catalog = _model_profile_catalog(model_config)
        llm_runtime_available, missing_llm_dependencies = _llm_runtime_capabilities()
        payload = {
            "status": "ok",
            "agent_root": str(AGENT_DIR),
            "toolsets": toolsets,
            "models": [item.get("id") for item in profile_catalog.get("profiles") or [] if isinstance(item, dict)],
            "model_profiles": profile_catalog.get("profiles") or [],
            "default_model_profile": profile_catalog.get("default_profile"),
            "default_toolset": tools_config.get("default_toolset"),
            "default_mode": (model_config.get("runtime") or {}).get("default_mode") if isinstance(model_config, dict) else None,
            "model_config_path": str((AGENT_DIR / "configs" / "model.yaml").resolve()),
            "llm_runtime_available": llm_runtime_available,
            "missing_llm_dependencies": missing_llm_dependencies,
            "model_warmup": dict(MODEL_WARMUP_STATE),
        }
        _json_response(self, payload)

    def _api_tools(self, toolset: str, auto_from_code: bool) -> None:
        from b3_tool_layer import get_tools_schema

        schema = get_tools_schema(
            str(AGENT_DIR / "configs" / "tools.yaml"),
            toolset,
            outdir=None,
            auto_from_code=auto_from_code,
        )
        _json_response(self, {"status": "ok", "toolset": toolset, "schema": schema})

    def _api_memory_index(self) -> None:
        from b5_memory import _memory_paths, _read_index

        paths = _memory_paths(str(AGENT_DIR / "configs" / "memory.yaml"))
        index = _read_index(paths["index"])
        _json_response(self, {"status": "ok", "index": index})

    def _api_memory_search(self) -> None:
        from b5_memory import VECTOR_DIMENSIONS, _memory_paths, _read_index, _search_by_keywords, _search_by_vector

        body = _read_json_body(self)
        query = (body.get("query") or "").strip()
        mode = (body.get("mode") or "keyword").strip()
        top_k = body.get("top_k") or 5
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        paths = _memory_paths(str(AGENT_DIR / "configs" / "memory.yaml"))
        index = _read_index(paths["index"])

        if mode == "vector":
            results = _search_by_vector(index, query, paths["root"], top_k=top_k)
            _json_response(self, {
                "status": "ok",
                "mode": "vector",
                "vector_method": "hashed_bag_of_terms_cosine",
                "dimensions": VECTOR_DIMENSIONS,
                "results": results,
            })
            return

        results = _search_by_keywords(index, query, paths["root"], top_k=top_k)
        _json_response(self, {"status": "ok", "mode": "keyword", "results": results})

    def _api_model_config(self) -> None:
        from common.io_utils import read_yaml

        config = read_yaml(AGENT_DIR / "configs" / "model.yaml")
        catalog = _model_profile_catalog(config or {})
        _json_response(self, {"status": "ok", "config": config, "model_profiles": catalog.get("profiles") or [], "default_model_profile": catalog.get("default_profile")})

    def _api_eval_b4_latest(self) -> None:
        summary_path = _latest_b4_eval_summary_path()
        if summary_path is None:
            _json_response(self, {
                "status": "ok",
                "exists": False,
                "profiles": [],
                "modes": [],
                "summary": {},
                "evaluation": {},
            })
            return
        evaluation = _read_json_file_if_exists(summary_path)
        if not isinstance(evaluation, dict):
            raise ValueError(f"invalid B4 evaluation summary: {summary_path}")
        payload = dict(evaluation)
        payload.update({
            "status": "ok",
            "exists": True,
            "path": str(summary_path),
            "evaluation": evaluation,
        })
        _json_response(self, payload)

    def _api_batch_run(self) -> None:
        from b1_agent_runtime_1 import run_batch_tasks

        body = _read_json_body(self)
        batch_path = body.get("batch_path")
        batch_payload = body.get("batch")
        run_dir = OUTPUTS_DIR / "batch" / _now_tag()
        run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(batch_payload, (dict, list)):
            batch_file = run_dir / "batch_input.json"
            batch_file.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        elif isinstance(batch_path, str) and batch_path.strip():
            batch_file = Path(batch_path).expanduser()
            if not batch_file.is_absolute():
                batch_file = AGENT_DIR / batch_file
            batch_file = batch_file.resolve()
            if not batch_file.is_file():
                raise FileNotFoundError(f"batch input not found: {batch_file}")
        else:
            raise ValueError("batch_path or batch is required")
        result = run_batch_tasks(
            str(batch_file),
            str(AGENT_DIR / "configs" / "tools.yaml"),
            str(AGENT_DIR / "configs" / "memory.yaml"),
            str(AGENT_DIR / "configs" / "model.yaml"),
            str(run_dir),
            body.get("llm_mode"),
        )
        _json_response(self, {"status": "ok", "run_dir": str(run_dir), "result": _enrich_batch_result(result)})

    def _api_memory_update(self) -> None:
        from b5_memory import update_memory

        body = _read_json_body(self)
        memory_id = str(body.get("memory_id") or "").strip()
        new_answer = str(body.get("new_answer") or "").strip()
        strategy = str(body.get("conflict_strategy") or "merge").strip()
        if not memory_id or not new_answer:
            raise ValueError("memory_id and new_answer are required")
        if strategy not in {"merge", "replace", "skip", "ask"}:
            raise ValueError("conflict_strategy must be merge, replace, skip, or ask")
        operation_dir = OUTPUTS_DIR / "memory_operations" / _now_tag()
        operation_dir.mkdir(parents=True, exist_ok=True)
        messages_path = operation_dir / "messages.json"
        trace_path = operation_dir / "trace.json"
        answer_path = operation_dir / "answer.md"
        messages_path.write_text(json.dumps(body.get("messages") or [], ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path.write_text(json.dumps(body.get("trace") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        answer_path.write_text(new_answer, encoding="utf-8")
        result = update_memory(
            str(AGENT_DIR / "configs" / "memory.yaml"), memory_id,
            str(messages_path), str(trace_path), str(answer_path), strategy, str(operation_dir),
        )
        _json_response(self, {"status": "ok", "operation_dir": str(operation_dir), "result": result})

    def _api_memory_impact(self) -> None:
        from b5_memory import analyze_bad_memory_impact

        body = _read_json_body(self)
        bad_memory_id = str(body.get("bad_memory_id") or "").strip()
        good_memory_id = str(body.get("good_memory_id") or "").strip()
        query = str(body.get("query") or "").strip()
        if not bad_memory_id or not good_memory_id or not query:
            raise ValueError("bad_memory_id, good_memory_id, and query are required")
        operation_dir = OUTPUTS_DIR / "memory_operations" / _now_tag()
        operation_dir.mkdir(parents=True, exist_ok=True)
        result = analyze_bad_memory_impact(
            str(AGENT_DIR / "configs" / "memory.yaml"),
            bad_memory_id, good_memory_id, query, str(operation_dir),
        )
        _json_response(self, {"status": "ok", "operation_dir": str(operation_dir), "result": result})

    def _api_chat(self) -> None:
        from b1_agent_runtime_1 import run_agent

        request = _parse_chat_request(_read_json_body(self))
        run_dir, input_path, conversation_id = _build_chat_execution(request)

        result = run_agent(
            str(input_path),
            str(AGENT_DIR / "configs" / "tools.yaml"),
            str(AGENT_DIR / "configs" / "memory.yaml"),
            str(AGENT_DIR / "configs" / "model.yaml"),
            str(run_dir),
            request["llm_mode"],
            request["resume"],
        )

        _set_latest_trace_dir(run_dir)
        payload = _load_ui_run_payload(run_dir)
        payload.update(
            {
                "conversation_id": result["conversation_id"],
                "execution_mode": result["execution_mode"],
                "elapsed_ms": result["elapsed_ms"],
                "agent_mode": result.get("agent_mode", request["agent_mode"]),
                "model_profile": request.get("model_profile"),
                "selected_memory": result.get("selected_memory"),
                "saved_memory": result.get("saved_memory"),
            }
        )
        _json_response(self, payload)

    def _api_chat_start(self) -> None:
        request = _parse_chat_request(_read_json_body(self))
        run_dir, input_path, conversation_id = _build_chat_execution(request)
        _start_background_job(run_dir, input_path, request["llm_mode"], request["resume"], conversation_id)
        _set_latest_trace_dir(run_dir)
        payload = _load_ui_run_payload(run_dir)
        payload.update(
            {
                "conversation_id": conversation_id,
                "execution_mode": "integrated",
                "agent_mode": request["agent_mode"],
                "model_profile": request.get("model_profile"),
                "started": True,
            }
        )
        _json_response(self, payload)

    def _api_chat_progress(self, query: dict[str, list[str]]) -> None:
        run_dir_value = (query.get("run_dir") or [""])[0]
        if not run_dir_value:
            raise ValueError("run_dir is required")
        run_dir = Path(run_dir_value).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        _json_response(self, _load_ui_run_payload(run_dir))

    def _api_chat_stop(self) -> None:
        body = _read_json_body(self)
        run_dir_value = str(body.get("run_dir") or "").strip()
        if not run_dir_value:
            raise ValueError("run_dir is required")
        run_dir = Path(run_dir_value).resolve()
        key = str(run_dir)
        with JOB_LOCK:
            job = JOB_REGISTRY.get(key)
            process = job.get("process") if isinstance(job, dict) else None
            thread = job.get("thread") if isinstance(job, dict) else None
            if isinstance(job, dict):
                job["status"] = "stopped"
        if not process and not thread:
            _json_response(self, {"status": "ok", "run_dir": str(run_dir), "stopped": False, "message": "job not found"})
            return
        _write_stop_marker(run_dir, "当前回答已由用户手动中断。")
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()
        _mark_job_finished(run_dir)
        _json_response(
            self,
            {
                "status": "ok",
                "run_dir": str(run_dir),
                "stopped": True,
                "cooperative": bool(thread),
            },
        )

    def _api_trace_latest(self) -> None:
        latest_dir = _latest_trace_dir()
        if latest_dir is None:
            _json_response(self, {"status": "ok", "exists": False})
            return
        trace_path = latest_dir / TRACE_FILENAME
        checkpoint_path = latest_dir / CHECKPOINT_FILENAME
        if not trace_path.exists() and not checkpoint_path.exists():
            _json_response(self, {"status": "ok", "exists": False})
            return
        payload = _load_ui_run_payload(latest_dir)
        payload["exists"] = True
        _json_response(self, payload)


class AgentStudioHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    if not STATIC_DIR.exists():
        raise RuntimeError(f"static dir not found: {STATIC_DIR}")
    parser = argparse.ArgumentParser(description="Agent Studio Web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-warmup", action="store_true", help="disable background loading of the default model")
    args = parser.parse_args(argv)
    host = "127.0.0.1"
    port = args.port
    host = args.host
    try:
        server = AgentStudioHTTPServer((host, port), AgentStudioHandler)
    except OSError as exc:
        if exc.errno in {errno.EADDRINUSE, 98, 10048}:
            raise RuntimeError(
                f"port {port} on host {host} is already in use. "
                f"Try `--port 0` for an automatic free port, or inspect the listener with "
                f"`ss -ltnp | grep :{port}` / `lsof -iTCP:{port} -sTCP:LISTEN -nP`."
            ) from exc
        raise
    actual_host, actual_port = server.server_address[:2]
    if not args.no_warmup:
        _start_model_warmup()
    print(f"Agent Studio Web running: http://{actual_host}:{actual_port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
