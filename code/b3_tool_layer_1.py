from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, get_args, get_origin

from common.errors import ToolArgumentError, ToolConfigError, ToolUnavailableError, error_to_dict
from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path, resolve_from_file
from common.schemas import make_skill_result, make_tool_message, normalize_tool_call
from common.skill_limits import enforce_skill_limits


bootstrap_project_root()


JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}

INJECTED_PARAMETERS = {"data_root", "output_dir"}
CACHE_FILENAME = "tool_call_cache.json"


def _load_tools_config(tools_config: str | Path) -> tuple[Path, dict]:
    config_path = Path(tools_config).resolve()
    config = read_yaml(config_path)
    if not isinstance(config, dict):
        raise ToolConfigError("tools.yaml must contain an object")
    if not isinstance(config.get("tools"), dict) or not isinstance(config.get("toolsets"), dict):
        raise ToolConfigError("tools.yaml must define tools and toolsets")
    return config_path, config


def _resolve_toolset(config: dict, toolset: str | None) -> tuple[str, list[str]]:
    selected = toolset or config.get("default_toolset")
    if not isinstance(selected, str) or selected not in config["toolsets"]:
        raise ToolConfigError(f"toolset does not exist: {selected}")
    names = config["toolsets"][selected]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ToolConfigError(f"toolset {selected} must be a list of tool names")
    return selected, names


def _parameter_schema(tool: dict) -> dict:
    raw_parameters = tool.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ToolConfigError("tool parameters must be an object")
    properties = {}
    for name, definition in raw_parameters.items():
        if not isinstance(definition, dict) or definition.get("type") not in JSON_TYPES:
            raise ToolConfigError(f"invalid parameter schema for {name}")
        properties[name] = dict(definition)
    required = tool.get("required", [])
    if not isinstance(required, list) or not all(name in properties for name in required):
        raise ToolConfigError("required parameters must reference declared properties")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotation_to_json_schema(annotation: Any) -> dict:
    if annotation is inspect.Signature.empty:
        return {"type": "string"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    if annotation is list or origin is list:
        item_schema = _annotation_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if str(annotation).endswith(" | None") and args:
        non_none = [item for item in args if item is not type(None)]
        if non_none:
            return _annotation_to_json_schema(non_none[0])
    return {"type": "string"}


def _auto_parameter_schema(function: Any, configured_tool: dict) -> dict:
    configured_parameters = configured_tool.get("parameters", {}) if isinstance(configured_tool.get("parameters"), dict) else {}
    signature = inspect.signature(function)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in INJECTED_PARAMETERS:
            continue
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        schema = _annotation_to_json_schema(parameter.annotation)
        configured = configured_parameters.get(name)
        if isinstance(configured, dict):
            schema.update({key: value for key, value in configured.items() if key in {"description", "items"}})
        schema.setdefault("description", "Auto-generated from Python function signature.")
        properties[name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _load_configured_function(tool: dict) -> Any:
    try:
        module = importlib.import_module(tool["module"])
        return getattr(module, tool["function"])
    except (ImportError, AttributeError, KeyError) as exc:
        raise ToolConfigError(f"cannot load configured tool function: {exc}") from exc


def _tool_schema(name: str, tool: dict, auto_from_code: bool) -> dict:
    for field in ("module", "function", "description", "returns"):
        if field not in tool:
            raise ToolConfigError(f"tool {name} missing {field}")
    returns = tool["returns"]
    if not isinstance(returns, dict):
        raise ToolConfigError(f"tool {name} returns must be an object")
    function = _load_configured_function(tool) if auto_from_code else None
    description = inspect.getdoc(function).splitlines()[0] if auto_from_code and inspect.getdoc(function) else tool["description"]
    parameters = _auto_parameter_schema(function, tool) if auto_from_code else _parameter_schema(tool)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "x-returns": {"type": "object", "properties": returns},
        },
    }


def get_tools_schema(
    tools_config: str,
    toolset: str,
    outdir: str | None = None,
    auto_from_code: bool = False,
) -> list[dict]:
    _, config = _load_tools_config(tools_config)
    selected, tool_names = _resolve_toolset(config, toolset)
    schema = []
    for name in tool_names:
        tool = config["tools"].get(name)
        if not isinstance(tool, dict):
            raise ToolConfigError(f"toolset references missing tool: {name}")
        schema.append(_tool_schema(name, tool, auto_from_code))
    if outdir:
        output_dir = Path(outdir)
        write_json(schema, output_dir / "tools_schema.json")
        write_json(
            {
                "status": "success",
                "toolset": selected,
                "tool_count": len(schema),
                "tools": tool_names,
                "schema_source": "python_signature" if auto_from_code else "tools_yaml",
            },
            output_dir / "tool_schema_report.json",
        )
    return schema


def _validate_args(args: dict, definition: dict) -> None:
    parameter_schema = _parameter_schema(definition)
    properties = parameter_schema["properties"]
    missing = [name for name in parameter_schema["required"] if name not in args]
    if missing:
        raise ToolArgumentError(f"missing required parameters: {', '.join(missing)}", details={"missing": missing})
    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ToolArgumentError(f"unknown parameters: {', '.join(unknown)}", details={"unknown": unknown})
    for name, value in args.items():
        expected_name = properties[name]["type"]
        expected = JSON_TYPES[expected_name]
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            valid = False
        else:
            valid = isinstance(value, expected)
        if not valid:
            raise ToolArgumentError(f"parameter {name} must be {expected_name}", details={"parameter": name, "expected_type": expected_name})
        if expected_name == "array" and "items" in properties[name]:
            item_type = properties[name]["items"].get("type")
            if item_type in JSON_TYPES and not all(isinstance(item, JSON_TYPES[item_type]) for item in value):
                raise ToolArgumentError(f"parameter {name} contains invalid items", details={"parameter": name, "expected_item_type": item_type})


def _error_result(name: str, args: dict, exc: Exception, latency_ms: float = 0.0) -> dict:
    return make_skill_result(
        name,
        "error",
        args,
        None,
        error_to_dict(exc),
        latency_ms,
    )


def _cache_key(name: str, args: dict) -> str:
    payload = json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(output_dir: Path | None) -> dict:
    if not output_dir:
        return {}
    cache_path = output_dir / CACHE_FILENAME
    if not cache_path.exists():
        return {}
    try:
        cache = read_json(cache_path)
    except Exception:
        return {}
    return cache if isinstance(cache, dict) else {}


def _save_cache(cache: dict, output_dir: Path | None) -> None:
    if output_dir:
        write_json(cache, output_dir / CACHE_FILENAME)


def _call_function(function: Any, args: dict, resolved_data_root: Path, output_dir: Path | None) -> dict:
    kwargs = dict(args)
    signature = inspect.signature(function)
    if "data_root" in signature.parameters:
        kwargs["data_root"] = str(resolved_data_root)
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = str(output_dir) if output_dir else None
    output = function(**kwargs)
    if not isinstance(output, dict):
        raise TypeError("tool function must return a JSON object")
    return output


def _execute_with_retries(
    name: str,
    args: dict,
    definition: dict,
    resolved_data_root: Path,
    output_dir: Path | None,
    retry_limit: int,
    started: float,
) -> tuple[dict, list[dict]]:
    module = importlib.import_module(definition["module"])
    function = getattr(module, definition["function"])
    attempts: list[dict] = []
    for attempt_index in range(retry_limit + 1):
        attempt_started = perf_counter()
        try:
            output = _call_function(function, args, resolved_data_root, output_dir)
            attempt_latency = round((perf_counter() - attempt_started) * 1000, 3)
            attempts.append({"attempt": attempt_index + 1, "status": "success", "latency_ms": attempt_latency})
            total_latency = round((perf_counter() - started) * 1000, 3)
            return make_skill_result(name, "success", args, output, None, total_latency), attempts
        except Exception as exc:
            attempt_latency = round((perf_counter() - attempt_started) * 1000, 3)
            error = error_to_dict(exc)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "error",
                    "error_code": error["code"],
                    "retryable": error["retryable"],
                    "latency_ms": attempt_latency,
                }
            )
            if error["retryable"] and attempt_index < retry_limit:
                continue
            total_latency = round((perf_counter() - started) * 1000, 3)
            result = _error_result(name, args, exc, total_latency)
            if result.get("error") and len(attempts) > 1:
                result["error"].setdefault("details", {})["attempts"] = attempts
            return result, attempts
    raise RuntimeError("unreachable retry state")


def _stats_from_records(log_records: list[dict]) -> dict:
    by_tool: dict[str, dict] = {}
    for record in log_records:
        name = str(record.get("name") or "unknown")
        item = by_tool.setdefault(
            name,
            {"tool_calls": 0, "success": 0, "error": 0, "cache_hits": 0, "retries": 0, "latency_sum_ms": 0.0},
        )
        item["tool_calls"] += 1
        if record.get("status") == "success":
            item["success"] += 1
        else:
            item["error"] += 1
        if record.get("cache_hit"):
            item["cache_hits"] += 1
        attempts = record.get("attempts") if isinstance(record.get("attempts"), list) else []
        item["retries"] += max(0, len(attempts) - 1)
        item["latency_sum_ms"] += float(record.get("latency_ms") or 0.0)
    for item in by_tool.values():
        total = item["tool_calls"]
        item["failure_rate"] = round(item["error"] / total, 4) if total else 0.0
        item["avg_latency_ms"] = round(item["latency_sum_ms"] / total, 3) if total else 0.0
        del item["latency_sum_ms"]
    total_calls = sum(item["tool_calls"] for item in by_tool.values())
    total_errors = sum(item["error"] for item in by_tool.values())
    return {
        "generated_at": now_iso(),
        "total_tool_calls": total_calls,
        "total_errors": total_errors,
        "failure_rate": round(total_errors / total_calls, 4) if total_calls else 0.0,
        "by_tool": by_tool,
    }


def execute_tool_calls(
    tool_calls: list[dict],
    tools_config: str,
    toolset: str | None = None,
    outdir: str | None = None,
    retry_limit: int | None = None,
    cache_enabled: bool | None = None,
) -> list[dict]:
    config_path, config = _load_tools_config(tools_config)
    selected, allowed_tools = _resolve_toolset(config, toolset)
    if not isinstance(tool_calls, list):
        raise ToolArgumentError("tool_calls must be a list")
    settings = config.get("settings", {}) if isinstance(config.get("settings"), dict) else {}
    data_root_setting = settings.get("data_root", "../data")
    resolved_data_root = resolve_from_file(data_root_setting, config_path)
    retry_count = retry_limit if retry_limit is not None else int(settings.get("retry_limit", 1))
    if retry_count < 0 or retry_count > 5:
        raise ToolArgumentError("retry_limit must be between 0 and 5")
    use_cache = bool(settings.get("cache_enabled", True)) if cache_enabled is None else cache_enabled
    tool_messages = []
    log_records = []
    output_dir = Path(outdir) if outdir else None
    cache = _load_cache(output_dir) if use_cache else {}
    for index, raw_call in enumerate(tool_calls):
        start = perf_counter()
        cache_hit = False
        attempts: list[dict] = []
        try:
            call = normalize_tool_call(raw_call, index)
        except Exception as exc:
            call = {"id": f"call_{index + 1:03d}", "name": "unknown", "args": {}}
            result = _error_result(call["name"], call["args"], exc)
        else:
            name = call["name"]
            args = call["args"]
            if name not in allowed_tools or name not in config["tools"]:
                result = _error_result(name, args, ToolUnavailableError(f"tool is not available in {selected}: {name}", details={"toolset": selected, "tool": name}))
            else:
                definition = config["tools"][name]
                try:
                    _validate_args(args, definition)
                    enforce_skill_limits(name, args)
                    key = _cache_key(name, args)
                    cached = cache.get(key) if use_cache else None
                    if isinstance(cached, dict) and isinstance(cached.get("skill_result"), dict):
                        result = deepcopy(cached["skill_result"])
                        cache_hit = True
                    else:
                        result, attempts = _execute_with_retries(
                            name,
                            args,
                            definition,
                            resolved_data_root,
                            output_dir,
                            retry_count,
                            start,
                        )
                        if use_cache and result.get("status") == "success":
                            cache[key] = {
                                "name": name,
                                "args": args,
                                "created_at": now_iso(),
                                "skill_result": result,
                            }
                except (ImportError, AttributeError) as exc:
                    raise RuntimeError(f"cannot load configured tool {name}: {exc}") from exc
                except Exception as exc:
                    latency_ms = round((perf_counter() - start) * 1000, 3)
                    result = _error_result(name, args, exc, latency_ms)
        content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        message = make_tool_message(call["id"], call["name"], content, result["status"])
        tool_messages.append(message)
        log_records.append(
            {
                "timestamp": now_iso(),
                "toolset": selected,
                "tool_call_id": call["id"],
                "name": call["name"],
                "status": result["status"],
                "args": call["args"],
                "skill_result": result,
                "latency_ms": result["latency_ms"],
                "cache_hit": cache_hit,
                "attempts": attempts,
            }
        )
    if outdir:
        write_json(tool_messages, output_dir / "tool_messages.json")
        for record in log_records:
            append_jsonl(record, output_dir / "tool_call_log.jsonl")
        write_json(_stats_from_records(log_records), output_dir / "tool_call_stats.json")
        if use_cache:
            _save_cache(cache, output_dir)
    return tool_messages


def _tool_call_matches_expected(call: dict, expected_tool: str, expected_args: dict | None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        normalized = normalize_tool_call(call, 0)
    except Exception as exc:
        return False, [f"invalid tool_call: {exc}"]
    if normalized["name"] != expected_tool:
        issues.append(f"expected tool {expected_tool}, got {normalized['name']}")
    if expected_args:
        for key, expected_value in expected_args.items():
            actual_value = normalized["args"].get(key)
            if actual_value != expected_value:
                issues.append(f"arg {key} expected {expected_value!r}, got {actual_value!r}")
    return not issues, issues


def evaluate_tool_call_accuracy(eval_cases_path: str, outdir: str | None = None) -> dict:
    payload = read_json(eval_cases_path)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ToolArgumentError("eval cases must be a list or an object with cases")
    variants: dict[str, dict] = {}
    details: list[dict] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ToolArgumentError(f"eval case {index} must be an object")
        case_id = str(case.get("case_id") or f"case_{index + 1:03d}")
        expected_tool = case.get("expected_tool")
        expected_args = case.get("expected_args") if isinstance(case.get("expected_args"), dict) else {}
        predictions = case.get("tool_calls_by_schema") or case.get("variants")
        if not isinstance(expected_tool, str) or not isinstance(predictions, dict):
            raise ToolArgumentError(f"eval case {case_id} missing expected_tool or variants")
        for variant_name, calls in predictions.items():
            calls_list = calls if isinstance(calls, list) else [calls]
            call = calls_list[0] if calls_list else {}
            ok, issues = _tool_call_matches_expected(call, expected_tool, expected_args)
            item = variants.setdefault(str(variant_name), {"cases": 0, "correct": 0, "errors": 0})
            item["cases"] += 1
            if ok:
                item["correct"] += 1
            else:
                item["errors"] += 1
            details.append({"case_id": case_id, "schema_variant": str(variant_name), "correct": ok, "issues": issues})
    for item in variants.values():
        item["accuracy"] = round(item["correct"] / item["cases"], 4) if item["cases"] else 0.0
    report = {"generated_at": now_iso(), "variants": variants, "details": details}
    if outdir:
        write_json(report, Path(outdir) / "tool_call_accuracy_report.json")
    return report


def _load_b4_tool_eval_module():
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file and Path(main_file).resolve().name == "b4_local_agent_llm.py":
        return main_module
    import b4_local_agent_llm as b4_module

    return b4_module


def compare_tools_injection_modes(*args, **kwargs) -> dict:
    """B3-owned entrypoint for comparing prompt-injected vs native tool schemas."""
    b4_module = _load_b4_tool_eval_module()
    return b4_module.compare_tools_injection_modes(*args, **kwargs)


def run_batch_evaluation(*args, **kwargs) -> dict:
    """B3-owned entrypoint for batch tool-calling evaluation across models/modes."""
    b4_module = _load_b4_tool_eval_module()
    return b4_module.run_batch_evaluation(*args, **kwargs)


def run_batch_plan_execute_evaluation(*args, **kwargs) -> dict:
    """B3-owned entrypoint for batch Plan-and-Execute evaluation."""
    b4_module = _load_b4_tool_eval_module()
    return b4_module.run_batch_plan_execute_evaluation(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tool schema, execute tool calls, or run tool-calling evaluations.")
    parser.add_argument("--tools_config")
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--tool_calls")
    parser.add_argument("--eval_cases")
    parser.add_argument("--model_config")
    parser.add_argument("--messages")
    parser.add_argument("--tools_schema")
    parser.add_argument("--eval_modes")
    parser.add_argument("--eval_profiles")
    parser.add_argument("--auto_schema", action="store_true")
    parser.add_argument("--retry_limit", type=int, default=None)
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_plan_steps", type=int, default=6)
    parser.add_argument("--evidence_policy", choices=["strict", "lite"], default="strict")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export_schema", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--evaluate_accuracy", action="store_true")
    action.add_argument("--compare_tools_injection", action="store_true")
    action.add_argument("--batch_eval", action="store_true")
    action.add_argument("--batch_plan_execute", action="store_true")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        if args.export_schema:
            if not args.tools_config:
                raise ValueError("--tools_config is required with --export_schema")
            config_path = resolve_cli_path(args.tools_config)
            if not args.toolset:
                _, config = _load_tools_config(config_path)
                args.toolset = config.get("default_toolset")
            get_tools_schema(str(config_path), args.toolset, str(outdir), args.auto_schema)
            print(outdir / "tools_schema.json")
        elif args.evaluate_accuracy:
            if not args.eval_cases:
                raise ValueError("--eval_cases is required with --evaluate_accuracy")
            evaluate_tool_call_accuracy(str(resolve_cli_path(args.eval_cases)), str(outdir))
            print(outdir / "tool_call_accuracy_report.json")
        elif args.compare_tools_injection:
            if not args.model_config or not args.messages or not args.tools_schema:
                raise ValueError("--model_config, --messages and --tools_schema are required with --compare_tools_injection")
            compare_tools_injection_modes(
                str(resolve_cli_path(args.model_config)),
                read_json(resolve_cli_path(args.messages)),
                read_json(resolve_cli_path(args.tools_schema)),
                str(outdir),
            )
            print(outdir / "comparison.json")
        elif args.batch_eval:
            if not args.model_config or not args.tools_schema or not args.eval_cases:
                raise ValueError("--model_config, --tools_schema and --eval_cases are required with --batch_eval")
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
            if not args.model_config or not args.tools_schema or not args.eval_cases:
                raise ValueError("--model_config, --tools_schema and --eval_cases are required with --batch_plan_execute")
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
        else:
            if not args.tools_config:
                raise ValueError("--tools_config is required with --execute")
            if not args.tool_calls:
                raise ValueError("--tool_calls is required with --execute")
            config_path = resolve_cli_path(args.tools_config)
            payload = read_json(resolve_cli_path(args.tool_calls))
            tool_calls = payload.get("tool_calls") if isinstance(payload, dict) else payload
            execute_tool_calls(
                tool_calls,
                str(config_path),
                args.toolset,
                str(outdir),
                args.retry_limit,
                not args.disable_cache,
            )
            print(outdir / "tool_messages.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
