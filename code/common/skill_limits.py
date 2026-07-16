from __future__ import annotations

from typing import Any

from common.errors import SkillLimitError


SKILL_LIMITS: dict[str, dict[str, Any]] = {
    "file_reader": {
        "max_chars": 10000,
    },
    "local_file_search": {
        "query_chars": 200,
        "top_k": 20,
        "max_files": 1000,
        "max_file_bytes": 1000000,
        "max_snippet_chars": 800,
        "max_file_types": 8,
        "max_include_globs": 20,
        "max_exclude_globs": 20,
    },
    "table_analyzer": {
        "max_rows_preview": 50,
    },
    "format_converter": {
        "text_chars": 20000,
    },
    "read_convert_file": {
        "max_chars": 8000,
    },
    "code_executor": {
        "code_chars": 12000,
        "timeout_seconds": 10,
        "max_allowed_imports": 8,
        "work_dir_chars": 80,
    },
}


def _value(args: dict[str, Any], name: str, default: Any = None) -> Any:
    return args[name] if name in args else default


def _require_int_at_most(skill_name: str, args: dict[str, Any], name: str, limit: int, default: int | None = None) -> None:
    value = _value(args, name, default)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        return
    if value > limit:
        raise SkillLimitError(
            f"{skill_name}.{name} exceeds policy limit: {value} > {limit}",
            details={"skill": skill_name, "parameter": name, "value": value, "limit": limit},
        )


def _require_text_at_most(skill_name: str, args: dict[str, Any], name: str, limit: int) -> None:
    value = args.get(name)
    if isinstance(value, str) and len(value) > limit:
        raise SkillLimitError(
            f"{skill_name}.{name} exceeds policy limit: {len(value)} chars > {limit}",
            details={"skill": skill_name, "parameter": name, "chars": len(value), "limit": limit},
        )


def _require_list_at_most(skill_name: str, args: dict[str, Any], name: str, limit: int) -> None:
    value = args.get(name)
    if isinstance(value, list) and len(value) > limit:
        raise SkillLimitError(
            f"{skill_name}.{name} exceeds policy limit: {len(value)} items > {limit}",
            details={"skill": skill_name, "parameter": name, "items": len(value), "limit": limit},
        )


def enforce_skill_limits(skill_name: str, input_data: dict[str, Any]) -> None:
    limits = SKILL_LIMITS.get(skill_name)
    if not limits:
        return
    args = input_data if isinstance(input_data, dict) else {}

    if skill_name == "file_reader":
        _require_int_at_most(skill_name, args, "max_chars", limits["max_chars"], default=2000)
        return

    if skill_name == "local_file_search":
        _require_text_at_most(skill_name, args, "query", limits["query_chars"])
        _require_int_at_most(skill_name, args, "top_k", limits["top_k"], default=5)
        _require_int_at_most(skill_name, args, "max_files", limits["max_files"], default=500)
        _require_int_at_most(skill_name, args, "max_file_bytes", limits["max_file_bytes"], default=1000000)
        _require_int_at_most(skill_name, args, "max_snippet_chars", limits["max_snippet_chars"], default=300)
        _require_list_at_most(skill_name, args, "file_types", limits["max_file_types"])
        _require_list_at_most(skill_name, args, "include_globs", limits["max_include_globs"])
        _require_list_at_most(skill_name, args, "exclude_globs", limits["max_exclude_globs"])
        return

    if skill_name == "table_analyzer":
        _require_int_at_most(skill_name, args, "max_rows_preview", limits["max_rows_preview"], default=5)
        return

    if skill_name == "format_converter":
        _require_text_at_most(skill_name, args, "text", limits["text_chars"])
        return

    if skill_name == "read_convert_file":
        _require_int_at_most(skill_name, args, "max_chars", limits["max_chars"], default=2000)
        return

    if skill_name == "code_executor":
        _require_text_at_most(skill_name, args, "code", limits["code_chars"])
        _require_int_at_most(skill_name, args, "timeout_seconds", limits["timeout_seconds"], default=3)
        _require_list_at_most(skill_name, args, "allowed_imports", limits["max_allowed_imports"])
        _require_text_at_most(skill_name, args, "work_dir", limits["work_dir_chars"])
