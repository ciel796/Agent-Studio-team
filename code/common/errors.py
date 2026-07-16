from __future__ import annotations

import json
from typing import Any


class AgentError(Exception):
    code = "AGENT_ERROR"
    category = "internal"
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.details = details or {}
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        return error_to_dict(self, self.code, self.category, self.retryable, self.details)


class ToolConfigError(AgentError):
    code = "TOOL_CONFIG_ERROR"
    category = "configuration"


class ToolArgumentError(AgentError):
    code = "TOOL_ARGUMENT_ERROR"
    category = "invalid_input"


class ToolUnavailableError(AgentError):
    code = "TOOL_UNAVAILABLE"
    category = "configuration"


class SkillInputError(AgentError):
    code = "SKILL_INVALID_INPUT"
    category = "invalid_input"


class ResourceNotFoundError(AgentError):
    code = "RESOURCE_NOT_FOUND"
    category = "resource_not_found"


class UnsupportedFormatError(AgentError):
    code = "UNSUPPORTED_FORMAT"
    category = "unsupported_format"


class PathSecurityError(AgentError):
    code = "PATH_SECURITY_ERROR"
    category = "security"


class SandboxPolicyError(AgentError):
    code = "SANDBOX_POLICY_ERROR"
    category = "security"


class SkillExecutionError(AgentError):
    code = "SKILL_EXECUTION_ERROR"
    category = "execution"


class SkillLimitError(AgentError):
    code = "SKILL_LIMIT_EXCEEDED"
    category = "policy"


class ExecutionTimeoutError(AgentError):
    code = "EXECUTION_TIMEOUT"
    category = "timeout"
    retryable = True


def _cause_chain(exc: Exception) -> list[dict[str, str]]:
    causes: list[dict[str, str]] = []
    current = exc.__cause__ or exc.__context__
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return causes


def _classify_exception(exc: Exception) -> tuple[str, str, bool]:
    if isinstance(exc, AgentError):
        return exc.code, exc.category, exc.retryable
    message = str(exc).casefold()
    exc_name = type(exc).__name__.casefold()
    if isinstance(exc, FileNotFoundError):
        return "RESOURCE_NOT_FOUND", "resource_not_found", False
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED", "permission", False
    if isinstance(exc, TimeoutError):
        return "EXECUTION_TIMEOUT", "timeout", True
    if isinstance(exc, json.JSONDecodeError):
        return "JSON_PARSE_ERROR", "invalid_input", False
    if isinstance(exc, ImportError):
        return "IMPORT_ERROR", "configuration", False
    if isinstance(exc, AttributeError):
        return "ATTRIBUTE_ERROR", "configuration", False
    if "path escapes" in message:
        return "PATH_SECURITY_ERROR", "security", False
    if "not found" in message:
        return "RESOURCE_NOT_FOUND", "resource_not_found", False
    if "only supports" in message or "does not support" in message or "unsupported" in message:
        return "UNSUPPORTED_FORMAT", "unsupported_format", False
    if "sandbox" in message or "sandbox" in exc_name or "not allowed" in message:
        return "SANDBOX_POLICY_ERROR", "security", False
    if isinstance(exc, ValueError):
        return "INVALID_INPUT", "invalid_input", False
    return "INTERNAL_ERROR", "internal", False


def error_to_dict(
    exc: Exception,
    code: str | None = None,
    category: str | None = None,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inferred_code, inferred_category, inferred_retryable = _classify_exception(exc)
    error = {
        "type": type(exc).__name__,
        "code": code or inferred_code,
        "category": category or inferred_category,
        "message": str(exc),
        "retryable": inferred_retryable if retryable is None else retryable,
        "details": details or getattr(exc, "details", {}) or {},
    }
    causes = _cause_chain(exc)
    if causes:
        error["causes"] = causes
    return error
