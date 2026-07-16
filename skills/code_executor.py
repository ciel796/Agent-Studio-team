from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import traceback
from pathlib import Path
from typing import Any

from skills import DEFAULT_DATA_ROOT, resolve_data_path


MAX_CODE_CHARS = 12000
MAX_TIMEOUT_SECONDS = 10
MAX_STDOUT_CHARS = 8000
MAX_STDERR_CHARS = 4000


SAFE_MODULES = {
    "collections": __import__("collections"),
    "datetime": __import__("datetime"),
    "decimal": __import__("decimal"),
    "fractions": __import__("fractions"),
    "functools": __import__("functools"),
    "itertools": __import__("itertools"),
    "json": json,
    "math": math,
    "random": __import__("random"),
    "re": __import__("re"),
    "statistics": statistics,
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

DANGEROUS_NAMES = {
    "__builtins__",
    "__debug__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}

DANGEROUS_ATTRS = {
    "chmod",
    "chown",
    "execv",
    "fork",
    "kill",
    "mro",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "spawn",
    "subclasses",
    "system",
    "unlink",
}


class SandboxViolation(ValueError):
    pass


class SandboxExecutionError(RuntimeError):
    pass


class SandboxValidator(ast.NodeVisitor):
    def __init__(self, allowed_imports: set[str]) -> None:
        self.allowed_imports = allowed_imports

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            root_name = alias.name.split(".", 1)[0]
            if root_name not in self.allowed_imports:
                raise SandboxViolation(f"import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.level:
            raise SandboxViolation("relative imports are not allowed")
        root_name = (node.module or "").split(".", 1)[0]
        if root_name not in self.allowed_imports:
            raise SandboxViolation(f"import is not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in DANGEROUS_NAMES or node.id.startswith("__"):
            raise SandboxViolation(f"name is not allowed: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("__") or node.attr in DANGEROUS_ATTRS:
            raise SandboxViolation(f"attribute is not allowed: {node.attr}")
        self.generic_visit(node)


def _normalize_allowed_imports(allowed_imports: list[str] | None) -> set[str]:
    if allowed_imports is None:
        return {"collections", "functools", "itertools", "json", "math", "re", "statistics"}
    requested = {item.strip() for item in allowed_imports if isinstance(item, str) and item.strip()}
    unknown = requested - set(SAFE_MODULES)
    if unknown:
        raise SandboxViolation(f"imports are not whitelisted: {', '.join(sorted(unknown))}")
    return requested


def _transform_last_expression(tree: ast.Module) -> ast.Module:
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=last.value,
        )
        ast.fix_missing_locations(tree)
    return tree


def _safe_import(name: str, globals_: dict | None = None, locals_: dict | None = None, fromlist: tuple = (), level: int = 0):
    if level:
        raise ImportError("relative imports are not allowed")
    root_name = name.split(".", 1)[0]
    if root_name not in SAFE_MODULES:
        raise ImportError(f"import is not allowed: {name}")
    if root_name != name:
        return __import__(name, globals_, locals_, fromlist, level)
    return SAFE_MODULES[root_name]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _worker(code: str, allowed_imports: list[str] | None, work_dir: str, result_queue: mp.Queue) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        os.chdir(work_dir)
        allowed = _normalize_allowed_imports(allowed_imports)
        tree = ast.parse(code, mode="exec")
        SandboxValidator(allowed).visit(tree)
        tree = _transform_last_expression(tree)
        compiled = compile(tree, "<sandbox>", "exec")
        builtins = dict(SAFE_BUILTINS)
        builtins["__import__"] = _safe_import
        globals_dict: dict[str, Any] = {"__builtins__": builtins, "__name__": "__sandbox__"}
        for name in allowed:
            globals_dict[name] = SAFE_MODULES[name]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, globals_dict, globals_dict)
        result_queue.put(
            {
                "status": "success",
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "result": _jsonable(globals_dict.get("__result__")),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                },
            }
        )


def _resolve_work_dir(work_dir: str, data_root: str | None) -> Path:
    root = Path(data_root).resolve() if data_root else DEFAULT_DATA_ROOT.resolve()
    path, _ = resolve_data_path(work_dir or "sandbox", str(root))
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_executor(
    code: str,
    timeout_seconds: int = 3,
    allowed_imports: list[str] | None = None,
    work_dir: str = "sandbox",
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code is too long; maximum is {MAX_CODE_CHARS} characters")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be an integer between 1 and {MAX_TIMEOUT_SECONDS}")

    allowed = sorted(_normalize_allowed_imports(allowed_imports))
    work_path = _resolve_work_dir(work_dir, data_root)
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(target=_worker, args=(code, allowed, str(work_path), result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        raise TimeoutError("code execution timed out")

    try:
        payload = result_queue.get(timeout=1)
    except queue.Empty as exc:
        raise SandboxExecutionError("sandbox process exited without returning a result") from exc

    if payload.get("status") != "success":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        message = error.get("message") or "sandbox execution failed"
        raise SandboxExecutionError(f"{error.get('type', 'SandboxError')}: {message}")

    return {
        "stdout": str(payload.get("stdout", ""))[:MAX_STDOUT_CHARS],
        "stderr": str(payload.get("stderr", ""))[:MAX_STDERR_CHARS],
        "result": payload.get("result"),
        "sandbox": {
            "work_dir": Path(work_dir or "sandbox").as_posix(),
            "timeout_seconds": timeout_seconds,
            "allowed_imports": allowed,
        },
    }
