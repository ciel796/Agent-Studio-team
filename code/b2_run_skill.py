from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import DEFAULT_DATA_ROOT, bootstrap_project_root, resolve_cli_path
from common.errors import error_to_dict
from common.schemas import make_skill_result
from common.skill_limits import enforce_skill_limits


bootstrap_project_root()


SKILL_MODULES = {
    "calculator": "skills.calculator",
    "code_executor": "skills.code_executor",
    "file_reader": "skills.file_reader",
    "local_file_search": "skills.local_file_search",
    "read_convert_file": "skills.read_convert_file",
    "table_analyzer": "skills.table_analyzer",
    "format_converter": "skills.format_converter",
}


def run_skill(skill_name: str, input_data: dict, data_root: str | None = None, output_dir: str | None = None) -> dict:
    if skill_name not in SKILL_MODULES:
        raise ValueError(f"unknown skill: {skill_name}")
    if not isinstance(input_data, dict):
        raise ValueError("skill input must be a JSON object")
    start = perf_counter()
    try:
        enforce_skill_limits(skill_name, input_data)
        module = importlib.import_module(SKILL_MODULES[skill_name])
        function = getattr(module, skill_name)
        kwargs = dict(input_data)
        signature = inspect.signature(function)
        if "data_root" in signature.parameters:
            kwargs["data_root"] = data_root or str(DEFAULT_DATA_ROOT)
        if "output_dir" in signature.parameters:
            kwargs["output_dir"] = output_dir
        output = function(**kwargs)
        status = "success"
        error = None
    except Exception as exc:  # Skill exceptions are a structured business result.
        output = None
        status = "error"
        error = error_to_dict(exc)
    latency_ms = round((perf_counter() - start) * 1000, 3)
    return make_skill_result(skill_name, status, input_data, output, error, latency_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local Agent skill.")
    parser.add_argument("--skill", required=True, choices=sorted(SKILL_MODULES))
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--data_root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = resolve_cli_path(args.input)
        outdir = resolve_cli_path(args.outdir)
        input_data = read_json(input_path)
        data_root = str(resolve_cli_path(args.data_root)) if args.data_root else None
        outdir.mkdir(parents=True, exist_ok=True)
        result = run_skill(args.skill, input_data, data_root, str(outdir))
        result_path = outdir / f"{args.skill}_result.json"
        write_json(result, result_path)
        append_jsonl(
            {
                "timestamp": now_iso(),
                "skill_name": args.skill,
                "status": result["status"],
                "result_path": str(result_path),
                "latency_ms": result["latency_ms"],
            },
            outdir / "skill_run_log.jsonl",
        )
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
