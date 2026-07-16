from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _clean_text(value: str) -> str:
    if not any("\ud800" <= char <= "\udfff" for char in value):
        return value
    try:
        return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeError:
        return value.encode("utf-8", "replace").decode("utf-8")


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {_clean_json_value(key): _clean_json_value(item) for key, item in value.items()}
    return value


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(_clean_text(text), encoding="utf-8")
    try:
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def write_text(text: str, path: str | Path) -> Path:
    return _atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | Path) -> Path:
    text = json.dumps(_clean_json_value(obj), ensure_ascii=False, indent=2) + "\n"
    return _atomic_write_text(path, text)


def read_yaml(path: str | Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    text = "".join(json.dumps(_clean_json_value(record), ensure_ascii=False) + "\n" for record in records)
    return _atomic_write_text(path, text)


def append_jsonl(record: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_clean_json_value(record), ensure_ascii=False) + "\n")
    return target
