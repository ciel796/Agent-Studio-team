from __future__ import annotations

from skills.file_reader import file_reader
from skills.format_converter import format_converter


MAX_READ_CHARS = 8000


def read_convert_file(
    path: str,
    target_format: str,
    max_chars: int = 2000,
    output_filename: str | None = None,
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Read a local file and convert its content with the existing format converter."""

    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if max_chars > MAX_READ_CHARS:
        raise ValueError(f"max_chars exceeds limit: {max_chars} > {MAX_READ_CHARS}")
    read_result = file_reader(path, max_chars=max_chars, data_root=data_root)
    convert_result = format_converter(
        read_result["content"],
        target_format=target_format,
        output_filename=output_filename,
        output_dir=output_dir,
    )
    return {
        "source": read_result["source"],
        "read": {
            "num_chars": read_result["num_chars"],
            "truncated": read_result["truncated"],
        },
        "conversion": convert_result,
    }
