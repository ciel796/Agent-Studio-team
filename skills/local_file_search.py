from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass
from pathlib import Path

from skills import resolve_data_path


DEFAULT_FILE_TYPES = {"txt", "md", "py", "json", "yaml", "yml", "csv", "tsv", "log"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}


@dataclass(frozen=True)
class SearchHit:
    path: str
    score: float
    snippet: str
    line_number: int | None
    match_count: int
    match_type: str


def _normalize_file_types(file_types: list[str] | None) -> set[str]:
    raw_types = file_types or sorted(DEFAULT_FILE_TYPES)
    normalized = {item.lower().lstrip(".").strip() for item in raw_types if isinstance(item, str) and item.strip()}
    if not normalized:
        raise ValueError("file_types must contain at least one extension")
    unsupported = normalized - DEFAULT_FILE_TYPES
    if unsupported:
        raise ValueError(f"local_file_search does not support file types: {', '.join(sorted(unsupported))}")
    return normalized


def _matches_globs(relative_path: str, include_globs: list[str] | None, exclude_globs: list[str] | None) -> bool:
    if include_globs and not any(fnmatch.fnmatch(relative_path, pattern) for pattern in include_globs):
        return False
    if exclude_globs and any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude_globs):
        return False
    return True


def _is_excluded(path: Path, search_root: Path) -> bool:
    try:
        parts = path.relative_to(search_root).parts
    except ValueError:
        return True
    return any(part in DEFAULT_EXCLUDED_DIRS for part in parts[:-1])


def _candidate_files(
    search_root: Path,
    file_types: set[str],
    include_globs: list[str] | None,
    exclude_globs: list[str] | None,
    max_files: int,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(search_root.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file() or _is_excluded(path, search_root):
            continue
        relative_path = path.relative_to(search_root).as_posix()
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in file_types:
            continue
        if not _matches_globs(relative_path, include_globs, exclude_globs):
            continue
        files.append(path)
    return files


def _read_text_prefix(path: Path, max_file_bytes: int) -> tuple[str, bool]:
    with path.open("rb") as handle:
        raw = handle.read(max_file_bytes + 1)
    truncated = len(raw) > max_file_bytes
    if truncated:
        raw = raw[:max_file_bytes]
    return raw.decode("utf-8", errors="replace"), truncated


def _query_terms(query: str) -> list[str]:
    lowered = query.casefold()
    chunks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", lowered)
    unique_terms: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if re.fullmatch(r"[a-z0-9_]+", chunk):
            candidates = [chunk] if len(chunk) >= 2 and not chunk.isdigit() else []
        else:
            candidates = []
            if len(chunk) <= 4:
                candidates.append(chunk)
            for gram_size in (4, 3, 2):
                if len(chunk) < gram_size:
                    continue
                for index in range(len(chunk) - gram_size + 1):
                    candidates.append(chunk[index : index + gram_size])
        for term in candidates:
            if (
                not term
                or len(term) <= 1
                or term.isdigit()
                or term in unique_terms
            ):
                continue
            unique_terms.append(term)
    return unique_terms


def _term_weights(file_payloads: list[tuple[str, str]], terms: list[str]) -> dict[str, float]:
    if not file_payloads or not terms:
        return {}

    corpus_size = len(file_payloads)
    weights: dict[str, float] = {}
    for term in terms:
        document_frequency = 0
        for relative_path, lowered_text in file_payloads:
            if term in relative_path or term in lowered_text:
                document_frequency += 1
        weights[term] = 1.0 + math.log((corpus_size + 1) / (document_frequency + 1))
    return weights


def _snippet(text: str, start: int, end: int, max_snippet_chars: int) -> str:
    radius = max(40, max_snippet_chars // 2)
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\r\n", "\n").replace("\r", "\n")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet[: max_snippet_chars + 6]


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _score_file(
    relative_path: str,
    text: str,
    query: str,
    term_weights: dict[str, float],
    max_snippet_chars: int,
) -> SearchHit | None:
    lowered_text = text.casefold()
    lowered_path = relative_path.casefold()
    phrase = query.casefold().strip()

    score = 0.0
    match_count = 0
    best_start = -1
    best_end = -1
    match_type = "token"

    if phrase:
        phrase_hits = lowered_text.count(phrase)
        path_phrase_hits = lowered_path.count(phrase)
        if phrase_hits:
            best_start = lowered_text.find(phrase)
            best_end = best_start + len(phrase)
            score += phrase_hits * 12
            match_count += phrase_hits
            match_type = "exact"
        if path_phrase_hits:
            score += path_phrase_hits * 8
            match_count += path_phrase_hits
            if best_start < 0:
                match_type = "path"

    for term, weight in term_weights.items():
        text_hits = lowered_text.count(term)
        path_hits = lowered_path.count(term)
        if text_hits and best_start < 0:
            best_start = lowered_text.find(term)
            best_end = best_start + len(term)
        score += text_hits * (2.0 * weight) + path_hits * (3.0 * weight)
        match_count += text_hits + path_hits

    if score <= 0:
        return None

    if best_start >= 0:
        snippet = _snippet(text, best_start, best_end, max_snippet_chars)
        line_number = _line_number(text, best_start)
    else:
        snippet = relative_path
        line_number = None

    score = round(score + min(2.0, 2000 / max(len(text), 1)), 3)
    return SearchHit(relative_path, score, snippet, line_number, match_count, match_type)


def local_file_search(
    query: str,
    root_dir: str = ".",
    file_types: list[str] | None = None,
    top_k: int = 5,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_files: int = 500,
    max_file_bytes: int = 1_000_000,
    max_snippet_chars: int = 300,
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer between 1 and 50")
    if not isinstance(max_files, int) or isinstance(max_files, bool) or not 1 <= max_files <= 1000:
        raise ValueError("max_files must be an integer between 1 and 1000")
    if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or not 1024 <= max_file_bytes <= 10000000:
        raise ValueError("max_file_bytes must be an integer between 1024 and 10000000")
    if not isinstance(max_snippet_chars, int) or isinstance(max_snippet_chars, bool) or not 80 <= max_snippet_chars <= 800:
        raise ValueError("max_snippet_chars must be an integer between 80 and 800")

    search_root, data_root_path = resolve_data_path(root_dir, data_root)
    if not search_root.is_dir():
        raise FileNotFoundError(f"search directory not found: {root_dir}")

    normalized_types = _normalize_file_types(file_types)
    terms = _query_terms(query)
    files = _candidate_files(search_root, normalized_types, include_globs, exclude_globs, max_files)

    file_payloads: list[tuple[Path, str, str]] = []
    skipped_files = 0
    truncated_files = 0
    for path in files:
        try:
            text, truncated = _read_text_prefix(path, max_file_bytes)
        except OSError:
            skipped_files += 1
            continue
        if "\x00" in text[:4096]:
            skipped_files += 1
            continue
        if truncated:
            truncated_files += 1
        relative_path = path.relative_to(data_root_path).as_posix()
        file_payloads.append((path, relative_path, text))

    term_weights = _term_weights([(relative_path.casefold(), text.casefold()) for _, relative_path, text in file_payloads], terms)
    results: list[SearchHit] = []
    for _, relative_path, text in file_payloads:
        hit = _score_file(relative_path, text, query, term_weights, max_snippet_chars)
        if hit:
            results.append(hit)

    results.sort(key=lambda item: (-item.score, item.path))
    return {
        "query": query,
        "root_dir": root_dir,
        "files_scanned": len(files),
        "total_matches": len(results),
        "truncated_files": truncated_files,
        "skipped_files": skipped_files,
        "results": [
            {
                "path": item.path,
                "score": item.score,
                "snippet": item.snippet,
                "line_number": item.line_number,
                "match_count": item.match_count,
                "match_type": item.match_type,
            }
            for item in results[:top_k]
        ],
    }
