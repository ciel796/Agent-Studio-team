
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys

from pathlib import Path
from typing import Optional

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.text_utils import compare_text, compute_keyword_score, extract_snippet, extract_words, generate_summary

from typing import TypedDict

class MemoryPaths(TypedDict):
    root: Path
    global_dir: Path
    conversations: Path
    index: Path
    max_chars: int


# ==================== 向量检索函数 ====================

def load_memory_with_vector(
    config_path: str,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.01,
    use_global_only: bool = True,
    outdir: Optional[str] = None,
) -> dict:
    """使用本地哈希词袋向量检索记忆。"""
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])

    if use_global_only:
        filtered_index = {
            k: v for k, v in index.items()
            if isinstance(v, dict) and v.get("memory_type") == "global"
        }
    else:
        filtered_index = index
    results = _search_by_vector(
        filtered_index,
        query,
        paths["root"],
        top_k=top_k,
        min_similarity=min_similarity,
    )

    result = {
        "status": "success",
        "query": query,
        "search_mode": "vector",
        "vector_method": "hashed_bag_of_terms_cosine",
        "vector_dimensions": VECTOR_DIMENSIONS,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "use_global_only": use_global_only,
        "results": results,
        "total_matched": len(results),
        "errors": []
    }
    
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "search_vector",
                "mode": "vector",
                "query": query,
                "count": len(results),
                "top_k": top_k,
                "vector_method": "hashed_bag_of_terms_cosine",
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result


def _extract_relevant_snippet(content: str, query: str, context_chars: int = 120) -> str:
    """提取与查询最相关的文本片段"""
    if not query or not content:
        return content[:200] if content else ""
    
    query_words = extract_words(query)
    if not query_words:
        return content[:200] + "..." if len(content) > 200 else content
    
    sentences = content.replace('\\n', ' ').split('。')
    best_sentence = ""
    best_score = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_words = extract_words(sent)
        if not sent_words:
            continue
        matched = sum(1 for w in query_words if w in sent_words)
        if matched > best_score:
            best_score = matched
            best_sentence = sent
    
    if best_sentence and len(best_sentence) > 50:
        return best_sentence[:context_chars] + "..." if len(best_sentence) > context_chars else best_sentence
    
    return content[:context_chars] + "..." if len(content) > context_chars else content

def _memory_paths(config_path: str | Path) -> MemoryPaths:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return {
        "root": root,
        "global_dir": root / memory["global_memory_dir"],  # 使用 global_dir
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "max_chars": max_chars,
    }


def _read_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
) -> dict:
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    ordered_ids = []
    if use_global_memory:
        ordered_ids.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    ordered_ids.extend(selected_memory_ids)
    ordered_ids = list(dict.fromkeys(ordered_ids))

    docs = []
    errors = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for memory_id in ordered_ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        relative_path = metadata.get("path")
        if not isinstance(relative_path, str):
            errors.append({"memory_id": memory_id, "type": "InvalidMetadata", "message": "memory path is missing"})
            continue
        document_path = (paths["root"] / relative_path).resolve()
        try:
            document_path.relative_to(paths["root"].resolve())
        except ValueError:
            errors.append({"memory_id": memory_id, "type": "InvalidPath", "message": "memory path escapes root"})
            continue
        if not document_path.is_file():
            errors.append({"memory_id": memory_id, "type": "FileNotFoundError", "message": f"memory file not found: {relative_path}"})
            continue
        original = read_text(document_path)
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        if included:
            docs.append(
                {
                    "memory_id": memory_id,
                    "memory_type": metadata.get("memory_type"),
                    "title": metadata.get("title", memory_id),
                    "summary": metadata.get("summary", ""),
                    "path": relative_path,
                    "content": included,
                    "original_chars": len(original),
                    "included_chars": len(included),
                    "truncated": truncated,
                }
            )
            remaining -= len(included)
    if errors and docs:
        status = "partial"
    elif errors:
        status = "error"
    else:
        status = "success"
    result = {
        "status": status,
        "query": query,
        "selected_memory_docs": docs,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": errors,
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "selected_ids": [item["memory_id"] for item in docs],
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    return result


# ==================== 进阶功能1: 关键词检索 ====================

def _search_by_keywords(index: dict, query: str, root: Path, top_k: int = 5, min_score: float = 0.05) -> list:
    """
    按关键词检索排序，返回最相关的前k个记忆文档
    """
    if not query:
        return []
    
    results = []
    
    for memory_id, metadata in index.items():
        if not isinstance(metadata, dict):
            continue
            
        relative_path = metadata.get("path")
        if not relative_path:
            continue
            
        document_path = (root / relative_path).resolve()
        if not document_path.is_file():
            continue
        
        try:
            content = read_text(document_path)
        except Exception:
            continue
        
        # 计算匹配分数（标题权重1.5倍）
        title_score = compute_keyword_score(metadata.get("title", ""), query) * 1.5
        content_score = compute_keyword_score(content, query)
        total_score = max(title_score, content_score)
        
        if total_score >= min_score:
            # 提取匹配片段
            snippet = extract_snippet(content, query, 80)
            
            results.append({
                "memory_id": memory_id,
                "memory_type": metadata.get("memory_type"),
                "title": metadata.get("title", memory_id),
                "path": relative_path,
                "score": round(total_score, 4),
                "snippet": snippet,
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
                "matched_chars": len(content)
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


VECTOR_DIMENSIONS = 256


def _hash_token(token: str, dimensions: int = VECTOR_DIMENSIONS) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _vector_terms(text: str) -> list[str]:
    words = extract_words(text)
    terms: list[str] = []
    for word in words:
        terms.append(word)
        if len(word) >= 4:
            terms.extend(word[index : index + 2] for index in range(len(word) - 1))
    if terms:
        return terms
    compact = re.sub(r"\s+", "", text.lower())
    return [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]


def _hashed_text_vector(text: str, dimensions: int = VECTOR_DIMENSIONS) -> dict[int, float]:
    vector: dict[int, float] = {}
    for term in _vector_terms(text):
        index = _hash_token(term, dimensions)
        vector[index] = vector.get(index, 0.0) + 1.0
    return vector


def _cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _search_by_vector(
    index: dict,
    query: str,
    root: Path,
    top_k: int = 5,
    min_similarity: float = 0.01,
    max_chars_per_doc: int = 2000,
) -> list:
    """使用本地哈希词袋向量返回相似度最高的记忆文档。"""
    if not query:
        return []

    query_vector = _hashed_text_vector(query)
    results = []
    for memory_id, metadata in index.items():
        if not isinstance(metadata, dict):
            continue
        relative_path = metadata.get("path")
        if not relative_path:
            continue
        document_path = (root / relative_path).resolve()
        if not document_path.is_file():
            continue
        try:
            content = read_text(document_path)
        except Exception:
            continue
        
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc]

        searchable_text = "\n".join(
            str(value or "")
            for value in (metadata.get("title"), metadata.get("summary"), content)
        )
        similarity = _cosine_similarity(query_vector, _hashed_text_vector(searchable_text))
        if similarity >= min_similarity:
            snippet = _extract_relevant_snippet(content, query, 120)
            results.append({
                "memory_id": memory_id,
                "memory_type": metadata.get("memory_type"),
                "title": metadata.get("title", memory_id),
                "path": relative_path,
                "similarity": round(similarity, 4),
                "snippet": snippet,
                "content_preview": content[:300] + "..." if len(content) > 300 else content,
                "matched_chars": len(content)
            })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ==================== 进阶功能2: 增强版load_memory ====================

def load_memory_advanced(
    config_path: str,
    selected_memory_ids: Optional[list[str]] = None,
    use_global_memory: bool = False,
    query: Optional[str] = None,
    top_k: int = 5,
    search_mode: str = "auto",  # "id", "keyword", "vector", "auto"
    outdir: Optional[str] = None,
) -> dict:
    """
    增强版 load_memory，支持关键词检索
    
    参数:
        search_mode: 
            "id" - 按ID查找（原有功能）
            "keyword" - 关键词检索
            "auto" - 有query且无ID时自动使用关键词检索
    """
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    # 自动模式：有query且没有指定ID时，使用关键词检索
    if search_mode == "auto":
        if query and not selected_memory_ids:
            search_mode = "keyword"
        else:
            search_mode = "id"
    
    # 关键词检索模式
    if search_mode == "keyword" and query:
        search_results = _search_by_keywords(index, query, paths["root"], top_k)
        result = {
            "status": "success",
            "query": query,
            "search_mode": "keyword",
            "top_k": top_k,
            "results": search_results,
            "total_matched": len(search_results),
            "errors": []
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "selected_memory.json")
            append_jsonl(
                {
                    "timestamp": now_iso(),
                    "operation": "search",
                    "mode": "keyword",
                    "query": query,
                    "count": len(search_results)
                },
                output_dir / "memory_log.jsonl",
            )
        return result

    if search_mode == "vector" and query:
        search_results = _search_by_vector(index, query, paths["root"], top_k)
        result = {
            "status": "success",
            "query": query,
            "search_mode": "vector",
            "top_k": top_k,
            "vector_method": "hashed_bag_of_terms_cosine",
            "vector_dimensions": VECTOR_DIMENSIONS,
            "results": search_results,
            "total_matched": len(search_results),
            "errors": []
        }
        if outdir:
            output_dir = Path(outdir)
            write_json(result, output_dir / "selected_memory.json")
            append_jsonl(
                {
                    "timestamp": now_iso(),
                    "operation": "search",
                    "mode": "vector",
                    "query": query,
                    "count": len(search_results),
                    "vector_method": "hashed_bag_of_terms_cosine",
                },
                output_dir / "memory_log.jsonl",
            )
        return result
    
    # ID查找模式（原有功能）
    return load_memory(
        config_path,
        selected_memory_ids or [],
        use_global_memory,
        query,
        outdir
    )


# ==================== 进阶功能3: 增强版保存（带摘要压缩） ====================

def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def _dedupe_summary_items(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", str(item or "")).strip(" -\t\r\n")
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _conversation_summary(messages: list[dict], previous_summary: str = "", max_chars: int = 800) -> str:
    directive_markers = (
        "请记住", "记住", "每次", "每一次", "以后", "后续", "要求", "偏好", "必须", "不要",
        "结尾", "回复时", "回答时", "优先", "always", "remember", "prefer", "must", "should", "do not",
    )
    context_markers = (
        "我是", "我的", "项目", "背景", "目标", "正在", "使用", "名字", "地点", "预算", "公司", "系统",
        "i am", "my ", "project", "goal", "context", "budget",
    )
    trivial_patterns = (
        r"^\s*[\d.]+\s*[+\-*/×÷%]\s*[\d.]+\s*(?:[=＝]\s*[?？]?)?\s*$",
        r"^\s*(?:你好|您好|嗨|hello|hi|why|为什么)\s*[?？!！]*\s*$",
        r"^\s*(?:how should i calculate|calculate|计算)\s+[\d.]+\s*[+\-*/×÷%]\s*[\d.]+\s*[?？]*\s*$",
    )
    directives: list[str] = []
    contexts: list[str] = []
    recent: list[str] = []
    user_messages = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    for content in user_messages:
        compact = re.sub(r"\s+", " ", content).strip()
        lowered = compact.casefold()
        if any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in trivial_patterns):
            continue
        if any(marker.casefold() in lowered for marker in directive_markers):
            directives.append(compact)
            continue
        if any(marker.casefold() in lowered for marker in context_markers):
            contexts.append(compact)
            continue
        recent.append(compact)

    previous_directives = []
    if previous_summary:
        for line in previous_summary.splitlines():
            stripped = line.strip().lstrip("-").strip()
            if stripped.startswith("#") or stripped in {"用户要求与偏好", "关键上下文", "近期有效任务", "未决事项"}:
                continue
            lowered = stripped.casefold()
            if stripped and any(marker.casefold() in lowered for marker in directive_markers):
                previous_directives.append(stripped)

    directives = _dedupe_summary_items([*previous_directives, *directives])
    contexts = _dedupe_summary_items(contexts)
    recent = [item for item in _dedupe_summary_items(recent) if item.casefold() not in {x.casefold() for x in directives + contexts}]

    unresolved = []
    if messages:
        last = messages[-1] if isinstance(messages[-1], dict) else {}
        last_content = str(last.get("content") or "").strip()
        if last.get("role") == "assistant" and (
            last_content.startswith("ASK_USER:")
            or "请澄清" in last_content
            or "请补充" in last_content
            or "could you please clarify" in last_content.casefold()
        ):
            unresolved.append(last_content)

    sections = [
        ("用户要求与偏好", directives),
        ("关键上下文", contexts),
        ("近期有效任务", recent[-3:]),
        ("未决事项", unresolved),
    ]
    lines = []
    for title, items in sections:
        if not items:
            continue
        lines.append(f"## {title}")
        lines.extend(f"- {item}" for item in items)
    if not lines:
        return "暂无需要跨轮次保留的要求、上下文或关键信息。"
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 14].rstrip() + "\n- …（已压缩）"


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    paths = _memory_paths(config_path)
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global_dir"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    
    # 生成更有意义的title（从用户消息中提取）
    title = f"{save_type.title()} {conversation_id}"
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_content = msg.get("content", "")
            if user_content:
                title_parts = extract_words(user_content)[:8]
                if title_parts:
                    title = " ".join(title_parts[:5])
                    if len(title) > 40:
                        title = title[:40] + "..."
                    break
    
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    previous_summary = existing.get("summary", "") if isinstance(existing, dict) else ""
    summary = (
        _conversation_summary(messages, previous_summary)
        if save_type == "conversation"
        else generate_summary(answer, 200)
    )
    
    markdown = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{now}`\n\n"
        "## Summary\n\n"
        f"{summary}\n\n"
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )
    write_text(markdown, target_path)
    created_at = existing.get("created_at", now)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
    }
    write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {"timestamp": now, "operation": "save", "status": "success", "memory_id": memory_id},
            output_dir / "memory_log.jsonl",
        )
    return result


# ==================== 进阶功能4: 记忆更新与冲突管理 ====================

def update_memory(
    config_path: str,
    memory_id: str,
    new_messages_path: str,
    new_trace_path: str,
    new_answer_path: str,
    conflict_strategy: str = "merge",  # merge | replace | skip | ask
    outdir: Optional[str] = None,
) -> dict:
    """
    更新已存在的记忆文档
    
    参数:
        conflict_strategy:
            merge - 合并新旧内容
            replace - 用新内容替换
            skip - 如果有冲突则跳过
            ask - 返回冲突信息供人工决策
    """
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    # 检查记忆是否存在
    if memory_id not in index:
        raise ValueError(f"Memory ID '{memory_id}' not found")
    
    existing_metadata = index[memory_id]
    old_path = paths["root"] / existing_metadata["path"]
    
    if not old_path.exists():
        raise FileNotFoundError(f"Old memory file not found: {old_path}")
    
    # 读取新旧内容
    old_content = read_text(old_path)
    new_messages = read_json(new_messages_path)
    new_trace = read_json(new_trace_path)
    new_answer = read_text(new_answer_path).strip()
    
    # 对比分析
    comparison = compare_text(old_content, new_answer)
    
    # 根据策略处理冲突
    if conflict_strategy == "skip" and comparison["change_type"] in ["conflict", "replace"]:
        return {
            "status": "skipped",
            "memory_id": memory_id,
            "change_type": comparison["change_type"],
            "comparison": comparison,
            "message": f"Skipped due to {comparison['change_type']} (overlap: {comparison['overlap_ratio']})"
        }
    
    if conflict_strategy == "ask":
        return {
            "status": "need_decision",
            "memory_id": memory_id,
            "change_type": comparison["change_type"],
            "comparison": comparison,
            "old_content_preview": old_content[:500],
            "new_content_preview": new_answer[:500],
            "message": "Manual decision required. Use --conflict_strategy merge/replace to continue"
        }
    
    # 合并或替换：生成新文档
    now = now_iso()
    target_dir = paths["conversations"] if existing_metadata.get("memory_type") == "conversation" else paths["global_dir"]
    relative_dir = "conversations" if existing_metadata.get("memory_type") == "conversation" else "global"
    
    # 生成合并后的内容
    if conflict_strategy == "merge":
        # 合并：保留旧内容 + 补充新内容
        merged_content = _merge_memory_content(old_content, new_answer, comparison)
        final_answer = merged_content
        change_note = "merged"
    else:  # replace
        final_answer = new_answer
        change_note = "replaced"
    
    # 重新生成标题
    title = existing_metadata.get("title", f"Updated {memory_id}")
    summary = generate_summary(final_answer, 200)
    
    # 保存新文档
    target_path = Path(target_dir) / f"{existing_metadata.get('conversation_id', memory_id)}.md"
    relative_path = f"{relative_dir}/{existing_metadata.get('conversation_id', memory_id)}.md"
    
    markdown = (
        f"# {title} (Updated)\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- updated_at: `{now}`\n"
        f"- change_type: `{change_note}`\n"
        f"- previous_overlap: `{comparison['overlap_ratio']}`\n\n"
        "## Summary\n\n"
        f"{summary}\n\n"
        "## Final Answer\n\n"
        f"{final_answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(new_messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(new_trace, ensure_ascii=False, indent=2)}\n```\n"
    )
    write_text(markdown, target_path)
    
    # 更新索引
    index[memory_id] = {
        **existing_metadata,
        "updated_at": now,
        "summary": summary,
        "change_type": change_note,
        "previous_overlap": comparison["overlap_ratio"],
        "path": relative_path,
    }
    write_json(index, paths["index"])
    
    result = {
        "status": "updated",
        "memory_id": memory_id,
        "change_type": change_note,
        "comparison": comparison,
        "conflict_strategy": conflict_strategy,
        "path": relative_path,
        "updated_at": now,
    }
    
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "updated_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "update",
                "status": "updated",
                "memory_id": memory_id,
                "change_type": change_note,
                "overlap": comparison["overlap_ratio"]
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result


def _merge_memory_content(old_content: str, new_content: str, comparison: dict) -> str:
    """合并新旧内容"""
    if comparison["change_type"] == "supplement":
        # 补充：合并新旧内容
        return f"{old_content}\n\n---\n\n## Additional Information\n\n{new_content}"
    elif comparison["change_type"] == "conflict":
        # 冲突：标记差异
        return f"{old_content}\n\n---\n\n## Conflicting Information (Manual Review Needed)\n\n{new_content}"
    else:
        # 其他情况：直接替换
        return new_content


# ==================== 进阶功能5: 错误记忆影响分析 ====================

def analyze_bad_memory_impact(
    config_path: str,
    bad_memory_id: str,
    good_memory_id: str,
    query: str,
    outdir: Optional[str] = None,
) -> dict:
    """
    分析错误记忆对回答的影响
    对比使用错误记忆 vs 正确记忆时的差异
    """
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    def load_memory_content(memory_id: str) -> dict:
        metadata = index.get(memory_id)
        if not metadata:
            return {"error": f"Memory {memory_id} not found"}
        document_path = paths["root"] / metadata["path"]
        if not document_path.exists():
            return {"error": f"File not found: {document_path}"}
        content = read_text(document_path)
        return {
            "memory_id": memory_id,
            "title": metadata.get("title"),
            "content": content,
            "length": len(content),
            "type": metadata.get("memory_type")
        }
    
    bad = load_memory_content(bad_memory_id)
    good = load_memory_content(good_memory_id)
    
    # 计算差异
    impact = {
        "bad_memory_id": bad_memory_id,
        "good_memory_id": good_memory_id,
        "bad_exists": "error" not in bad,
        "good_exists": "error" not in good,
        "analysis": {}
    }
    
    if "error" not in bad and "error" not in good:
        # 计算内容重叠度
        overlap = compare_text(bad["content"], good["content"])
        
        # 计算与查询的相关性
        bad_score = compute_keyword_score(bad["content"], query)
        good_score = compute_keyword_score(good["content"], query)
        
        impact["analysis"] = {
            "content_overlap": overlap,
            "bad_relevance_score": round(bad_score, 4),
            "good_relevance_score": round(good_score, 4),
            "length_difference": bad["length"] - good["length"],
            "type_difference": bad.get("type") != good.get("type"),
            "recommendation": (
                "Use good_memory_id" if good_score > bad_score else 
                "Use bad_memory_id may be better" if bad_score > good_score else
                "Similar relevance, review manually"
            )
        }
    else:
        impact["analysis"]["error"] = bad.get("error") or good.get("error")
    
    if outdir:
        output_dir = Path(outdir)
        write_json(impact, output_dir / "memory_impact_analysis.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "analyze",
                "bad_memory_id": bad_memory_id,
                "good_memory_id": good_memory_id,
                "recommendation": impact["analysis"].get("recommendation")
            },
            output_dir / "memory_log.jsonl",
        )
    
    return impact


# ==================== 命令行入口 ====================

def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    # 进阶参数：关键词检索
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results for keyword search")
    parser.add_argument("--search_mode", choices=["id", "keyword", "vector", "auto"], default="auto",
                        help="Search mode: id, keyword, vector, or auto")
    # 保存参数
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    # 更新参数
    parser.add_argument("--update_memory_id", help="Memory ID to update")
    parser.add_argument("--update_messages_path", help="Path to new messages JSON")
    parser.add_argument("--update_trace_path", help="Path to new trace JSON")
    parser.add_argument("--update_answer_path", help="Path to new answer")
    parser.add_argument("--conflict_strategy", choices=["merge", "replace", "skip", "ask"], default="merge",
                        help="Strategy for handling conflicts during update")
    # 分析参数
    parser.add_argument("--analyze_bad", help="Bad memory ID for impact analysis")
    parser.add_argument("--analyze_good", help="Good memory ID for impact analysis")
    parser.add_argument("--analyze_query", help="Query for impact analysis")
    # 向量检索参数
    parser.add_argument("--min_similarity", type=float, default=0.01,
                        help="Minimum similarity threshold for vector search (0-1)")
    parser.add_argument("--use_global_only", type=parse_bool, default=True,
                        help="Only search global memory for vector search")
    # 输出
    parser.add_argument("--outdir", required=True)
    return parser




def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        
        # ===== 保存模式 =====
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            result = save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(outdir / "saved_memory.json")
            return 0
        
        # ===== 更新模式 =====
        if args.update_memory_id:
            if not all([args.update_messages_path, args.update_trace_path, args.update_answer_path]):
                raise ValueError("Update requires --update_messages_path, --update_trace_path, --update_answer_path")
            result = update_memory(
                str(config_path),
                args.update_memory_id,
                args.update_messages_path,
                args.update_trace_path,
                args.update_answer_path,
                args.conflict_strategy,
                str(outdir),
            )
            print(outdir / "updated_memory.json")
            return 0
        
        # ===== 分析模式 =====
        if args.analyze_bad and args.analyze_good:
            if not args.analyze_query:
                raise ValueError("Analysis requires --analyze_query")
            result = analyze_bad_memory_impact(
                str(config_path),
                args.analyze_bad,
                args.analyze_good,
                args.analyze_query,
                str(outdir),
            )
            print(outdir / "memory_impact_analysis.json")
            return 0
        
        
        # ===== 向量检索模式 =====
        if args.search_mode == "vector":
            if not args.query:
                raise ValueError("Vector search requires --query")
            result = load_memory_with_vector(
                str(config_path),
                args.query,
                args.top_k,
                args.min_similarity,
                bool(args.use_global_only) if args.use_global_only is not None else True,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
            return 0

        # ===== 查找模式 =====
        if args.select_memory_ids is None and args.use_global_memory is None and args.query is None:
            raise ValueError("Need --select_memory_ids, --use_global_memory, or --query")
        
        # 使用增强版 load_memory_advanced
        result = load_memory_advanced(
            str(config_path),
            args.select_memory_ids or [],
            bool(args.use_global_memory) if args.use_global_memory is not None else False,
            args.query,
            args.top_k,
            args.search_mode,
            str(outdir),
        )
        print(outdir / "selected_memory.json")
        return 0
        
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
