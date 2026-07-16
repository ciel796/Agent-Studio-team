"""
文本处理工具模块
用于关键词检索、摘要生成、文本对比、向量检索等
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Any, Optional, Union

# ==================== 基础文本处理 ====================

def extract_words(text: str) -> List[str]:
    """提取中英文单词"""
    if not text:
        return []
    return re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text.lower())


def compute_keyword_score(text: str, query: str) -> float:
    """
    计算文本与查询的关键词匹配分数
    使用简化的TF-IDF思想：词频 + 位置加权
    """
    if not query or not text:
        return 0.0
    
    query_words = extract_words(query)
    text_words = extract_words(text)
    
    if not query_words or not text_words:
        return 0.0
    
    text_counter = Counter(text_words)
    
    matched = 0.0
    total_weight = 0.0
    
    for i, word in enumerate(query_words):
        weight = 1.0 / (i + 1)  # 位置权重，靠前的词权重更高
        total_weight += weight
        
        if word in text_counter:
            # 词频贡献
            tf = text_counter[word] / len(text_words)
            # 位置贡献
            try:
                pos = text_words.index(word)
                position_score = 1.0 / (pos + 1)
            except ValueError:
                position_score = 0
            matched += weight * (1 + 0.3 * tf + 0.2 * position_score)
    
    if total_weight == 0:
        return 0.0
    
    base_score = matched / total_weight
    # 长度归一化：避免短文本得分虚高
    length_norm = min(1.0, len(text_words) / 50)
    return base_score * (0.7 + 0.3 * length_norm)


def generate_summary(text: str, max_len: int = 200) -> str:
    """生成文本摘要（不依赖LLM的规则版本）"""
    if not text:
        return ""
    
    text = text.strip()
    
    if len(text) <= max_len:
        return text
    
    # 按句子分割
    sentences = re.split(r'[。！？.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text[:max_len] + "..."
    
    # 取前几句作为摘要
    summary = ""
    for sent in sentences[:3]:
        if len(summary) + len(sent) + 2 <= max_len:
            summary += sent + "。"
        else:
            break
    
    if not summary:
        summary = text[:max_len]
    elif len(summary) < max_len * 0.4 and len(sentences) > 3:
        # 如果摘要太短，补充关键词
        keywords = extract_words(text)[:10]
        if keywords:
            summary += f" 关键词: {', '.join(keywords[:5])}"
    
    if len(summary) > max_len:
        summary = summary[:max_len] + "..."
    
    return summary


def compare_text(old_text: str, new_text: str) -> Dict[str, Any]:
    """对比两段文本，识别变更类型"""
    if not old_text:
        return {"change_type": "new", "details": "No existing content"}
    
    old_words = set(extract_words(old_text))
    new_words = set(extract_words(new_text))
    
    if not old_words:
        return {"change_type": "new", "details": "Old text has no words"}
    
    overlap = old_words & new_words
    old_only = old_words - new_words
    new_only = new_words - old_words
    
    overlap_ratio = len(overlap) / len(old_words) if old_words else 0
    
    if overlap_ratio > 0.8:
        change_type = "supplement"
    elif overlap_ratio > 0.4:
        change_type = "conflict"
    elif len(new_only) > 0 and overlap_ratio < 0.3:
        change_type = "replace"
    else:
        change_type = "unknown"
    
    return {
        "change_type": change_type,
        "overlap_ratio": round(overlap_ratio, 3),
        "old_word_count": len(old_words),
        "new_word_count": len(new_words),
        "old_only": list(old_only)[:10],
        "new_only": list(new_only)[:10],
        "overlap_words": list(overlap)[:10]
    }


def extract_snippet(content: str, query: str, context_chars: int = 100) -> str:
    """提取包含查询关键词的文本片段"""
    if not query or not content:
        return content[:200] if content else ""
    
    query_words = extract_words(query)
    if not query_words:
        return content[:200]
    
    for word in query_words[:3]:
        if word in content.lower():
            idx = content.lower().find(word)
            start = max(0, idx - context_chars)
            end = min(len(content), idx + context_chars + len(word))
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet
    
    return content[:200] + "..."


# ==================== 向量检索工具（可选依赖） ====================

# 向量检索依赖状态
_HAS_VECTOR_SUPPORT = False

try:
    # type: ignore
    import numpy as np  # type: ignore
    # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_VECTOR_SUPPORT = True
except ImportError:
    np = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    _HAS_VECTOR_SUPPORT = False

# 全局模型缓存
_embedding_model = None
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_vector_support_available() -> bool:
    """检查向量检索是否可用"""
    return _HAS_VECTOR_SUPPORT


def get_embedding_model(model_name: Optional[str] = None):
    """
    获取嵌入模型（单例模式，延迟加载）
    """
    global _embedding_model
    
    if not _HAS_VECTOR_SUPPORT:
        raise ImportError(
            "Vector search requires numpy and sentence-transformers. "
            "Run: pip install numpy sentence-transformers"
        )
    
    if _embedding_model is None:
        name = model_name or _EMBEDDING_MODEL_NAME
        _embedding_model = SentenceTransformer(name)  # type: ignore
    return _embedding_model


def compute_embedding(text: str, model_name: Optional[str] = None):
    """
    计算文本的向量嵌入
    """
    if not _HAS_VECTOR_SUPPORT:
        raise ImportError(
            "Vector search requires numpy and sentence-transformers. "
            "Run: pip install numpy sentence-transformers"
        )
    
    if not text:
        return np.zeros(384)  # type: ignore
    
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=True)  # type: ignore


def compute_similarity(embedding1, embedding2) -> float:
    """
    计算两个向量的余弦相似度（已归一化）
    """
    if not _HAS_VECTOR_SUPPORT:
        raise ImportError("numpy is required for vector similarity")
    return float(np.dot(embedding1, embedding2))  # type: ignore


def batch_compute_embeddings(texts: List[str], model_name: Optional[str] = None):
    """
    批量计算文本的向量嵌入
    """
    if not _HAS_VECTOR_SUPPORT:
        raise ImportError(
            "Vector search requires numpy and sentence-transformers. "
            "Run: pip install numpy sentence-transformers"
        )
    
    if not texts:
        return np.array([])  # type: ignore
    
    model = get_embedding_model(model_name)
    # 限制单个文本长度，避免OOM
    truncated_texts = [t[:2000] for t in texts]
    return model.encode(truncated_texts, normalize_embeddings=True)  # type: ignore