# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
RAG intent detection: detect when user expresses interest in local/uploaded documents.
Phrase-based only (no single-word triggers like "local" to avoid false positives).
"""

import logging
import re
from typing import Dict, List, Tuple

from src.config.loader import get_bool_env, load_yaml_config

logger = logging.getLogger(__name__)

# Total kill switch: when False, detect_rag_intent always returns False
_ENABLE_KEY = "ENABLE_RAG_INTENT_DETECTION"
_CONF_PATH = "conf.yaml"
_RAG_PATTERNS_KEY = "RAG_INTENT_PATTERNS"

# Default phrases (no single-word "local" or "my" to avoid false positives)
_DEFAULT_EN_PHRASES: Tuple[str, ...] = (
    "uploaded",
    "my documents",
    "my papers",
    "my files",
    "my uploads",
    "rag",
    "knowledge base",
    "private docs",
    "local search",
    "local files",
    "local documents",
    "local knowledge",
    "from my documents",
    "in my papers",
    "use my files",
    "search my uploads",
    "based on my documents",
    "based on my papers",
    "based on my files",
)
_DEFAULT_ZH_PHRASES: Tuple[str, ...] = (
    "上传",
    "我的文档",
    "我的论文",
    "我的文件",
    "知识库",
    "私有",
    "本地搜索",
    "本地文件",
    "本地文档",
    "结合本地",
    "结合上传",
    "结合我的",
)
# URI prefixes (user mentions RAG resources)
_URI_PATTERNS = (re.compile(r"rag://", re.I), re.compile(r"milvus://", re.I))

# Cache: (en_phrases_lower, zh_phrases) loaded once
_cached_phrases: Tuple[List[str], Tuple[str, ...]] | None = None


def _rag_intent_enabled() -> bool:
    """Read ENABLE_RAG_INTENT_DETECTION: conf first, then env. Default True."""
    conf = load_yaml_config(_CONF_PATH)
    val = conf.get(_ENABLE_KEY)
    if val is not None:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(val)
    return get_bool_env(_ENABLE_KEY, True)


def _load_phrases() -> Tuple[List[str], Tuple[str, ...]]:
    """Load phrase lists (from conf or defaults). Cached on first call."""
    global _cached_phrases
    if _cached_phrases is not None:
        return _cached_phrases
    conf = load_yaml_config(_CONF_PATH)
    patterns = conf.get(_RAG_PATTERNS_KEY) or {}
    extra_keywords: List[str] = []
    if isinstance(patterns, dict):
        kw = patterns.get("keywords")
        if isinstance(kw, list):
            extra_keywords = [str(x).strip() for x in kw if x]
    en = [p.lower() for p in _DEFAULT_EN_PHRASES]
    zh = list(_DEFAULT_ZH_PHRASES)
    for phrase in extra_keywords:
        if not phrase:
            continue
        en.append(phrase.lower())
        # Add to zh only if not already ascii (avoid duplicating en phrases in zh scan)
        if not phrase.isascii():
            zh.append(phrase)
    _cached_phrases = (en, tuple(zh))
    return _cached_phrases


def detect_rag_intent(text: str | None) -> bool:
    """
    Detect whether the user expresses intent to use local/uploaded documents (RAG).
    Uses phrase-based matching only; no single-word triggers like "local".
    Returns False if ENABLE_RAG_INTENT_DETECTION is false or text is empty.
    """
    if not _rag_intent_enabled():
        logger.debug("RAG intent detection disabled by %s", _ENABLE_KEY)
        return False
    if not text or not isinstance(text, str):
        return False
    text_stripped = text.strip()
    if not text_stripped:
        return False
    en_phrases, zh_phrases = _load_phrases()
    text_lower = text_stripped.lower()
    for phrase in en_phrases:
        if phrase in text_lower:
            logger.debug("RAG intent detected (en phrase): %r", phrase)
            logger.info("RAG intent detected")
            return True
    for phrase in zh_phrases:
        if phrase in text_stripped:
            logger.debug("RAG intent detected (zh phrase): %r", phrase)
            logger.info("RAG intent detected")
            return True
    for pattern in _URI_PATTERNS:
        if pattern.search(text_stripped):
            logger.debug("RAG intent detected (uri prefix)")
            logger.info("RAG intent detected")
            return True
    return False
