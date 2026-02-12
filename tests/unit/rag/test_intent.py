# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for RAG intent detection (detect_rag_intent)."""

from unittest.mock import patch

import pytest

from src.rag import intent as intent_mod


def _reset_phrase_cache():
    """Reset phrase cache so next detect_rag_intent uses fresh config."""
    intent_mod._cached_phrases = None


@pytest.fixture(autouse=True)
def reset_cache():
    _reset_phrase_cache()
    yield
    _reset_phrase_cache()


# --- detect_rag_intent: positive cases (default enabled) ---


def test_detect_rag_intent_chinese_positive():
    """Chinese: '请结合我上传的论文进行分析' -> True."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("请结合我上传的论文进行分析") is True
        assert intent_mod.detect_rag_intent("用我的文档总结一下") is True
        assert intent_mod.detect_rag_intent("根据知识库回答") is True


def test_detect_rag_intent_english_positive():
    """English: 'Search my local documents for X' -> True."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("Search my local documents for X") is True
        assert intent_mod.detect_rag_intent("use my uploads") is True
        assert intent_mod.detect_rag_intent("based on my papers") is True


def test_detect_rag_intent_uri_positive():
    """URI prefixes rag://, milvus:// -> True."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("see rag://collection/doc") is True
        assert intent_mod.detect_rag_intent("milvus://default/foo") is True


# --- false positives (no single-word 'local') ---


def test_detect_rag_intent_no_false_positive_local_weather():
    """'local weather', 'local news' -> False (no single-word local)."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("What is the local weather") is False
        assert intent_mod.detect_rag_intent("local news today") is False


def test_detect_rag_intent_unrelated():
    """Unrelated sentence -> False."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("What is the weather") is False
        assert intent_mod.detect_rag_intent("Explain quantum computing") is False


# --- kill switch ---


def test_detect_rag_intent_disabled_returns_false():
    """When ENABLE_RAG_INTENT_DETECTION is false, always False."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=False):
        assert intent_mod.detect_rag_intent("请结合我上传的论文") is False
        assert intent_mod.detect_rag_intent("my documents") is False


# --- empty / invalid input ---


def test_detect_rag_intent_empty():
    """Empty or None text -> False."""
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("") is False
        assert intent_mod.detect_rag_intent(None) is False
        assert intent_mod.detect_rag_intent("   ") is False


def test_detect_rag_intent_non_string():
    """Non-string input -> False."""
    assert intent_mod.detect_rag_intent(123) is False


# --- conf RAG_INTENT_PATTERNS merge ---


def test_detect_rag_intent_conf_keywords_merged():
    """When conf has RAG_INTENT_PATTERNS.keywords, they are used in addition to defaults."""
    _reset_phrase_cache()
    conf = {
        "RAG_INTENT_PATTERNS": {
            "keywords": ["custom rag phrase", "扩展词"],
        },
    }
    with patch.object(intent_mod, "load_yaml_config", return_value=conf):
        intent_mod._load_phrases()
    with patch.object(intent_mod, "_rag_intent_enabled", return_value=True):
        assert intent_mod.detect_rag_intent("my documents") is True
        assert intent_mod.detect_rag_intent("custom rag phrase") is True
        assert intent_mod.detect_rag_intent("请用扩展词") is True
