# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for rag.embed (get_embedder, embed_documents/embed_query). Spec: U-embed-01 to 07."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.embed import clear_embedder_cache, get_embedder


def _make_mock_embedder(dim=8):
    m = MagicMock()
    m.embed_documents = MagicMock(return_value=[[0.1] * dim for _ in range(2)])
    m.embed_query = MagicMock(return_value=[0.1] * dim)
    return m


# --- U-embed-01: get_embedder(config=None) loads from .env only (_embed_config_from_env) ---
@patch("src.rag.embed._embed_config_from_env")
@patch("src.rag.embed.create_openai_embedder")
def test_get_embedder_none_uses_env_config(mock_create, mock_embed_config_from_env):
    mock_embed_config_from_env.return_value = {"provider": "openai", "model": "x", "api_key": "sk-mock", "base_url": "", "dimensions": 1536, "batch_size": 0}
    mock_create.return_value = _make_mock_embedder()
    emb = get_embedder(None)
    assert emb is not None
    assert hasattr(emb, "embed_documents")
    assert hasattr(emb, "embed_query")
    mock_embed_config_from_env.assert_called_once()


# --- U-embed-02: get_embedder with explicit config ---
@patch("src.rag.embed.create_openai_embedder")
def test_get_embedder_explicit_config_returns_embedder(mock_create):
    mock_create.return_value = _make_mock_embedder()
    config = {"provider": "openai", "model": "text-embedding-3-small", "api_key": "sk-mock"}
    emb = get_embedder(config)
    assert emb is not None
    mock_create.assert_called_once()


# --- U-embed-03: embed_documents output shape ---
@patch("src.rag.embed.create_openai_embedder")
def test_embed_documents_shape(mock_create):
    clear_embedder_cache()
    mock_emb = _make_mock_embedder(dim=1536)
    mock_create.return_value = mock_emb
    emb = get_embedder({"provider": "openai", "model": "x", "api_key": "sk-mock"})
    out = emb.embed_documents(["a", "b"])
    assert len(out) == 2
    assert len(out[0]) == 1536
    assert len(out[1]) == 1536
    assert all(isinstance(x, float) for x in out[0])


# --- U-embed-04: embed_query and embed_documents dimension consistent ---
@patch("src.rag.embed.create_openai_embedder")
def test_embed_query_dimension_consistent(mock_create):
    clear_embedder_cache()
    mock_emb = _make_mock_embedder(dim=1536)
    mock_create.return_value = mock_emb
    emb = get_embedder({"provider": "openai", "model": "x", "api_key": "sk-mock"})
    q = emb.embed_query("q")
    docs = emb.embed_documents(["a"])
    assert len(q) == len(docs[0])
    assert len(q) == 1536


# --- U-embed-05: provider=dashscope branch ---
@patch("src.rag.embed.DashscopeEmbeddings")
def test_get_embedder_dashscope_returns_dashscope(mock_dashscope_class):
    mock_dashscope_class.return_value = _make_mock_embedder()
    config = {"provider": "dashscope", "model": "xxx", "api_key": "mock"}
    emb = get_embedder(config)
    assert emb is not None
    mock_dashscope_class.assert_called_once()


# --- U-embed-06: get_embedder(None) uses .env (RAG_EMBEDDING_*) via _embed_config_from_env ---
@patch("src.rag.embed.get_int_env")
@patch("src.rag.embed.get_str_env")
def test_get_embedder_none_uses_env(mock_get_str_env, mock_get_int_env):
    mock_get_str_env.side_effect = lambda k, d="": {"RAG_EMBEDDING_PROVIDER": "openai", "RAG_EMBEDDING_MODEL": "", "RAG_EMBEDDING_API_KEY": "", "RAG_EMBEDDING_BASE_URL": ""}.get(k, d)
    mock_get_int_env.return_value = 0
    with patch("src.rag.embed.create_openai_embedder") as mock_create:
        mock_create.return_value = _make_mock_embedder()
        get_embedder(None)
    assert mock_get_str_env.call_count >= 1


# --- U-embed-07: [edge case] invalid provider ---
@patch("src.rag.embed.create_openai_embedder")
def test_get_embedder_invalid_provider_falls_back_to_openai(mock_create):
    mock_create.return_value = _make_mock_embedder()
    emb = get_embedder({"provider": "invalid", "model": "x", "api_key": "sk-mock"})
    assert emb is not None
    mock_create.assert_called_once()


# --- Spec 1.3: Embedder cache, lookup_embed_batch, clear_embedder_cache ---
@patch("src.rag.embed._embed_config_from_env")
@patch("src.rag.embed.create_openai_embedder")
def test_get_embedder_cache_reuse(mock_create, mock_embed_config_from_env):
    """Same config from env: two get_embedder(None) return same cached instance."""
    mock_embed_config_from_env.return_value = {"provider": "openai", "model": "m", "api_key": "k", "base_url": "", "dimensions": 1536, "batch_size": 0}
    mock_create.return_value = _make_mock_embedder()
    clear_embedder_cache()
    emb1 = get_embedder(None)
    emb2 = get_embedder(None)
    assert emb1 is emb2
    mock_create.assert_called_once()


@patch("src.rag.embed._embed_config_from_env")
@patch("src.rag.embed.create_openai_embedder")
def test_clear_embedder_cache_forces_new_instance(mock_create, mock_embed_config_from_env):
    """After clear_embedder_cache, next get_embedder creates new instance."""
    mock_embed_config_from_env.return_value = {"provider": "openai", "model": "m", "api_key": "k", "base_url": "", "dimensions": 1536, "batch_size": 0}
    mock_create.side_effect = [_make_mock_embedder(), _make_mock_embedder()]
    clear_embedder_cache()
    emb1 = get_embedder(None)
    clear_embedder_cache()
    emb2 = get_embedder(None)
    assert emb1 is not emb2
    assert mock_create.call_count == 2


def test_lookup_embed_batch_exact_and_prefix_order():
    """lookup_embed_batch: exact match and prefix order v4→v3→v2→v1."""
    from src.rag.embed import lookup_embed_batch

    assert lookup_embed_batch("text-embedding-v4", "dashscope") == 10
    assert lookup_embed_batch("text-embedding-v3", "dashscope") == 10
    assert lookup_embed_batch("text-embedding-v2-turbo", "dashscope") == 25
    assert lookup_embed_batch("text-embedding-v1", "dashscope") == 25
    assert lookup_embed_batch("text-embedding-3-small", "openai") == 512
    assert lookup_embed_batch("unknown", "openai") == 512
    assert lookup_embed_batch("unknown", "dashscope") == 10


# --- Dashscope empty input short-circuit (moved from test_milvus, belongs in embed) ---
def test_dashscope_embed_documents_empty_returns_empty():
    """DashscopeEmbeddings.embed_documents([]) returns [] without calling API."""
    from src.rag.embed.dashscope import DashscopeEmbeddings

    # Mock client so create() is never called for empty input
    class NoCallClient:
        class _Emb:
            def create(self, *a, **k):
                raise AssertionError("Should not be called for empty input")

        embeddings = _Emb()

    emb = DashscopeEmbeddings(model="m", api_key="k")
    emb._client = NoCallClient()  # type: ignore[assignment]
    assert emb.embed_documents([]) == []
