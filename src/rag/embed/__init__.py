# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Embedding: get_embedder() returns cached embedder (thread-safe). Config from .env only (RAG_EMBEDDING_*)."""

import hashlib
import json
import threading
from typing import Any, Dict

from src.config.loader import get_int_env, get_str_env
from src.rag.common.config import get_embedding_dimension_for_model
from src.rag.embed.dashscope import DashscopeEmbeddings
from src.rag.embed.openai import create_openai_embedder

_embedder_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

EMBED_BATCH_DEFAULTS: Dict[str, int] = {
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 512,
    "text-embedding-ada-002": 512,
    "text-embedding-v4": 10,
    "text-embedding-v3": 10,
    "text-embedding-v2": 25,
    "text-embedding-v1": 25,
}


def _embed_env(key_suffix: str, default: str = "") -> str:
    """Read RAG_EMBEDDING_<key_suffix> from env."""
    return get_str_env(f"RAG_EMBEDDING_{key_suffix}", default)


def _embed_env_int(default: int = 0) -> int:
    """Read RAG_EMBEDDING_DIM from env."""
    return get_int_env("RAG_EMBEDDING_DIM", default)


def _resolve_embed_config(config: dict | None) -> dict:
    """Resolve config: passed-in dict, or from .env (RAG_EMBEDDING_*) only."""
    if config is not None and config:
        return config
    return _embed_config_from_env()


def _embed_cache_key(config: dict) -> str:
    """Cache key from provider, model, base_url (no api_key)."""
    safe = {
        k: config.get(k)
        for k in ("provider", "model", "base_url")
        if config.get(k) is not None and config.get(k) != ""
    }
    return hashlib.md5(json.dumps(safe, sort_keys=True).encode()).hexdigest()


def _effective_embed_dim(config: dict) -> int:
    """Single source for embedding dimension: config + env, then model-based default."""
    explicit = config.get("dimensions")
    if explicit is None:
        explicit = _embed_env_int(0)
    if not explicit or explicit <= 0:
        model = (config.get("model") or _embed_env("MODEL", "") or "").strip()
        return get_embedding_dimension_for_model(model or "", None)
    return explicit


def _create_embedder(config: dict) -> Any:
    """Create embedder instance from config."""
    provider = (config.get("provider") or "").strip().lower() or _embed_env("PROVIDER", "openai")
    model = config.get("model") or _embed_env("MODEL", "")
    api_key = config.get("api_key") or _embed_env("API_KEY", "")
    base_url = config.get("base_url") or _embed_env("BASE_URL", "")
    dimensions = _effective_embed_dim(config)
    if provider == "dashscope":
        return DashscopeEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            encoding_format="float",
        )
    return create_openai_embedder(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        dimensions=dimensions,
        encoding_format="float",
    )


def _lookup_embed_batch(model: str, provider: str) -> int:
    """Return batch size for model (exact then prefix); default openai 512, dashscope 10."""
    model = (model or "").strip()
    provider = (provider or "").strip().lower()
    if model and model in EMBED_BATCH_DEFAULTS:
        return EMBED_BATCH_DEFAULTS[model]
    for prefix in ("text-embedding-v4", "text-embedding-v3", "text-embedding-v2", "text-embedding-v1"):
        if model.startswith(prefix):
            return EMBED_BATCH_DEFAULTS.get(prefix, 10)
    return 512 if provider == "openai" else 10


def lookup_embed_batch(model: str, provider: str) -> int:
    """Public API: return default embed batch size for model/provider. Used by pipeline when RAG_EMBEDDING_BATCH_SIZE is unset."""
    return _lookup_embed_batch(model, provider)


def clear_embedder_cache() -> None:
    """Clear embedder cache (tests or config hot reload)."""
    with _cache_lock:
        _embedder_cache.clear()


def get_embedding_dimension(config: dict | None = None) -> int:
    """Return vector dimension for current embed config. Retrieve uses this for schema match."""
    cfg = _resolve_embed_config(config) or {}
    return _effective_embed_dim(cfg)


def get_embedder(config: dict | None = None) -> Any:
    """Return embedder (embed_documents, embed_query). Cached by config."""
    config = _resolve_embed_config(config) or {}
    key = _embed_cache_key(config)
    with _cache_lock:
        if key not in _embedder_cache:
            _embedder_cache[key] = _create_embedder(config)
        return _embedder_cache[key]


def _embed_config_from_env() -> dict:
    """Build embed config from RAG_EMBEDDING_* env vars (.env only)."""
    return {
        "provider": _embed_env("PROVIDER", "openai"),
        "model": _embed_env("MODEL", ""),
        "api_key": _embed_env("API_KEY", ""),
        "base_url": _embed_env("BASE_URL", ""),
        "dimensions": _embed_env_int(0),
        "batch_size": get_int_env("RAG_EMBEDDING_BATCH_SIZE", 0),
    }


def get_embed_config(config: dict | None = None) -> dict:
    """Return effective embed config (for callers that need batch_size etc.). From .env when config is None."""
    return _resolve_embed_config(config) or {}
