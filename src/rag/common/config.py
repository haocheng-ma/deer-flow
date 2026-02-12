# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG pipeline config: INGESTION_PIPELINE and RAG_EMBED from conf.yaml (cwd-relative). RAG_EMBEDDING_* env overrides at runtime."""

from typing import Any, Dict

from src.config.loader import load_yaml_config

# Model name -> vector dimension; must match provider docs. Add entry when supporting a new embed model.
EMBEDDING_MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-v4": 2048,
}

DEFAULT_EMBEDDING_DIM = 1536


def get_embedding_dimension_for_model(model: str | None, explicit_dim: int | None) -> int:
    """Return embedding dimension; explicit_dim overrides. Used by embed and retrieve for consistency."""
    if explicit_dim is not None and explicit_dim > 0:
        return explicit_dim
    return EMBEDDING_MODEL_DIMENSIONS.get((model or "").strip(), DEFAULT_EMBEDDING_DIM)


def _get_config_path(config_path: str | None = None) -> str:
    """Config file path; default conf.yaml (cwd-relative). Caller may pass explicit path."""
    if config_path is None:
        return "conf.yaml"
    return config_path


def load_ingestion_pipeline_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load INGESTION_PIPELINE section from conf.yaml. RAG Embedding is configured via .env only (RAG_EMBEDDING_*)."""
    raw = load_yaml_config(_get_config_path(config_path))
    return raw.get("INGESTION_PIPELINE", {})
