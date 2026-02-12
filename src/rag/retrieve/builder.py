# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Process-global Retriever singleton (thread-safe). Tests should call reset_retriever() for isolation."""

from src.config.tools import SELECTED_RAG_PROVIDER, RAGProvider
from src.rag.retrieve.milvus import MilvusProvider
from src.rag.retrieve.qdrant import QdrantProvider
from src.rag.retrieve.retriever import Retriever

_retriever_instance: Retriever | None = None
_retriever_provider: str | None = None


def build_retriever() -> Retriever | None:
    """Return shared Retriever for current provider; rebuilds if provider changed."""
    global _retriever_instance, _retriever_provider
    current = SELECTED_RAG_PROVIDER or None
    if _retriever_instance is None or _retriever_provider != current:
        _retriever_provider = current
        if current == RAGProvider.MILVUS.value:
            _retriever_instance = MilvusProvider()
        elif current == RAGProvider.QDRANT.value:
            _retriever_instance = QdrantProvider()
        elif current:
            raise ValueError(f"Unsupported RAG provider: {current}")
        else:
            _retriever_instance = None
    return _retriever_instance


def reset_retriever() -> None:
    """Clear shared Retriever. Call in tests or when switching provider at runtime."""
    global _retriever_instance, _retriever_provider
    _retriever_instance = None
    _retriever_provider = None
