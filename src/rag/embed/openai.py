# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI embeddings; shared by pipeline and retrieve."""

from typing import List

from langchain_openai import OpenAIEmbeddings


def create_openai_embedder(
    *,
    model: str,
    api_key: str = "",
    base_url: str = "",
    dimensions: int | None = None,
    encoding_format: str = "float",
) -> "OpenAIEmbedder":
    """Create OpenAI-compatible embedder (base_url override for Dashscope etc.)."""
    kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url or None,
        "encoding_format": encoding_format,
    }
    if dimensions is not None and dimensions > 0:
        kwargs["dimensions"] = dimensions
    return OpenAIEmbedder(OpenAIEmbeddings(**kwargs))


class OpenAIEmbedder:
    """Unified interface: embed_documents / embed_query."""

    def __init__(self, langchain_embeddings: OpenAIEmbeddings):
        self._emb = langchain_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)
