# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Dashscope embeddings (OpenAI-compatible API); shared by pipeline and retrieve."""

from typing import List, Sequence

from openai import OpenAI


class DashscopeEmbeddings:
    """OpenAI-compatible embeddings wrapper (e.g. Dashscope)."""

    def __init__(self, **kwargs) -> None:
        self._client = OpenAI(
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", ""),
        )
        self._model = kwargs.get("model", "")
        self._encoding_format = kwargs.get("encoding_format", "float")

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not clean_texts:
            return []
        resp = self._client.embeddings.create(
            model=self._model,
            input=clean_texts,
            encoding_format=self._encoding_format,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)
