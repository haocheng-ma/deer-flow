# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG Retriever: list resources, query, ingest_chunks. Only pipeline calls ingest_chunks after Parse→Chunk→Embed."""

import abc
from typing import List

from src.rag.types import ChunkDocWithVector, Resource


class Chunk:
    """Retrieved chunk (content + similarity)."""

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity


class RetrievedDocument:
    """Query result document (id, url, title, chunks). Not rag.types.Document (Parse/Chunk)."""

    def __init__(
        self,
        id: str,
        url: str | None = None,
        title: str | None = None,
        chunks: list[Chunk] | None = None,
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks or []

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": "\n\n".join([c.content for c in self.chunks]),
        }
        if self.url:
            d["url"] = self.url
        if self.title:
            d["title"] = self.title
        return d


class Retriever(abc.ABC):
    """RAG provider: list resources, query, ingest chunks."""

    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        """List resources; optional query may affect order. Returns list of Resource."""
        pass

    @abc.abstractmethod
    async def list_resources_async(self, query: str | None = None) -> list[Resource]:
        """Async list_resources."""
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self, query: str, resources: list[Resource] | None = None
    ) -> list[RetrievedDocument]:
        """Return matching documents (with chunks). If resources given, filter by those URIs."""
        pass

    @abc.abstractmethod
    async def query_relevant_documents_async(
        self, query: str, resources: list[Resource] | None = None
    ) -> list[RetrievedDocument]:
        """Async query_relevant_documents."""
        pass

    @abc.abstractmethod
    def ingest_chunks(
        self,
        chunks_with_vectors: List[ChunkDocWithVector],
        resource_metadata: dict,
    ) -> Resource:
        """Write chunks to store and return Resource. Raise NotImplementedError if not supported."""
        pass
