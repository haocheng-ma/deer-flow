# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, grpc
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    VectorParams,
)

from src.config.loader import get_str_env
from src.rag.retrieve.retriever import Chunk, RetrievedDocument, Retriever
from src.rag.types import ChunkDocWithVector, Resource
from src.rag.embed import get_embedder, get_embedding_dimension

logger = logging.getLogger(__name__)

SCROLL_SIZE = 64


class QdrantProvider(Retriever):
    def __init__(self) -> None:
        self.location: str = get_str_env("QDRANT_LOCATION", ":memory:")
        self.api_key: str = get_str_env("QDRANT_API_KEY", "")
        self.collection_name: str = get_str_env("QDRANT_COLLECTION", "documents")

        top_k_raw = get_str_env("QDRANT_TOP_K", "10")
        self.top_k: int = int(top_k_raw) if top_k_raw.isdigit() else 10

        self.embedding_dim: int = get_embedding_dimension()

        self._embedder = None

        self.client: Any = None
        self.vector_store: Any = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _ensure_collection_exists(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)

    def _string_to_uuid(self, text: str) -> str:
        namespace = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace, text))

    def _scroll_all_points(
        self,
        scroll_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Any]:
        results = []
        next_offset = None
        stop_scrolling = False

        while not stop_scrolling:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=SCROLL_SIZE,
                offset=next_offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and getattr(next_offset, "num", 0) == 0
                and getattr(next_offset, "uuid", "") == ""
            )
            results.extend(points)

        return results

    def _connect(self) -> None:
        client_kwargs = {"location": self.location}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        self.client = QdrantClient(**client_kwargs)

        self._ensure_collection_exists()

        try:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self._get_embedder(),
            )
        except Exception:
            self.vector_store = None

    def _get_embedding(self, text: str) -> List[float]:
        return self._get_embedder().embed_query(text=text.strip())

    def list_resources(self, query: Optional[str] = None) -> List[Resource]:
        resources: List[Resource] = []

        if not self.client:
            try:
                self._connect()
            except Exception:
                return []

        try:
            if query and self.vector_store:
                docs = self.vector_store.similarity_search(query, k=100)
                for d in docs:
                    meta = d.metadata or {}
                    uri = meta.get("url", "") or f"qdrant://{meta.get('id', '')}"
                    if any(r.uri == uri for r in resources):
                        continue
                    resources.append(
                        Resource(
                            uri=uri,
                            title=meta.get("title", "") or meta.get("id", "Unnamed"),
                            description="Stored Qdrant document",
                        )
                    )
            else:
                all_points = self._scroll_all_points(
                    with_payload=True,
                    with_vectors=False,
                )

                for point in all_points:
                    payload = point.payload or {}
                    doc_id = payload.get("doc_id", str(point.id))
                    uri = payload.get("url", "") or f"qdrant://{doc_id}"
                    resources.append(
                        Resource(
                            uri=uri,
                            title=payload.get("title", "") or doc_id,
                            description="Stored Qdrant document",
                        )
                    )

            logger.info(
                "Successfully listed %d resources from Qdrant collection: %s",
                len(resources),
                self.collection_name,
            )
        except Exception:
            logger.warning("Failed to query Qdrant for resources: %s", self.collection_name)
            return []
        return resources

    async def list_resources_async(self, query: Optional[str] = None) -> List[Resource]:
        """Async list_resources via asyncio.to_thread."""
        return await asyncio.to_thread(self.list_resources, query)

    def query_relevant_documents(
        self, query: str, resources: Optional[List[Resource]] = None
    ) -> List[RetrievedDocument]:
        resources = resources or []
        if not self.client:
            self._connect()

        query_embedding = self._get_embedding(query)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        ).points

        documents = {}

        for result in search_results:
            payload = result.payload or {}
            doc_id = payload.get("doc_id", str(result.id))
            content = payload.get("content", "")
            title = payload.get("title", "")
            url = payload.get("url", "")
            score = result.score

            if resources:
                doc_in_resources = False
                for resource in resources:
                    if (url and url in resource.uri) or doc_id in resource.uri:
                        doc_in_resources = True
                        break
                if not doc_in_resources:
                    continue

            if doc_id not in documents:
                documents[doc_id] = RetrievedDocument(id=doc_id, url=url, title=title, chunks=[])

            chunk = Chunk(content=content, similarity=score)
            documents[doc_id].chunks.append(chunk)

        return list(documents.values())

    async def query_relevant_documents_async(
        self, query: str, resources: Optional[List[Resource]] = None
    ) -> List[RetrievedDocument]:
        """Async query_relevant_documents via asyncio.to_thread."""
        return await asyncio.to_thread(
            self.query_relevant_documents, query, resources
        )

    def ingest_chunks(
        self,
        chunks_with_vectors: List[ChunkDocWithVector],
        resource_metadata: dict,
    ) -> Resource:
        raise NotImplementedError("Ingest not supported for provider: qdrant")

    def create_collection(self) -> None:
        if not self.client:
            self._connect()
        else:
            self._ensure_collection_exists()

    def close(self) -> None:
        if hasattr(self, "client") and self.client:
            try:
                if hasattr(self.client, "close"):
                    self.client.close()
                self.client = None
                self.vector_store = None
            except Exception as e:
                logger.warning("Exception occurred while closing QdrantProvider: %s", e)

    def __del__(self) -> None:
        self.close()
