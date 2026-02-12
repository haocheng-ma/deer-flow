# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from langchain_milvus.vectorstores import Milvus as LangchainMilvus
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from src.config.loader import get_str_env
from src.rag.common.config import load_ingestion_pipeline_config
from src.rag.embed import get_embedder, get_embedding_dimension
from src.rag.retrieve.retriever import Chunk, RetrievedDocument, Retriever
from src.rag.types import ChunkDocWithVector, Resource

logger = logging.getLogger(__name__)


def _sanitize_metadata_for_store(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only scalar types supported by Milvus/LangChain (e.g. serialize list/dict to JSON string)."""
    out: Dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


class MilvusRetriever(Retriever):
    """Retriever backed by Milvus (Lite or remote). Embedding from RAG_EMBED / RAG_EMBEDDING_*."""

    def __init__(self) -> None:
        self.uri: str = get_str_env("MILVUS_URI", "http://localhost:19530")
        self.user: str = get_str_env("MILVUS_USER")
        self.password: str = get_str_env("MILVUS_PASSWORD")
        self.collection_name: str = get_str_env("MILVUS_COLLECTION", "documents")

        top_k_raw = get_str_env("MILVUS_TOP_K", "10")
        self.top_k: int = int(top_k_raw) if top_k_raw.isdigit() else 10

        self.vector_field: str = get_str_env("MILVUS_VECTOR_FIELD", "embedding")
        self.id_field: str = get_str_env("MILVUS_ID_FIELD", "id")
        self.content_field: str = get_str_env("MILVUS_CONTENT_FIELD", "content")
        self.title_field: str = get_str_env("MILVUS_TITLE_FIELD", "title")
        self.url_field: str = get_str_env("MILVUS_URL_FIELD", "url")
        self.metadata_field: str = get_str_env("MILVUS_METADATA_FIELD", "metadata")

        self._embedder = None
        self.embedding_dim = get_embedding_dimension()

        self.client: Any = None

    def _get_embedder(self):
        """Lazy init embedder via get_embedder()."""
        if self._embedder is None:
            self._embedder = get_embedder()
            self.embedding_model = self._embedder
        return self._embedder

    def _create_collection_schema(self) -> CollectionSchema:
        """Build Milvus CollectionSchema with vector and metadata fields."""
        fields = [
            FieldSchema(
                name=self.id_field,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=self.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
            ),
            FieldSchema(
                name=self.content_field, dtype=DataType.VARCHAR, max_length=65535
            ),
            FieldSchema(name=self.title_field, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, max_length=1024),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for DeerFlow RAG documents: {self.collection_name}",
            enable_dynamic_field=True,
        )
        return schema

    def _ensure_collection_exists(self) -> None:
        """Ensure collection exists; create if missing (Milvus Lite only; LangChain creates on first use)."""
        if self._is_milvus_lite():
            try:
                collections = self.client.list_collections()
                if self.collection_name not in collections:
                    schema = self._create_collection_schema()
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        schema=schema,
                        index_params={
                            "field_name": self.vector_field,
                            "index_type": "IVF_FLAT",
                            "metric_type": "IP",
                            "params": {"nlist": 1024},
                        },
                    )
                    logger.info("Created Milvus collection: %s", self.collection_name)

            except Exception as e:
                logger.warning("Could not ensure collection exists: %s", e)
        else:
            logger.warning(
                "Could not ensure collection exists: %s", self.collection_name
            )

    def _get_insert_batch_size(self) -> int:
        """Read from load_ingestion_pipeline_config()['store']['insert_batch_size']; default 200."""
        try:
            cfg = load_ingestion_pipeline_config()
            store = (cfg or {}).get("store") or {}
            size = store.get("insert_batch_size")
            if size is not None and isinstance(size, int) and size > 0:
                return size
        except Exception:
            pass
        return 200

    def _insert_chunk_with_vector(
        self,
        doc_id: str,
        content: str,
        title: str,
        url: str,
        metadata: Dict[str, Any],
        vector: List[float],
    ) -> None:
        """Insert one chunk with pre-computed vector (used by ingest_chunks)."""
        try:
            safe_meta = _sanitize_metadata_for_store(metadata)
            if self._is_milvus_lite():
                data = [
                    {
                        self.id_field: doc_id,
                        self.vector_field: vector,
                        self.content_field: content,
                        self.title_field: title,
                        self.url_field: url,
                        **safe_meta,
                    }
                ]
                self.client.insert(collection_name=self.collection_name, data=data)
            else:
                self.client.add_embeddings(
                    texts=[content],
                    embeddings=[vector],
                    metadatas=[
                        {
                            self.id_field: doc_id,
                            self.title_field: title,
                            self.url_field: url,
                            **safe_meta,
                        }
                    ],
                    ids=[doc_id],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to insert document chunk: {str(e)}") from e

    def _connect(self) -> None:
        """Create Milvus client (idempotent)."""
        try:
            if self._is_milvus_lite():
                self.client = MilvusClient(self.uri)
                self._ensure_collection_exists()
            else:
                connection_args = {"uri": self.uri}
                if self.user:
                    connection_args["user"] = self.user
                if self.password:
                    connection_args["password"] = self.password

                self.embedding_model = self._get_embedder()
                self.client = LangchainMilvus(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    connection_args=connection_args,
                    drop_old=False,
                )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}") from e

    def _is_milvus_lite(self) -> bool:
        """True if URI is local Milvus Lite (e.g. .db path without http(s))."""
        return self.uri.endswith(".db") or (
            not self.uri.startswith(("http://", "https://")) and "://" not in self.uri
        )

    def _get_embedding(self, text: str) -> List[float]:
        """Return embedding for text via get_embedder()."""
        try:
            if not isinstance(text, str):
                raise ValueError(f"Text must be a string, got {type(text)}")
            if not text.strip():
                raise ValueError("Text cannot be empty or only whitespace")
            embeddings = self._get_embedder().embed_query(text=text.strip())

            if not isinstance(embeddings, list) or not embeddings:
                raise ValueError(f"Invalid embedding format: {type(embeddings)}")

            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    def list_resources(self, query: Optional[str] = None) -> List[Resource]:
        """List resources. When query is None or empty, list by filter/expr only (no semantic search).
        Lite: queries by expr; remote: uses similarity_search with expr filter; query is passed as-is
        or as empty string when only listing (to satisfy SDK signature)."""
        resources: List[Resource] = []

        if not self.client:
            try:
                self._connect()
            except Exception:
                return []

        try:
            _source_filter = 'source == "uploaded"'
            if self._is_milvus_lite():
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=_source_filter,
                    output_fields=[self.id_field, self.title_field, self.url_field],
                    limit=100,
                )
                for r in results:
                    resources.append(
                        Resource(
                            uri=r.get(self.url_field, "")
                            or f"milvus://{r.get(self.id_field, '')}",
                            title=r.get(self.title_field, "")
                            or r.get(self.id_field, "Unnamed"),
                            description="Stored Milvus document",
                        )
                    )
            else:
                # When query is None or empty, use placeholder so similarity_search receives a string;
                # filtering is by expr only (list resources, no semantic search).
                _query = (query or "").strip() or " "
                docs: Iterable[Any] = self.client.similarity_search(
                    _query,
                    k=100,
                    expr=_source_filter,
                )
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    if resources and any(
                        r.uri == meta.get(self.url_field, "")
                        or r.uri == f"milvus://{meta.get(self.id_field, '')}"
                        for r in resources
                    ):
                        continue
                    resources.append(
                        Resource(
                            uri=meta.get(self.url_field, "")
                            or f"milvus://{meta.get(self.id_field, '')}",
                            title=meta.get(self.title_field, "")
                            or meta.get(self.id_field, "Unnamed"),
                            description="Stored Milvus document",
                        )
                    )
                logger.info(
                    "Succeed listed %d resources from Milvus collection: %s",
                    len(resources),
                    self.collection_name,
                )
        except Exception:
            logger.warning("Failed to query Milvus for resources: %s", self.collection_name)
            return []
        return resources

    async def _ensure_connected_async(self) -> None:
        """Ensure client is connected from a context with an event loop (main thread).
        Prevents AsyncMilvusClient init failure when sync methods run in asyncio.to_thread."""
        if not self.client:
            self._connect()

    async def list_resources_async(self, query: Optional[str] = None) -> List[Resource]:
        """Async list_resources via asyncio.to_thread."""
        await self._ensure_connected_async()
        return await asyncio.to_thread(self.list_resources, query)

    def query_relevant_documents(
        self, query: str, resources: Optional[List[Resource]] = None
    ) -> List[RetrievedDocument]:
        """Vector similarity search; returns RetrievedDocument list (each with Chunks). Optional resources filter."""
        resources = resources or []
        try:
            if not self.client:
                self._connect()

            query_embedding = self._get_embedding(query)

            if self._is_milvus_lite():
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    anns_field=self.vector_field,
                    param={"metric_type": "IP", "params": {"nprobe": 10}},
                    limit=self.top_k,
                    output_fields=[
                        self.id_field,
                        self.content_field,
                        self.title_field,
                        self.url_field,
                    ],
                )

                documents = {}

                for result_list in search_results:
                    for result in result_list:
                        entity = result.get("entity", {})
                        doc_id = entity.get(self.id_field, "")
                        content = entity.get(self.content_field, "")
                        title = entity.get(self.title_field, "")
                        url = entity.get(self.url_field, "")
                        score = result.get("distance", 0.0)

                        if resources:
                            doc_in_resources = False
                            for resource in resources:
                                if (
                                    url and url in resource.uri
                                ) or doc_id in resource.uri:
                                    doc_in_resources = True
                                    break
                            if not doc_in_resources:
                                continue

                        if doc_id not in documents:
                            documents[doc_id] = RetrievedDocument(
                                id=doc_id, url=url, title=title, chunks=[]
                            )

                        chunk = Chunk(content=content, similarity=score)
                        documents[doc_id].chunks.append(chunk)

                return list(documents.values())

            else:
                search_results = self.client.similarity_search_with_score(
                    query=query, k=self.top_k
                )

                documents = {}

                for doc, score in search_results:
                    metadata = doc.metadata or {}
                    doc_id = metadata.get(self.id_field, "")
                    title = metadata.get(self.title_field, "")
                    url = metadata.get(self.url_field, "")
                    content = doc.page_content

                    if resources:
                        doc_in_resources = False
                        for resource in resources:
                            if (url and url in resource.uri) or doc_id in resource.uri:
                                doc_in_resources = True
                                break
                        if not doc_in_resources:
                            continue

                    if doc_id not in documents:
                        documents[doc_id] = RetrievedDocument(
                            id=doc_id, url=url, title=title, chunks=[]
                        )

                    chunk = Chunk(content=content, similarity=score)
                    documents[doc_id].chunks.append(chunk)

                return list(documents.values())

        except Exception as e:
            raise RuntimeError(f"Failed to query documents from Milvus: {str(e)}") from e

    async def query_relevant_documents_async(
        self, query: str, resources: Optional[List[Resource]] = None
    ) -> List[RetrievedDocument]:
        """Async query_relevant_documents via asyncio.to_thread."""
        await self._ensure_connected_async()
        return await asyncio.to_thread(
            self.query_relevant_documents, query, resources
        )

    def create_collection(self) -> None:
        """Ensure collection exists."""
        if not self.client:
            self._connect()
        else:
            if self._is_milvus_lite():
                self._ensure_collection_exists()

    def close(self) -> None:
        """Release client (idempotent)."""
        if hasattr(self, "client") and self.client:
            try:
                if self._is_milvus_lite() and hasattr(self.client, "close"):
                    self.client.close()
                self.client = None
            except Exception:
                pass

    def _sanitize_filename(self, filename: str, max_length: int = 200) -> str:
        """Sanitize filename for doc_id/URI; keep alphanumeric, dots, hyphens, underscores."""
        sanitized = Path(filename).name
        sanitized = re.sub(r"[^\w.\-]", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_.")
        if not sanitized:
            sanitized = "unnamed_file"
        if len(sanitized) > max_length:
            parts = sanitized.rsplit(".", 1)
            if len(parts) == 2 and len(parts[1]) <= 10:
                ext = "." + parts[1]
                base = parts[0][: max_length - len(ext)]
                sanitized = base + ext
            else:
                sanitized = sanitized[:max_length]

        return sanitized

    def ingest_chunks(
        self,
        chunks_with_vectors: List[ChunkDocWithVector],
        resource_metadata: dict,
    ) -> Resource:
        """Write pre-embedded chunks to Milvus in batches; schema matches single insert."""
        if not self.client:
            self._connect()

        filename = resource_metadata.get("filename", "unnamed")
        safe_filename = self._sanitize_filename(filename)
        title = resource_metadata.get("title") or safe_filename.replace(".md", "").replace("_", " ").title()
        source = resource_metadata.get("source", "uploaded")
        timestamp = int(time.time() * 1000)
        url = f"milvus://{self.collection_name}/{safe_filename}"
        base_name = safe_filename.rsplit(".", 1)[0] if "." in safe_filename else safe_filename
        content_hash = hashlib.md5(
            f"{safe_filename}_{len(chunks_with_vectors)}_{timestamp}".encode()
        ).hexdigest()[:8]
        doc_id_prefix = f"uploaded_{base_name}_{content_hash}"
        batch_size = self._get_insert_batch_size()

        for start in range(0, len(chunks_with_vectors), batch_size):
            batch = chunks_with_vectors[start : start + batch_size]
            if self._is_milvus_lite():
                data = []
                for i, c in enumerate(batch):
                    chunk_id = c.id or (
                        f"{doc_id_prefix}_chunk_{start + i}"
                        if len(chunks_with_vectors) > 1
                        else doc_id_prefix
                    )
                    meta = dict(c.metadata) if c.metadata else {}
                    meta["source"] = source  # Force "uploaded" for list_resources filter
                    meta.setdefault("file", safe_filename)
                    meta.setdefault("timestamp", timestamp)
                    safe_meta = _sanitize_metadata_for_store(meta)
                    data.append({
                        self.id_field: chunk_id,
                        self.vector_field: c.vector,
                        self.content_field: c.content,
                        self.title_field: title,
                        self.url_field: url,
                        **safe_meta,
                    })
                try:
                    self.client.insert(collection_name=self.collection_name, data=data)
                except Exception as e:
                    raise RuntimeError(
                        f"Batch insert failed (chunks {start}-{start + len(batch)}): {e}"
                    ) from e
            else:
                texts = []
                embeddings = []
                metadatas = []
                ids_list = []
                for i, c in enumerate(batch):
                    chunk_id = c.id or (
                        f"{doc_id_prefix}_chunk_{start + i}"
                        if len(chunks_with_vectors) > 1
                        else doc_id_prefix
                    )
                    meta = dict(c.metadata) if c.metadata else {}
                    meta["source"] = source  # Force "uploaded" for list_resources filter
                    meta.setdefault("file", safe_filename)
                    meta.setdefault("timestamp", timestamp)
                    safe_meta = _sanitize_metadata_for_store(meta)
                    texts.append(c.content)
                    embeddings.append(c.vector)
                    metadatas.append({
                        self.id_field: chunk_id,
                        self.title_field: title,
                        self.url_field: url,
                        **safe_meta,
                    })
                    ids_list.append(chunk_id)
                try:
                    self.client.add_embeddings(
                        texts=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids_list,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Batch insert failed (chunks {start}-{start + len(batch)}): {e}"
                    ) from e

        return Resource(
            uri=url,
            title=title,
            description="Uploaded file",
        )

    def __del__(self) -> None:  # pragma: no cover
        """Best-effort cleanup on GC."""
        self.close()


class MilvusProvider(MilvusRetriever):
    """Backward-compatible alias for MilvusRetriever."""

    pass
