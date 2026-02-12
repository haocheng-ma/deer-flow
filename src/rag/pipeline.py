# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG orchestration: Parse → Chunk → Embed → Store. Batch size and temp cleanup see run_index."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.rag.chunk import get_chunker
from src.rag.common.config import load_ingestion_pipeline_config
from src.rag.common.utils import sanitize_filename
from src.rag.embed import get_embed_config, get_embedder, lookup_embed_batch
from src.rag.parse import get_parser
from src.rag.retrieve import build_retriever
from src.rag.types import ChunkDocWithVector, Document, Resource

logger = logging.getLogger(__name__)

Stage = Literal["parse", "chunk", "embed", "store"]
# MinerU API batch limit; do not increase without checking MinerU docs.
MAX_MINERU_BATCH = 200


def _ensure_path(
    item: Union[Path, Tuple[bytes, str]], temp_dir: Path
) -> Tuple[Path, bool]:
    """Convert item to Path; if bytes, write to temp and return (path, True) so caller can clean up."""
    if isinstance(item, tuple):
        content, filename = item
        safe_name = sanitize_filename(filename)
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        path.write_bytes(content)
        return (path, True)
    return (Path(item), False)


def _group_docs_by_file_id(
    docs_all: List[Document], path_to_file_id: Dict[str, str]
) -> Dict[str, List[Document]]:
    """Group documents by file_id; use metadata['filename'] as fallback when unknown and log warning."""
    out: Dict[str, List[Document]] = {}
    for doc in docs_all:
        filename = (doc.metadata or {}).get("filename", "")
        file_id = path_to_file_id.get(filename, filename)
        # Warn when we used filename as file_id but it was not in this batch's path mapping
        used_fallback_file_id = (
            bool(filename) and bool(path_to_file_id) and file_id not in path_to_file_id
        )
        if used_fallback_file_id:
            logger.warning("Unknown filename in docs, fallback: %s", filename)
        out.setdefault(file_id, []).append(doc)
    return out


class IndexSuccessEntry(BaseModel):
    """One successful file in run_index. API: { file_id, resource }."""

    file_id: str = Field(..., description="File identifier (e.g. filename)")
    resource: Resource = Field(..., description="Resource with uri, title, description")


class IndexFailedEntry(BaseModel):
    """One failed file in run_index. API: { file_id, stage, reason }."""

    file_id: str = Field(..., description="File identifier (e.g. filename)")
    stage: Stage = Field(..., description="Stage where failure occurred")
    reason: str = Field(..., description="Failure reason")


class IndexResult(BaseModel):
    """Result of run_index. Typed successes/failed for schema and validation."""

    trace_id: str = Field(..., description="Index trace id")
    successes: List[IndexSuccessEntry] = Field(
        default_factory=list,
        description="Successfully indexed files: file_id + resource (uri, title, description)",
    )
    failed: List[IndexFailedEntry] = Field(
        default_factory=list,
        description="Failed files: file_id, stage (parse/chunk/embed/store), reason",
    )


class RAGPipeline:
    """Runs Parse → Chunk → Embed → Store. If retriever is None, run_index fails all at stage 'store'."""

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_ingestion_pipeline_config()
        self._config = config or {}
        self._parser_config = self._config.get("parser") or {}
        self._chunker_config = self._config.get("chunker") or {}
        pipeline_cfg = self._config.get("pipeline") or {}
        self._temp_dir = Path(pipeline_cfg.get("temp_dir", "/tmp/deer-flow"))
        self._retriever = build_retriever()  # May be None if RAG provider not configured

    async def run_index(
        self,
        files_or_bytes_list: List[Union[Path, Tuple[bytes, str]]],
        options: Optional[dict] = None,
    ) -> IndexResult:
        """Run Parse (batches ≤200) → group by file_id → Chunk→Embed→Store per file. Per-file failures; temp files from bytes cleaned per batch."""
        opts = options or {}
        trace_id = opts.get("trace_id") or f"trace_{uuid.uuid4().hex[:12]}"
        successes: List[Tuple[str, Resource]] = []
        failed: List[Tuple[str, str, str]] = []

        if not files_or_bytes_list:
            return IndexResult(trace_id=trace_id, successes=[], failed=[])

        parser = get_parser(self._parser_config)
        chunker = get_chunker(self._chunker_config)
        embedder = get_embedder()
        pipeline_cfg = self._config.get("pipeline") or {}
        # Product limit per file; config key max_chunks_per_file (default 2000)
        max_chunks_per_file = pipeline_cfg.get("max_chunks_per_file", 2000)
        embed_cfg = get_embed_config()
        max_embed_batch = (embed_cfg or {}).get("batch_size")
        if not max_embed_batch:
            model = (embed_cfg or {}).get("model") or ""
            provider = ((embed_cfg or {}).get("provider") or "").strip().lower()
            max_embed_batch = lookup_embed_batch(model, provider)

        if self._retriever is None:
            for item in files_or_bytes_list:
                failed.append((_file_id(item), "store", "RAG provider not configured"))
            return IndexResult(
                trace_id=trace_id,
                successes=[IndexSuccessEntry(file_id=f, resource=r) for f, r in successes],
                failed=[IndexFailedEntry(file_id=f, stage=s, reason=r) for f, s, r in failed],
            )

        batches = [
            files_or_bytes_list[i : i + MAX_MINERU_BATCH]
            for i in range(0, len(files_or_bytes_list), MAX_MINERU_BATCH)
        ]
        for batch_idx, batch in enumerate(batches):
            temp_paths: List[Path] = []
            path_to_file_id: Dict[str, str] = {}
            try:
                paths: List[Path] = []
                for item in batch:
                    path, need_cleanup = _ensure_path(item, self._temp_dir)
                    paths.append(path)
                    path_to_file_id[path.name] = _file_id(item)
                    if need_cleanup:
                        temp_paths.append(path)

                logger.info(
                    "Parse batch %d/%d, files=%d",
                    batch_idx + 1,
                    len(batches),
                    len(paths),
                )
                docs_all, parse_failed = await parser.parse_files(
                    paths, temp_dir=str(self._temp_dir)
                )
                for path_name, reason in parse_failed:
                    file_id = path_to_file_id.get(path_name, path_name)
                    failed.append((file_id, "parse", reason))
                    logger.warning("Parse failed: %s, reason=%s", path_name, reason)

                by_file = _group_docs_by_file_id(docs_all, path_to_file_id)
                for file_id, docs in by_file.items():
                    current_stage: Stage = "chunk"
                    try:
                        if not docs or not any((d.content or "").strip() for d in docs):
                            failed.append((file_id, "parse", "no content"))
                            continue
                        if hasattr(chunker, "process_sync"):
                            chunked = await asyncio.to_thread(chunker.process_sync, docs)
                        else:
                            chunked = await chunker.process(docs)
                        if not chunked:
                            failed.append((file_id, "chunk", "no chunks produced"))
                            continue
                        if len(chunked) > max_chunks_per_file:
                            failed.append(
                                (
                                    file_id,
                                    "chunk",
                                    f"chunk count exceeds limit (max={max_chunks_per_file})",
                                )
                            )
                            logger.warning(
                                "Chunk limit exceeded: file_id=%s, count=%d, max=%d",
                                file_id,
                                len(chunked),
                                max_chunks_per_file,
                            )
                            continue
                        current_stage = "embed"
                        texts = [c.content for c in chunked]
                        batch_starts = range(0, len(texts), max_embed_batch)
                        embed_tasks = [
                            asyncio.to_thread(
                                embedder.embed_documents,
                                texts[i : i + max_embed_batch],
                            )
                            for i in batch_starts
                        ]
                        batch_results = await asyncio.gather(*embed_tasks)
                        vectors = [v for batch_vecs in batch_results for v in batch_vecs]
                        if len(vectors) != len(chunked):
                            failed.append((file_id, "embed", "embed count mismatch"))
                            continue
                        current_stage = "store"
                        chunks_with_vectors = [
                            ChunkDocWithVector(
                                id=c.id,
                                content=c.content,
                                metadata=dict(c.metadata),
                                vector=vec,
                            )
                            for c, vec in zip(chunked, vectors)
                        ]
                        resource_metadata = {
                            "filename": file_id,
                            "title": file_id,
                            "source": "uploaded",
                        }
                        resource = await asyncio.to_thread(
                            self._retriever.ingest_chunks,
                            chunks_with_vectors,
                            resource_metadata,
                        )
                        res = (
                            resource
                            if hasattr(resource, "model_dump")
                            else Resource(
                                uri=resource.uri,
                                title=resource.title,
                                description=getattr(resource, "description", "") or "Uploaded file",
                            )
                        )
                        successes.append((file_id, res))
                    except NotImplementedError as e:
                        failed.append((file_id, "store", str(e)))
                    except Exception as e:
                        logger.exception(
                            "Pipeline failed for %s: %s", file_id, e
                        )
                        failed.append((file_id, current_stage, str(e)))
            finally:
                for p in temp_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError as e:
                        logger.warning("Failed to remove temp file %s: %s", p, e)

        return IndexResult(
            trace_id=trace_id,
            successes=[IndexSuccessEntry(file_id=f, resource=r) for f, r in successes],
            failed=[IndexFailedEntry(file_id=f, stage=s, reason=r) for f, s, r in failed],
        )


def _file_id(item: Union[Path, Tuple[bytes, str]]) -> str:
    if isinstance(item, tuple):
        return sanitize_filename(item[1])
    return Path(item).name


def _to_resource_dict(resource: Any) -> Dict[str, Any]:
    """Resource to JSON-serializable dict. Used when building Resource from retriever.ingest_chunks result."""
    if hasattr(resource, "model_dump"):
        return resource.model_dump()
    return {
        "uri": resource.uri,
        "title": resource.title,
        "description": getattr(resource, "description", "") or "",
    }


async def run_index(
    files_or_bytes_list: List[Union[Path, Tuple[bytes, str]]],
    options: Optional[dict] = None,
    config: Optional[dict] = None,
) -> IndexResult:
    """Build RAGPipeline and run run_index."""
    pipeline = RAGPipeline(config=config)
    return await pipeline.run_index(files_or_bytes_list, options)
