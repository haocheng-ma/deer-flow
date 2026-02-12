# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for RAGPipeline.run_index success/failure, temp file cleanup, IndexResult structure."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.common.utils import sanitize_filename
from src.rag.pipeline import IndexFailedEntry, IndexResult, IndexSuccessEntry, RAGPipeline
from src.rag.types import Document, Resource


def _make_parse_files_return(paths):
    """Return (docs, parse_failed) for parse_files(paths); docs use path.name in metadata['filename']."""
    if not paths:
        return ([], [])
    name = Path(paths[0]).name
    docs = [
        Document(id="doc1", content="Chunk one.", metadata={"filename": name}),
        Document(id="doc2", content="Chunk two.", metadata={"filename": name}),
    ]
    return (docs, [])


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse_files = AsyncMock(side_effect=lambda paths, **kw: _make_parse_files_return(paths))
    return parser


@pytest.fixture
def mock_chunker():
    chunker = MagicMock()
    chunks = [
        Document(id="doc1_chunk_0", content="Chunk one.", metadata={"chunk_index": 0}),
        Document(id="doc2_chunk_0", content="Chunk two.", metadata={"chunk_index": 0}),
    ]
    chunker.process = AsyncMock(return_value=chunks)
    chunker.process_sync = MagicMock(return_value=chunks)  # PF-2: pipeline uses to_thread(process_sync)
    return chunker


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_documents = MagicMock(return_value=[[0.1] * 8, [0.2] * 8])
    return embedder


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.ingest_chunks = MagicMock(
        return_value=Resource(uri="milvus://documents/f.pdf", title="f", description="Uploaded file")
    )
    return retriever


@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_full_success(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_chunker, mock_embedder, mock_retriever,
):
    mock_get_parser.return_value = mock_parser
    mock_get_chunker.return_value = mock_chunker
    mock_get_embedder.return_value = mock_embedder
    mock_build_retriever.return_value = mock_retriever

    config = {"parser": {"api_token": "x"}, "chunker": {"type": "token"}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([Path("/fake/f.pdf")], options=None))

    assert isinstance(result, IndexResult)
    assert result.trace_id.startswith("trace_")
    assert len(result.successes) == 1
    assert result.successes[0].file_id == "f.pdf"
    assert result.successes[0].resource.uri == "milvus://documents/f.pdf"
    assert result.failed == []
    mock_retriever.ingest_chunks.assert_called_once()


@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_parse_empty_then_failed(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_retriever,
):
    parser = MagicMock()
    parser.parse = AsyncMock(return_value=[])
    # parse_files returns docs with empty content; pipeline treats as no content and adds to failed
    parser.parse_files = AsyncMock(
        return_value=([
            Document(id="d", content="", metadata={"filename": "empty.pdf"}),
        ], [])
    )
    mock_get_parser.return_value = parser
    mock_get_chunker.return_value = MagicMock()
    mock_get_embedder.return_value = MagicMock()
    mock_build_retriever.return_value = mock_retriever

    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([Path("/fake/empty.pdf")], options=None))

    assert len(result.successes) == 0
    assert len(result.failed) == 1
    assert result.failed[0].file_id == "empty.pdf"
    assert result.failed[0].stage == "parse"
    assert "no content" in result.failed[0].reason
    mock_retriever.ingest_chunks.assert_not_called()


@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_chunk_empty_then_failed(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_retriever,
):
    mock_get_parser.return_value = mock_parser
    chunker = MagicMock()
    chunker.process = AsyncMock(return_value=[])
    chunker.process_sync = MagicMock(return_value=[])  # PF-2: pipeline prefers to_thread(process_sync)
    mock_get_chunker.return_value = chunker
    mock_get_embedder.return_value = MagicMock()
    mock_build_retriever.return_value = mock_retriever

    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([Path("/fake/n chunks.pdf")], options=None))

    assert len(result.successes) == 0
    assert len(result.failed) == 1
    assert result.failed[0].file_id == "n chunks.pdf"
    assert result.failed[0].stage == "chunk"
    assert "no chunks" in result.failed[0].reason
    mock_retriever.ingest_chunks.assert_not_called()


@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_bytes_temp_file_cleaned(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_chunker, mock_embedder, mock_retriever, tmp_path,
):
    mock_get_parser.return_value = mock_parser
    mock_get_chunker.return_value = mock_chunker
    mock_get_embedder.return_value = mock_embedder
    mock_build_retriever.return_value = mock_retriever

    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": str(tmp_path)}}
    pipeline = RAGPipeline(config=config)
    content = b"# Hello"
    result = asyncio.run(pipeline.run_index([(content, "hello.md")], options=None))

    assert len(result.successes) == 1
    assert result.successes[0].file_id == "hello.md"
    assert not (tmp_path / "hello.md").exists()


@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_store_not_implemented_in_failed(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_chunker, mock_embedder, mock_retriever,
):
    mock_get_parser.return_value = mock_parser
    mock_get_chunker.return_value = mock_chunker
    mock_get_embedder.return_value = mock_embedder
    mock_retriever.ingest_chunks.side_effect = NotImplementedError(
        "Ingest not supported for provider: qdrant"
    )
    mock_build_retriever.return_value = mock_retriever

    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([Path("/fake/f.pdf")], options=None))

    assert len(result.successes) == 0
    assert len(result.failed) == 1
    assert result.failed[0].file_id == "f.pdf"
    assert result.failed[0].stage == "store"
    assert "qdrant" in result.failed[0].reason


@patch("src.rag.pipeline.build_retriever")
def test_run_index_no_retriever_all_failed(mock_build_retriever):
    mock_build_retriever.return_value = None

    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([Path("/fake/a.pdf"), Path("/fake/b.pdf")], options=None))

    assert len(result.successes) == 0
    assert len(result.failed) == 2
    assert result.failed[0].file_id == "a.pdf"
    assert result.failed[1].file_id == "b.pdf"
    assert all(f.stage == "store" and "not configured" in f.reason for f in result.failed)


@patch("src.rag.pipeline.build_retriever")
def test_run_index_uses_options_trace_id(mock_build_retriever):
    """Caller-provided options.trace_id should flow through to IndexResult."""
    mock_build_retriever.return_value = None
    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(
        pipeline.run_index([Path("/fake/a.pdf")], options={"trace_id": "fixed-trace-123"})
    )
    assert result.trace_id == "fixed-trace-123"
    assert len(result.failed) == 1
    assert result.failed[0].file_id == "a.pdf"


# --- U-pipe-10: [Security] sanitize_filename ---
def test_sanitize_filename_path_traversal():
    out = sanitize_filename("../etc/passwd")
    assert ".." not in out
    assert out in ("passwd", "unnamed_file")


def test_sanitize_filename_nul():
    out = sanitize_filename("a\x00b")
    assert "\x00" not in out


def test_sanitize_filename_empty_returns_unnamed():
    assert sanitize_filename("") == "unnamed_file"


def test_sanitize_filename_dot_returns_unnamed():
    assert sanitize_filename(".") == "unnamed_file"


# --- U-pipe-11: run_index multi-file call order (parse_files batch) ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_multifile_parse_order(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_chunker, mock_embedder, mock_retriever,
):
    parse_paths_seen = []

    async def record_parse_files(paths, **kw):
        parse_paths_seen.extend([Path(p).name for p in paths])
        docs = []
        for p in paths:
            name = Path(p).name
            docs.append(Document(id=f"d_{name}", content="x", metadata={"filename": name}))
        return (docs, [])

    mock_parser.parse_files = AsyncMock(side_effect=record_parse_files)
    mock_get_parser.return_value = mock_parser
    one_chunk = [Document(id="c1", content="x", metadata={"chunk_index": 0})]
    mock_chunker.process = AsyncMock(return_value=one_chunk)
    mock_chunker.process_sync = MagicMock(return_value=one_chunk)
    mock_get_chunker.return_value = mock_chunker
    mock_embedder.embed_documents = MagicMock(return_value=[[0.1] * 8])
    mock_get_embedder.return_value = mock_embedder
    mock_build_retriever.return_value = mock_retriever
    mock_retriever.ingest_chunks = MagicMock(
        return_value=Resource(uri="u", title="t", description="d")
    )
    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    p1, p2, p3 = Path("/fake/a.pdf"), Path("/fake/b.pdf"), Path("/fake/c.pdf")
    asyncio.run(pipeline.run_index([p1, p2, p3]))
    assert len(parse_paths_seen) == 3
    assert parse_paths_seen[0] == "a.pdf"
    assert parse_paths_seen[1] == "b.pdf"
    assert parse_paths_seen[2] == "c.pdf"


# --- U-pipe-12: [edge case] run_index empty list ---
@patch("src.rag.pipeline.build_retriever")
def test_run_index_empty_list_returns_empty_result(mock_build_retriever):
    mock_build_retriever.return_value = None
    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": "/tmp/rag_test"}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([]))
    assert isinstance(result, IndexResult)
    assert result.successes == []
    assert result.failed == []
    assert result.trace_id


# --- U-pipe-13: [edge case] (bytes, "") filename ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
def test_run_index_bytes_empty_filename_sanitized(
    mock_get_parser, mock_get_chunker, mock_get_embedder, mock_build_retriever,
    mock_parser, mock_chunker, mock_embedder, mock_retriever, tmp_path,
):
    mock_get_parser.return_value = mock_parser
    # bytes input writes to temp; path.name becomes uuid_unnamed_file; path_to_file_id maps to unnamed_file
    def _parse_files(paths, **kw):
        name = Path(paths[0]).name
        return ([Document(id="d1", content="x", metadata={"filename": name})], [])
    mock_parser.parse_files = AsyncMock(side_effect=_parse_files)
    mock_get_chunker.return_value = mock_chunker
    one_chunk = [Document(id="c1", content="x", metadata={})]
    mock_chunker.process = AsyncMock(return_value=one_chunk)
    mock_chunker.process_sync = MagicMock(return_value=one_chunk)
    mock_get_embedder.return_value = mock_embedder
    mock_embedder.embed_documents = MagicMock(return_value=[[0.1] * 8])
    mock_build_retriever.return_value = mock_retriever
    mock_retriever.ingest_chunks = MagicMock(
        return_value=Resource(uri="u", title="t", description="d")
    )
    config = {"parser": {}, "chunker": {}, "pipeline": {"temp_dir": str(tmp_path)}}
    pipeline = RAGPipeline(config=config)
    result = asyncio.run(pipeline.run_index([(b"x", "")]))
    assert result.successes[0].file_id == "unnamed_file"
    assert len(result.successes) == 1
    for f in tmp_path.iterdir():
        assert f.parent == tmp_path
