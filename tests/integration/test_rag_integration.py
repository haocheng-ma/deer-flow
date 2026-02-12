# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
RAG pipeline integration tests (Test Spec §4.2, §7 multi-file/dir E2E).

- I-pipe-01a: All mocked; IndexResult.successes has file_id and Resource; embed called; ingest_chunks vector dim matches chunk count
- I-pipe-01b: Optional real embed; skip when SKIP_REAL_EMBED or no API key
- I-pipe-02: When pipeline config is None, use load_ingestion_pipeline_config()
- I-pipe-03: Real chunk + mock rest; embed texts count matches chunk output count
- I-pipe-04: Multi-file run_index -> IndexResult.successes length matches file count
- I-pipe-05: parse_failed non-empty -> IndexResult.failed contains (file_id, "parse", reason)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.pipeline import IndexResult, RAGPipeline
from src.rag.types import Document, Resource


def _two_docs_for_file(filename: str):
    """Two documents with metadata filename for path_to_file_id grouping."""
    return [
        Document(id="d1", content="Chunk one. Some more text.", metadata={"filename": filename}),
        Document(id="d2", content="Chunk two. Extra.", metadata={"filename": filename}),
    ]


@pytest.fixture
def sample_md_path(tmp_path):
    """A minimal .md file for pipeline run_index."""
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\nParagraph one.\n\nParagraph two.", encoding="utf-8")
    return p


@pytest.fixture
def mock_parser_returning_docs():
    """Parser that returns (docs, parse_failed) for parse_files (I-pipe-01a, I-pipe-03)."""
    parser = MagicMock()
    parser.parse_files = AsyncMock(
        return_value=(
            _two_docs_for_file("sample.md"),
            [],
        )
    )
    return parser


@pytest.fixture
def mock_chunker_returning_two():
    """Chunker that returns 2 chunks (for I-pipe-01a); includes process_sync for PF-2 path."""
    two_chunks = [
        Document(id="c1", content="Chunk one.", metadata={"chunk_index": 0}),
        Document(id="c2", content="Chunk two.", metadata={"chunk_index": 1}),
    ]
    chunker = MagicMock()
    chunker.process = AsyncMock(return_value=two_chunks)
    chunker.process_sync = MagicMock(return_value=two_chunks)
    return chunker


@pytest.fixture
def mock_embedder_dim8():
    """Embedder returning 2 vectors of dim 8 (for I-pipe-01a)."""
    emb = MagicMock()
    emb.embed_documents = MagicMock(return_value=[[0.1] * 8, [0.2] * 8])
    return emb


@pytest.fixture
def mock_retriever_returns_resource():
    """Retriever that returns a Resource from ingest_chunks."""
    retriever = MagicMock()
    retriever.ingest_chunks = MagicMock(
        return_value=Resource(
            uri="milvus://documents/sample.md",
            title="sample.md",
            description="Uploaded file",
        )
    )
    return retriever


# --- I-pipe-01a: integration (all mocked) ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@pytest.mark.asyncio
async def test_integration_pipeline_all_mock_success(
    mock_get_parser,
    mock_get_chunker,
    mock_get_embedder,
    mock_build_retriever,
    mock_parser_returning_docs,
    mock_chunker_returning_two,
    mock_embedder_dim8,
    mock_retriever_returns_resource,
    sample_md_path,
):
    """
    I-pipe-01a: mock parse/chunk/embed/retriever return valid data;
    RAGPipeline().run_index([path]) -> IndexResult.successes has file_id and Resource;
    embed is called; ingest_chunks vector dim matches chunk count.
    """
    mock_get_parser.return_value = mock_parser_returning_docs
    mock_get_chunker.return_value = mock_chunker_returning_two
    mock_get_embedder.return_value = mock_embedder_dim8
    mock_build_retriever.return_value = mock_retriever_returns_resource

    config = {
        "parser": {"api_token": "x"},
        "chunker": {"type": "token"},
        "pipeline": {"temp_dir": str(sample_md_path.parent / "temp_rag")},
    }
    pipeline = RAGPipeline(config=config)
    result = await pipeline.run_index([sample_md_path], options=None)

    assert isinstance(result, IndexResult)
    assert result.trace_id
    assert len(result.successes) == 1
    entry = result.successes[0]
    assert entry.file_id == "sample.md"
    assert entry.resource.uri == "milvus://documents/sample.md"
    assert entry.resource.title == "sample.md"
    assert result.failed == []

    mock_embedder_dim8.embed_documents.assert_called_once()
    call_texts = mock_embedder_dim8.embed_documents.call_args[0][0]
    assert len(call_texts) == 2
    mock_build_retriever.return_value.ingest_chunks.assert_called_once()
    call_args = mock_retriever_returns_resource.ingest_chunks.call_args
    chunks_with_vectors = call_args[0][0]
    assert len(chunks_with_vectors) == 2
    for c in chunks_with_vectors:
        assert len(c.vector) == 8


# --- I-pipe-02: pipeline config None uses load_ingestion_pipeline_config ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@patch("src.rag.pipeline.load_ingestion_pipeline_config")
@pytest.mark.asyncio
async def test_integration_pipeline_config_none_uses_load_config(
    mock_load_config,
    mock_get_parser,
    mock_get_chunker,
    mock_get_embedder,
    mock_build_retriever,
    mock_parser_returning_docs,
    mock_chunker_returning_two,
    mock_embedder_dim8,
    mock_retriever_returns_resource,
    sample_md_path,
):
    """
    I-pipe-02: When config not passed, RAGPipeline() calls load_ingestion_pipeline_config();
    with valid config and rest mocked, run_index succeeds.
    """
    mock_load_config.return_value = {
        "parser": {"api_token": "x"},
        "chunker": {"type": "token"},
        "pipeline": {"temp_dir": str(sample_md_path.parent / "temp_rag2")},
    }
    mock_get_parser.return_value = mock_parser_returning_docs
    mock_get_chunker.return_value = mock_chunker_returning_two
    mock_get_embedder.return_value = mock_embedder_dim8
    mock_build_retriever.return_value = mock_retriever_returns_resource

    pipeline = RAGPipeline(config=None)
    assert pipeline._config == mock_load_config.return_value
    result = await pipeline.run_index([sample_md_path], options=None)

    mock_load_config.assert_called_once_with()
    assert len(result.successes) == 1
    assert result.failed == []


# --- I-pipe-03: real chunk + mock parse/embed/store; chunk count matches embed/ingest ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@pytest.mark.asyncio
async def test_integration_real_chunk_embed_and_ingest_count_match(
    mock_get_parser,
    mock_get_chunker,
    mock_get_embedder,
    mock_build_retriever,
    sample_md_path,
):
    """
    I-pipe-03: Real get_chunker + process(docs); mock parse (fixed List[Document]), embed, store;
    after run_index([path]), embed texts count and ingest_chunks chunk count match chunk output.
    """
    from src.rag.chunk import get_chunker

    parser = MagicMock()
    parser.parse_files = AsyncMock(
        return_value=(
            [
                Document(
                    id="single",
                    content="First sentence. Second sentence. Third sentence. Fourth. Fifth.",
                    metadata={"filename": "sample.md"},
                )
            ],
            [],
        )
    )
    mock_get_parser.return_value = parser

    real_chunker = get_chunker({"type": "sentence"})
    mock_get_chunker.return_value = real_chunker

    embed_calls = []

    def record_embed_documents(texts):
        embed_calls.append(len(texts))
        return [[0.1] * 8 for _ in texts]

    mock_emb = MagicMock()
    mock_emb.embed_documents = MagicMock(side_effect=record_embed_documents)
    mock_get_embedder.return_value = mock_emb

    ingest_chunk_counts = []

    def record_ingest(chunks_with_vectors, resource_metadata):
        ingest_chunk_counts.append(len(chunks_with_vectors))
        return Resource(
            uri="milvus://documents/sample.md",
            title="sample.md",
            description="Uploaded file",
        )

    retriever = MagicMock()
    retriever.ingest_chunks = MagicMock(side_effect=record_ingest)
    mock_build_retriever.return_value = retriever

    config = {
        "parser": {"api_token": "x"},
        "chunker": {"type": "sentence"},
        "pipeline": {"temp_dir": str(sample_md_path.parent / "temp_rag3")},
    }
    pipeline = RAGPipeline(config=config)
    result = await pipeline.run_index([sample_md_path], options=None)

    assert len(result.successes) == 1
    assert len(embed_calls) == 1
    num_texts = embed_calls[0]
    assert num_texts >= 1
    assert len(ingest_chunk_counts) == 1
    assert ingest_chunk_counts[0] == num_texts


# --- I-pipe-01b: optional real embed (skip when no API key or SKIP_REAL_EMBED) ---
@pytest.mark.skipif(
    os.environ.get("SKIP_REAL_EMBED", "").lower() in ("1", "true", "yes"),
    reason="SKIP_REAL_EMBED is set",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DASHSCOPE_API_KEY"),
    reason="No OPENAI_API_KEY or DASHSCOPE_API_KEY for real embed",
)
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@pytest.mark.asyncio
async def test_integration_pipeline_real_embed_mock_rest(
    mock_get_parser,
    mock_get_chunker,
    mock_build_retriever,
    mock_parser_returning_docs,
    mock_chunker_returning_two,
    mock_retriever_returns_resource,
    sample_md_path,
):
    """
    I-pipe-01b: Real get_embedder (needs OPENAI_API_KEY or DASHSCOPE_API_KEY);
    mock parse/chunk/retriever; run_index succeeds; ingest_chunks receives vectors matching chunk count.
    """
    mock_get_parser.return_value = mock_parser_returning_docs
    mock_get_chunker.return_value = mock_chunker_returning_two
    mock_build_retriever.return_value = mock_retriever_returns_resource
    os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("RAG_EMBEDDING_API_KEY"):
        os.environ["RAG_EMBEDDING_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("RAG_EMBEDDING_API_KEY"):
        os.environ["RAG_EMBEDDING_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    config = {
        "parser": {"api_token": "x"},
        "chunker": {"type": "token"},
        "pipeline": {"temp_dir": str(sample_md_path.parent / "temp_rag1b")},
    }
    pipeline = RAGPipeline(config=config)
    result = await pipeline.run_index([sample_md_path], options=None)

    assert len(result.successes) == 1
    assert result.failed == []
    call_args = mock_retriever_returns_resource.ingest_chunks.call_args
    chunks_with_vectors = call_args[0][0]
    assert len(chunks_with_vectors) == 2
    for c in chunks_with_vectors:
        assert isinstance(c.vector, list)
        assert len(c.vector) > 0


# --- I-pipe-04: multi-file run_index -> IndexResult check (Spec §7 multi-file E2E) ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@pytest.mark.asyncio
async def test_integration_pipeline_multifile_success(
    mock_get_parser,
    mock_get_chunker,
    mock_get_embedder,
    mock_build_retriever,
    mock_embedder_dim8,
    mock_chunker_returning_two,
    tmp_path,
):
    """Multi-file run_index([path1, path2]) -> successes length 2, failed empty."""
    path_a = tmp_path / "a.md"
    path_b = tmp_path / "b.md"
    path_a.write_text("# A\nContent A.", encoding="utf-8")
    path_b.write_text("# B\nContent B.", encoding="utf-8")

    async def parse_files(paths, **kwargs):
        docs = []
        for p in paths:
            name = p.name
            docs.append(
                Document(id=f"d_{name}", content=f"Content {name}.", metadata={"filename": name})
            )
        return (docs, [])

    parser = MagicMock()
    parser.parse_files = AsyncMock(side_effect=parse_files)
    mock_get_parser.return_value = parser
    mock_get_chunker.return_value = mock_chunker_returning_two
    mock_get_embedder.return_value = mock_embedder_dim8
    retriever = MagicMock()
    retriever.ingest_chunks = MagicMock(
        side_effect=lambda chunks, meta: Resource(
            uri=f"milvus://documents/{meta['filename']}",
            title=meta["filename"],
            description="Uploaded file",
        )
    )
    mock_build_retriever.return_value = retriever

    config = {
        "parser": {},
        "chunker": {"type": "token"},
        "pipeline": {"temp_dir": str(tmp_path / "temp_multi")},
    }
    pipeline = RAGPipeline(config=config)
    result = await pipeline.run_index([path_a, path_b], options=None)

    assert isinstance(result, IndexResult)
    assert len(result.successes) == 2
    assert result.failed == []
    file_ids = {result.successes[i].file_id for i in range(2)}
    assert file_ids == {"a.md", "b.md"}
    assert retriever.ingest_chunks.call_count == 2


# --- I-pipe-05: parse_failed -> failed list ---
@patch("src.rag.pipeline.build_retriever")
@patch("src.rag.pipeline.get_embedder")
@patch("src.rag.pipeline.get_chunker")
@patch("src.rag.pipeline.get_parser")
@pytest.mark.asyncio
async def test_integration_pipeline_parse_failed_in_failed(
    mock_get_parser,
    mock_get_chunker,
    mock_get_embedder,
    mock_build_retriever,
    sample_md_path,
):
    """When parse_files returns parse_failed, corresponding file_id goes to result.failed with stage=parse."""
    parser = MagicMock()
    parser.parse_files = AsyncMock(
        return_value=(
            [],
            [("sample.md", "MinerU API: state=failed")],
        )
    )
    mock_get_parser.return_value = parser
    mock_get_chunker.return_value = MagicMock()
    mock_get_embedder.return_value = MagicMock()
    mock_build_retriever.return_value = MagicMock()

    config = {
        "parser": {},
        "chunker": {},
        "pipeline": {"temp_dir": str(sample_md_path.parent / "temp_fail")},
    }
    pipeline = RAGPipeline(config=config)
    result = await pipeline.run_index([sample_md_path], options=None)

    assert len(result.successes) == 0
    assert len(result.failed) == 1
    assert result.failed[0].file_id == "sample.md"
    assert result.failed[0].stage == "parse"
    assert "state=failed" in result.failed[0].reason
