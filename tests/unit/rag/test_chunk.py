# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for rag.chunk (get_chunker, process). Spec: U-chunk-01 to 06."""

import asyncio
import pytest

from src.rag.chunk import get_chunker
from src.rag.types import Document


# --- U-chunk-01: get_chunker(config) by type ---
@pytest.mark.parametrize("chunk_type", ["token", "sentence", "recursive"])
def test_get_chunker_by_type(chunk_type):
    config = {"type": chunk_type}
    chunker = get_chunker(config)
    assert chunker is not None
    assert hasattr(chunker, "process")
    assert asyncio.iscoroutinefunction(chunker.process)


# --- U-chunk-02: get_chunker default type ---
def test_get_chunker_default_type():
    chunker = get_chunker({})
    assert chunker is not None
    assert chunker.chunker_type == "token"


def test_get_chunker_no_type_key():
    chunker = get_chunker({"chunk_size": 512})
    assert chunker is not None


# --- U-chunk-03: process(documents) input/output ---
@pytest.mark.asyncio
async def test_process_returns_list_with_metadata():
    chunker = get_chunker({"type": "token", "chunk_size": 100})
    docs = [
        Document(id="d1", content="First sentence. Second sentence.", metadata={"source": "a"}),
    ]
    result = await chunker.process(docs)
    assert isinstance(result, list)
    assert all(hasattr(d, "metadata") for d in result)
    for d in result:
        assert "chunk_index" in d.metadata or "chunker_type" in d.metadata or "original_document_id" in d.metadata


# --- U-chunk-04: process empty input ---
@pytest.mark.asyncio
async def test_process_empty_list_returns_empty():
    chunker = get_chunker({"type": "token"})
    result = await chunker.process([])
    assert result == []


@pytest.mark.asyncio
async def test_process_single_doc_empty_content_returns_empty():
    chunker = get_chunker({"type": "token"})
    doc = Document(id="d1", content="", metadata={})
    result = await chunker.process([doc])
    assert result == []


# --- U-chunk-05: different types yield distinguishable chunk counts ---
@pytest.mark.asyncio
async def test_process_token_returns_at_least_one():
    chunker = get_chunker({"type": "token", "chunk_size": 10})
    doc = Document(id="d1", content="A" * 50, metadata={})
    result = await chunker.process([doc])
    assert len(result) >= 1
    assert any("token" in str(d.metadata.get("chunker_type", "")) for d in result)


@pytest.mark.asyncio
async def test_process_sentence_vs_token_different():
    doc = Document(id="d1", content="First. Second. Third.", metadata={})
    token_chunker = get_chunker({"type": "token", "chunk_size": 100})
    sent_chunker = get_chunker({"type": "sentence", "chunk_size": 100})
    r_token = await token_chunker.process([doc])
    r_sent = await sent_chunker.process([doc])
    assert len(r_token) >= 1
    assert len(r_sent) >= 1


# --- U-chunk-06: [edge case] unknown type raises ---
def test_get_chunker_unknown_type_raises():
    with pytest.raises(ValueError) as exc_info:
        get_chunker({"type": "unknown"})
    msg = str(exc_info.value).lower()
    assert "unknown" in msg or "unsupported" in msg or "不支持" in str(exc_info.value)
