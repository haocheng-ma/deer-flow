# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for rag.parse (get_parser, MinerUParser). Spec: U-parse-01 to 05."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.parse import get_parser
from src.rag.parse.mineru import MinerUParser
from src.rag.types import Document


# --- U-parse-01: get_parser(config) return type ---
def test_get_parser_returns_object_with_async_parse_files():
    config = {"api_token": "x"}
    parser = get_parser(config)
    assert parser is not None
    assert hasattr(parser, "parse_files")
    assert asyncio.iscoroutinefunction(parser.parse_files)


# --- U-parse-02: MinerUParser can be instantiated ---
def test_mineru_parser_instantiate():
    parser = MinerUParser(api_token="test")
    assert hasattr(parser, "parse_files")
    assert asyncio.iscoroutinefunction(parser.parse_files)


# --- U-parse-03: parse_files returns (List[Document], parse_failed) ---
@pytest.mark.asyncio
async def test_parse_files_returns_list_of_documents(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n\nWorld.")
    parser = MinerUParser(api_token="mock")
    with patch.object(parser._parser, "parse_files", new_callable=AsyncMock) as m:
        m.return_value = (
            [
                Document(id="1", content="Hello World.", metadata={"source": "test.md"}),
            ],
            [],
        )
        docs, failed = await parser.parse_files([md_file])
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert failed == []
    doc = docs[0]
    assert hasattr(doc, "id")
    assert hasattr(doc, "content")
    assert hasattr(doc, "metadata")
    assert doc.id == "1"
    assert doc.content == "Hello World."


# --- U-parse-04: parse failure distinguishable in pipeline (see test_pipeline parse failure) ---
# Covered by test_run_index_parse_empty_then_failed: stage=="parse", reason non-empty.


# --- U-parse-05: [edge case] path is directory or missing ---
@pytest.mark.asyncio
async def test_parse_files_nonexistent_path_raises(tmp_path):
    parser = MinerUParser(api_token="mock")
    nonexistent = tmp_path / "nonexistent.pdf"
    assert not nonexistent.exists()
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        await parser.parse_files([nonexistent])


@pytest.mark.asyncio
async def test_parse_files_directory_behavior(tmp_path):
    """When path is a directory: implementation may proceed after exists() and return empty or one failed entry."""
    parser = MinerUParser(api_token="mock")
    with patch.object(parser._parser, "parse_files", new_callable=AsyncMock) as mock_pf:
        mock_pf.return_value = ([], [])
        docs, failed = await parser.parse_files([tmp_path])
        assert docs == []
        # implementation may return empty or one failed entry for directory input
        assert failed == [] or len(failed) == 1
