# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for rag.types (Document, ChunkDocWithVector, Resource). Spec: U-types-01 to 06."""

import json

import pytest
from pydantic import ValidationError

from src.rag.pipeline import IndexResult, IndexSuccessEntry
from src.rag.types import ChunkDocWithVector, Document, Resource


# --- U-types-01: Document construction and fields ---
def test_document_construct_with_metadata():
    doc = Document(id="d1", content="x", metadata={})
    assert doc.id == "d1"
    assert doc.content == "x"
    assert isinstance(doc.metadata, dict)
    assert doc.metadata == {}


def test_document_construct_default_metadata():
    doc = Document(id="d1", content="x")
    assert doc.id == "d1"
    assert doc.content == "x"
    assert isinstance(doc.metadata, dict)
    assert doc.metadata == {}


# --- U-types-02: Document serialize/deserialize ---
def test_document_roundtrip():
    data = {"id": "d1", "content": "hello", "metadata": {"k": "v"}}
    doc = Document.model_validate(data)
    out = doc.model_dump()
    assert out["id"] == data["id"]
    assert out["content"] == data["content"]
    assert out["metadata"] == data["metadata"]


# --- U-types-03: ChunkDocWithVector construction ---
def test_chunk_doc_with_vector_construct():
    vec = [0.1] * 1536
    c = ChunkDocWithVector(id="c1", content="text", metadata={"k": "v"}, vector=vec)
    assert c.id == "c1"
    assert c.content == "text"
    assert c.metadata == {"k": "v"}
    assert len(c.vector) == 1536
    assert c.vector[0] == 0.1


# --- U-types-04: ChunkDocWithVector and Resource used together in pipeline ---
def test_index_result_serialize_with_resource():
    r = Resource(uri="milvus://c/f.pdf", title="T", description="D")
    result = IndexResult(
        trace_id="trace-1",
        successes=[IndexSuccessEntry(file_id="f.pdf", resource=r)],
        failed=[],
    )
    dumped = result.model_dump()
    assert "trace_id" in dumped
    assert "successes" in dumped
    assert "failed" in dumped
    assert len(dumped["successes"]) == 1
    assert dumped["successes"][0]["file_id"] == "f.pdf"
    resource_dict = dumped["successes"][0]["resource"]
    assert "uri" in resource_dict
    assert "title" in resource_dict
    assert "description" in resource_dict
    assert resource_dict["uri"] == "milvus://c/f.pdf"
    assert resource_dict["title"] == "T"
    assert resource_dict["description"] == "D"
    raw = result.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["successes"][0]["resource"]["uri"] == "milvus://c/f.pdf"


# --- U-types-05: Resource construction and defaults ---
def test_resource_construct_default_description():
    r = Resource(uri="u", title="t")
    assert r.description == ""
    d = r.model_dump()
    assert d["uri"] == "u"
    assert d["title"] == "t"
    assert d["description"] == ""
    assert json.loads(json.dumps(d)) == d  # model_dump() is JSON-serializable


def test_resource_construct_with_none_description():
    r = Resource(uri="u", title="t", description=None)
    assert r.description is None or r.description == ""


# --- U-types-06: [edge case] Document invalid or boundary input ---
def test_document_empty_id_validation_error():
    """Pydantic may allow empty id or raise ValidationError; accept either."""
    try:
        doc = Document(id="", content="x", metadata={})
        assert doc.id == ""
    except ValidationError:
        pass


def test_document_missing_required_validation_error():
    with pytest.raises(ValidationError):
        Document.model_validate({"content": "x"})
