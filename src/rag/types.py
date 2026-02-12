# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG pipeline types: Document (Parse/Chunk), ChunkDocWithVector (Embed out, Store in), Resource (Store out)."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document (id, content, metadata). Parse output; Chunk input/output."""

    id: str = Field(..., description="Document unique id")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

    class Config:
        arbitrary_types_allowed = True


class ChunkDocWithVector(BaseModel):
    """Chunk with embedding vector. Embed output; Store (ingest_chunks) input."""

    id: str = Field(..., description="Chunk unique id")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    vector: List[float] = Field(..., description="Embedding vector")

    class Config:
        arbitrary_types_allowed = True


class Resource(BaseModel):
    """Retrievable resource returned by Store."""

    uri: str = Field(..., description="Resource URI")
    title: str = Field(..., description="Resource title")
    description: str | None = Field("", description="Resource description")
