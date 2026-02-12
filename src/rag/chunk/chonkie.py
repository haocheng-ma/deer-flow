# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Chunking via Chonkie (token/sentence/recursive). Uses rag.types.Document."""

import abc
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from chonkie.chunker.token import TokenChunker as ChonkieTokenChunker
from chonkie.chunker.sentence import SentenceChunker as ChonkieSentenceChunker
from chonkie.chunker.recursive import RecursiveChunker as ChonkieRecursiveChunker

from src.rag.common.exceptions import ToolExecutionError
from src.rag.types import Document

logger = logging.getLogger(__name__)


class _BaseChunker:
    """Base chunker interface."""
    def __init__(self, **kwargs):
        self.config = kwargs

    def get_name(self) -> str:
        return self.__class__.__name__

    async def process(self, input_data: Any, **kwargs) -> List[Document]:
        process_config = {**self.config, **kwargs}
        if isinstance(input_data, Document):
            return await self.chunk_document(input_data, **process_config)
        if isinstance(input_data, list) and all(isinstance(d, Document) for d in input_data):
            out = []
            for doc in input_data:
                out.extend(await self.chunk_document(doc, **process_config))
            return out
        if isinstance(input_data, str):
            doc = Document(id="temp_doc", content=input_data, metadata={"source": "string_input"})
            return await self.chunk_document(doc, **process_config)
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    @abc.abstractmethod
    async def chunk_document(self, document: Document, **kwargs) -> List[Document]:
        pass

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "chunk_size": {"type": "integer", "default": 2048, "description": "Max tokens per chunk"},
            "chunk_overlap": {"type": "integer", "default": 0, "description": "Overlap tokens between chunks"},
            "preserve_metadata": {"type": "boolean", "default": True, "description": "Preserve document metadata"},
        }


class ChonkieChunker(_BaseChunker):
    """Chonkie-based chunker abstraction."""
    def __init__(self, chunker_type: str = "token", **kwargs):
        super().__init__(**kwargs)
        self.chunker_type = chunker_type
        self._chunker = None

    def _initialize_chunker(self):
        raise NotImplementedError("Subclass must implement this method")

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = self._initialize_chunker()
        return self._chunker

    def chunk_document_sync(self, document: Document, **kwargs) -> List[Document]:
        """Sync chunking for asyncio.to_thread."""
        try:
            chunker = self.chunker
            chunks = chunker.chunk(document.content)
            result_docs = []
            metadata_base = dict(document.metadata) if document.metadata else {}
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document.id}_chunk_{i:04d}"
                metadata = {}
                if kwargs.get("preserve_metadata", True):
                    metadata.update(metadata_base)
                metadata.update({
                    "chunk_index": i,
                    "chunk_start": chunk.start_index,
                    "chunk_end": chunk.end_index,
                    "chunk_token_count": chunk.token_count,
                    "original_document_id": document.id,
                    "chunker_type": self.chunker_type,
                    "total_chunks": len(chunks),
                })
                result_docs.append(Document(id=chunk_id, content=chunk.text, metadata=metadata))
            logger.info("Chunked document: %s -> %d chunks", document.id, len(result_docs))
            return result_docs
        except Exception as e:
            logger.error("Chunk failed: %s - %s", document.id, e)
            raise ToolExecutionError(f"Chunk failed: {e}", tool_name=self.get_name(), cause=e) from e

    async def chunk_document(self, document: Document, **kwargs) -> List[Document]:
        """Async chunking: delegate to chunk_document_sync in thread."""
        return await asyncio.to_thread(self.chunk_document_sync, document, **kwargs)

    def process_sync(self, input_data: Any, **kwargs) -> List[Document]:
        """Sync process for pipeline asyncio.to_thread."""
        process_config = {**self.config, **kwargs}
        if isinstance(input_data, Document):
            return self.chunk_document_sync(input_data, **process_config)
        if isinstance(input_data, list) and all(isinstance(d, Document) for d in input_data):
            out = []
            for doc in input_data:
                out.extend(self.chunk_document_sync(doc, **process_config))
            return out
        if isinstance(input_data, str):
            doc = Document(id="temp_doc", content=input_data, metadata={"source": "string_input"})
            return self.chunk_document_sync(doc, **process_config)
        raise ValueError(f"Unsupported input type: {type(input_data)}")


class TokenChunker(ChonkieChunker):
    """Token-based chunker."""
    def __init__(self, **kwargs):
        super().__init__(chunker_type="token", **kwargs)

    def _initialize_chunker(self):
        tokenizer = self.config.get("tokenizer", "character")
        chunk_size = self.config.get("chunk_size", 2048)
        chunk_overlap = self.config.get("chunk_overlap", 0)
        return ChonkieTokenChunker(tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def get_config_schema(self) -> Dict[str, Any]:
        schema = super().get_config_schema()
        schema.update({
            "tokenizer": {"type": "string", "default": "character", "description": "Tokenizer type"},
            "chunk_size": {"type": "integer", "default": 2048, "description": "Max tokens per chunk"},
            "chunk_overlap": {"type": "integer", "default": 0, "description": "Overlap between chunks"},
        })
        return schema


class SentenceChunker(ChonkieChunker):
    """Sentence-based chunker."""
    def __init__(self, **kwargs):
        super().__init__(chunker_type="sentence", **kwargs)

    def _initialize_chunker(self):
        tokenizer = self.config.get("tokenizer", "character")
        chunk_size = self.config.get("chunk_size", 2048)
        chunk_overlap = self.config.get("chunk_overlap", 0)
        min_sentences_per_chunk = self.config.get("min_sentences_per_chunk", 1)
        min_characters_per_sentence = self.config.get("min_characters_per_sentence", 12)
        delim = self.config.get("delim", [". ", "! ", "? ", "\n"])
        include_delim = self.config.get("include_delim", "prev")
        return ChonkieSentenceChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=delim,
            include_delim=include_delim,
        )

    def get_config_schema(self) -> Dict[str, Any]:
        schema = super().get_config_schema()
        schema.update({
            "tokenizer": {"type": "string", "default": "character"},
            "chunk_size": {"type": "integer", "default": 2048},
            "chunk_overlap": {"type": "integer", "default": 0},
            "min_sentences_per_chunk": {"type": "integer", "default": 1},
            "min_characters_per_sentence": {"type": "integer", "default": 12},
            "delim": {"type": "array", "default": [". ", "! ", "? ", "\n"]},
            "include_delim": {"type": "string", "default": "prev", "enum": ["prev", "next"]},
        })
        return schema


class RecursiveChunker(ChonkieChunker):
    """Recursive chunker."""
    def __init__(self, **kwargs):
        super().__init__(chunker_type="recursive", **kwargs)

    def _initialize_chunker(self):
        tokenizer = self.config.get("tokenizer", "character")
        chunk_size = self.config.get("chunk_size", 2048)
        min_characters_per_chunk = self.config.get("min_characters_per_chunk", 24)
        rules = self.config.get("rules", None)
        if rules is None:
            from chonkie.types import RecursiveRules
            rules = RecursiveRules()
        return ChonkieRecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    def get_config_schema(self) -> Dict[str, Any]:
        schema = super().get_config_schema()
        schema.update({
            "tokenizer": {"type": "string", "default": "character"},
            "chunk_size": {"type": "integer", "default": 2048},
            "min_characters_per_chunk": {"type": "integer", "default": 24},
            "rules": {"type": "object", "default": None},
        })
        return schema


def create_chunker(chunker_type: str = "token", **kwargs) -> _BaseChunker:
    """Create chunker instance; chunker_type: token | sentence | recursive."""
    if chunker_type == "token":
        return TokenChunker(**kwargs)
    if chunker_type == "sentence":
        return SentenceChunker(**kwargs)
    if chunker_type == "recursive":
        return RecursiveChunker(**kwargs)
    raise ValueError(f"Unsupported chunker_type: {chunker_type}")


def get_chunker(config: dict):
    """Return chunker by config['type'] (token/sentence/recursive). Provides async process(documents) -> List[Document]."""
    cfg = dict(config)
    chunker_type = cfg.pop("type", "token")
    return create_chunker(chunker_type=chunker_type, **cfg)
