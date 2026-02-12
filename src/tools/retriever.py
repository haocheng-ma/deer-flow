# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from typing import List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config.tools import SELECTED_RAG_PROVIDER
from src.rag import Resource, Retriever, build_retriever

logger = logging.getLogger(__name__)


class RetrieverInput(BaseModel):
    keywords: str = Field(description="search keywords to look up")


def _retriever_tool_description(resources: list) -> str:
    """Dynamic description based on whether resources are specified (narrow scope) or all (full search)."""
    if resources:
        return (
            "Useful for retrieving information from the user-specified resource files (from @ mentions). "
            "Use this tool when the user referenced specific files. Input should be search keywords."
        )
    return (
        "Useful for searching the user's local RAG knowledge base (all uploaded documents). "
        "Use this when the user mentions local documents, uploaded papers, or private knowledge base. "
        "Should be higher priority than web search when the topic relates to user's own materials. Input: search keywords."
    )


class RetrieverTool(BaseTool):
    name: str = "local_search_tool"
    description: str = "Useful for retrieving information from local knowledge base. Input: search keywords."
    args_schema: Type[BaseModel] = RetrieverInput

    retriever: Retriever = Field(..., description="RAG retriever instance (required)")
    resources: list[Resource] = Field(default_factory=list)

    def _run(
        self,
        keywords: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str | list[dict]:
        logger.info(
            f"Retriever tool query: {keywords}", extra={"resources": self.resources}
        )
        documents = self.retriever.query_relevant_documents(keywords, self.resources)
        if not documents:
            return "No results found from the local knowledge base."
        return [doc.to_dict() for doc in documents]

    async def _arun(
        self,
        keywords: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str | list[dict]:
        logger.info(
            f"Retriever tool query: {keywords}", extra={"resources": self.resources}
        )
        documents = await self.retriever.query_relevant_documents_async(
            keywords, self.resources
        )
        if not documents:
            return "No results found from the local knowledge base."
        return [doc.to_dict() for doc in documents]


def get_retriever_tool(resources: List[Resource] | None = None) -> RetrieverTool | None:
    """Create retriever tool when RAG is configured. resources can be empty for full-KB search."""
    resources = resources or []
    retriever = build_retriever()
    if not retriever:
        return None
    logger.info(
        f"create retriever tool: {SELECTED_RAG_PROVIDER}, scope={'selected resources' if resources else 'full knowledge base'}"
    )
    return RetrieverTool(
        retriever=retriever,
        resources=resources,
        description=_retriever_tool_description(resources),
    )
