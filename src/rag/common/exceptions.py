# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG shared exceptions."""

from typing import Any, Dict, Optional


class ToolExecutionError(RuntimeError):
    """Unified tool execution error for RAG chunk/parse stages."""

    def __init__(
        self,
        message: str,
        tool_name: str = "",
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.tool_name = tool_name
        self.cause = cause
        self.details = details or {}
        super().__init__(f"[{tool_name}] {message}" if tool_name else message)
