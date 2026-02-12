# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""RAG common utilities (e.g. filename sanitization for upload/pipeline)."""

import os


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal. Single implementation for pipeline and server."""
    basename = os.path.basename(filename)
    sanitized = basename.replace("\x00", "").strip()
    if not sanitized or sanitized in (".", ".."):
        return "unnamed_file"
    return sanitized
