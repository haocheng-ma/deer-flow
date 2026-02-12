# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Parse stage: get_parser factory; currently MinerU only."""

from src.rag.parse.mineru import MinerUParser


def get_parser(config: dict):
    """Return parser (MinerU). Must provide parse(path) and parse_files(paths) -> (documents, parse_failed)."""
    return MinerUParser(**config)
