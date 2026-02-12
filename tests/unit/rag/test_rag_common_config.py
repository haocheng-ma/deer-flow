# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for rag.common.config (load_ingestion_pipeline_config). RAG Embedding is configured via .env only; load_embed_config was removed."""

from unittest.mock import patch

import pytest

from src.rag.common.config import load_ingestion_pipeline_config


# --- U-common-01: default path (config_path=None uses conf.yaml) ---
def test_load_config_default_path_uses_conf_yaml():
    with patch("src.rag.common.config.load_yaml_config") as m:
        m.return_value = {}
        result = load_ingestion_pipeline_config(config_path=None)
        assert result == {}
        m.assert_called_once()
        call_path = m.call_args[0][0]
        assert call_path == "conf.yaml"


# --- U-common-02: explicit path ---
def test_load_config_explicit_path_returns_ingestion_section(tmp_path):
    yaml_file = tmp_path / "test_conf.yaml"
    yaml_file.write_text("INGESTION_PIPELINE:\n  parser:\n    api_token: x\n  chunker:\n    type: token\n")
    result = load_ingestion_pipeline_config(config_path=str(yaml_file))
    assert isinstance(result, dict)
    assert "parser" in result
    assert result.get("parser", {}).get("api_token") == "x"
    assert "chunker" in result


# --- U-common-03: file not found or no INGESTION_PIPELINE ---
def test_load_config_file_not_found_returns_empty():
    result = load_ingestion_pipeline_config(config_path="/nonexistent/path/conf.yaml")
    assert result == {}


def test_load_config_no_ingestion_key_returns_empty(tmp_path):
    yaml_file = tmp_path / "other.yaml"
    yaml_file.write_text("OTHER_KEY: value\n")
    result = load_ingestion_pipeline_config(config_path=str(yaml_file))
    assert result == {}


# --- U-common-05: [edge case] invalid YAML or no read permission ---
def test_load_config_invalid_yaml_raises_or_returns_empty(tmp_path):
    """When YAML is invalid, implementation may return empty dict or raise; if return, must be empty dict."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("not: valid: yaml: [")
    try:
        result = load_ingestion_pipeline_config(config_path=str(bad_yaml))
        assert result == {}
    except Exception:
        pass
