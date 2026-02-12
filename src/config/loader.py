# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_str_env(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return default if val is None else str(val).strip()


def get_int_env(name: str, default: int = 0) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        logger.warning(
            "Invalid integer value for %s: %r, using default %s.", name, val, default
        )
        return default


def replace_env_vars(value: str) -> str:
    """Replace environment variables in string values."""
    if not isinstance(value, str):
        return value
    if value.startswith("$"):
        env_var = value[1:]
        return os.getenv(env_var, env_var)
    return value


def process_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process dictionary to replace environment variables."""
    if not config:
        return {}
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = process_dict(value)
        elif isinstance(value, str):
            result[key] = replace_env_vars(value)
        else:
            result[key] = value
    return result


_config_cache: Dict[str, Dict[str, Any]] = {}


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load and process YAML configuration file (UTF-8).

    Returns empty dict if file is missing, empty, or malformed (after logging).
    """
    if not os.path.exists(file_path):
        return {}

    if file_path in _config_cache:
        return _config_cache[file_path]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning("YAML parse error in %s: %s", file_path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", file_path, e)
        return {}

    processed_config = process_dict(config) if config is not None else {}
    _config_cache[file_path] = processed_config
    return processed_config
