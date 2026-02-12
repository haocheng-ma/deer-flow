# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MinerU parser: batch parse via MinerU API; uses rag.types.Document."""

import asyncio
import hashlib
import io
import json
import logging
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from src.rag.common.exceptions import ToolExecutionError
from src.rag.types import Document

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Thread-local requests Session for use inside asyncio.to_thread."""
    if not hasattr(_thread_local, "session") or _thread_local.session is None:
        _thread_local.session = requests.Session()
    return _thread_local.session


class _BaseParser:
    def __init__(self, **kwargs):
        self.config = kwargs
    def get_name(self) -> str:
        return "DocumentParser"
    def get_config_schema(self) -> Dict[str, Any]:
        return {}


class DocumentParser(_BaseParser):
    """Parse documents via MinerU API (batch). Override api_base_url in config for another endpoint."""

    def __init__(self, api_token: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        # api_base_url configurable for different MinerU environments
        self.api_base_url = kwargs.get("api_base_url") or "https://mineru.net/api/v4"
        self.api_token = api_token or kwargs.get("api_token")

        self.default_config = {
            "api_base_url": self.api_base_url,
            "api_token": self.api_token,
            "model_version": "pipeline",
            "enable_ocr": False,
            "language": "ch",
            "enable_formula": True,
            "enable_table": True,
            "timeout": 300,
            "max_file_size": 200 * 1024 * 1024,
            "poll_interval": 5,
            "max_poll_time": 600,
            "batch_size": 200,
            "callback": None,
            "seed": None,
            "extra_formats": [],
            "page_ranges": "",
            "temp_dir": None,
        }

        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.config.update(kwargs)

        if not self.config["api_token"]:
            logger.warning("MinerU API token not set; some features may be limited")
    

    async def parse_files(
        self, file_paths: List[Path], **kwargs
    ) -> Tuple[List[Document], List[Tuple[str, str]]]:
        """Batch parse files. Returns (documents, parse_failed list of (path_name, reason))."""
        if not file_paths:
            return [], []

        parse_config = {**self.config, **kwargs}
        batch_size = parse_config.get("batch_size", 200)
        if len(file_paths) > batch_size:
            raise ValueError(f"File count exceeds batch limit: {len(file_paths)} > {batch_size}")

        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            file_size = file_path.stat().st_size
            if file_size > parse_config["max_file_size"]:
                raise ValueError(
                    f"File size exceeds limit: {file_size} > {parse_config['max_file_size']}"
                )

        try:
            batch_id, file_urls = await self._request_batch_upload_urls(
                file_paths, parse_config
            )
            success_count = await self._upload_files(
                file_paths, file_urls, parse_config
            )
            if success_count == 0:
                raise ToolExecutionError(
                    message="All file uploads to parse service failed; check network or retry",
                    tool_name=self.get_name(),
                    details={"batch_id": batch_id, "file_count": len(file_paths)},
                )
            extract_results = await self._poll_batch_results(
                batch_id, parse_config, expected_success_count=success_count
            )
            documents, parse_failed = await self._process_extract_results(
                file_paths, extract_results, parse_config
            )
            return (documents, parse_failed)
        except Exception as e:
            logger.error("Batch parse failed: %s", e)
            raise ToolExecutionError(
                message=f"Batch parse failed: {str(e)}",
                tool_name=self.get_name(),
                cause=e,
                details={"input_data": [str(p) for p in file_paths]},
            ) from e

    async def _request_batch_upload_urls(self, file_paths: List[Path], config: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Request batch upload URLs from MinerU API; returns (batch_id, file_urls)."""
        url = f"{self.api_base_url}/file-urls/batch"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_token']}"
        }
        files_data = []
        for file_path in file_paths:
            files_data.append({
                "name": file_path.name,
                "data_id": self._generate_data_id(file_path),
                "is_ocr": config.get("enable_ocr", False),
                "page_ranges": config.get("page_ranges", "")
            })
        
        data = {
            "files": files_data,
            "model_version": config.get("model_version", "pipeline"),
            "enable_formula": config.get("enable_formula", True),
            "enable_table": config.get("enable_table", True),
            "language": config.get("language", "ch")
        }
        if config.get("callback"):
            data["callback"] = config["callback"]
        if config.get("seed"):
            data["seed"] = config["seed"]
        if config.get("extra_formats"):
            data["extra_formats"] = config["extra_formats"]
        
        try:
            response = await asyncio.to_thread(
                _get_session().post,
                url,
                headers=headers,
                json=data,
                timeout=config.get("timeout", 300),
            )

            if response.status_code != 200:
                raise ToolExecutionError(
                    message=f"Upload URL request failed: HTTP {response.status_code}",
                    tool_name=self.get_name(),
                    details={"response": response.text}
                )

            result = response.json()
            if result.get("code") != 0:
                raise ToolExecutionError(
                    message=f"Upload URL request failed: {result.get('msg', 'Unknown error')}",
                    tool_name=self.get_name(),
                    details={"result": result}
                )

            batch_id = result.get("data", {}).get("batch_id")
            file_urls = result.get("data", {}).get("file_urls", [])

            if not batch_id or not file_urls:
                raise ToolExecutionError(
                    message="Invalid batch_id or file_urls in response",
                    tool_name=self.get_name(),
                    details={"result": result}
                )

            if len(file_urls) != len(file_paths):
                raise ToolExecutionError(
                    message=f"URL count ({len(file_urls)}) does not match file count ({len(file_paths)})",
                    tool_name=self.get_name(),
                    details={"result": result}
                )

            logger.info("Batch upload URLs obtained batch_id=%s files=%d", batch_id, len(file_urls))
            return batch_id, file_urls

        except requests.exceptions.RequestException as e:
            raise ToolExecutionError(
                message=f"Request failed: {str(e)}",
                tool_name=self.get_name(),
                cause=e,
            ) from e

    async def _upload_files(
        self, file_paths: List[Path], file_urls: List[str], config: Dict[str, Any]
    ) -> int:
        """Upload files to OSS; return success count for poll early-exit."""
        upload_tasks = []
        for i, (file_path, upload_url) in enumerate(zip(file_paths, file_urls)):
            task = self._upload_single_file(file_path, upload_url, config)
            upload_tasks.append(task)

        results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Upload failed [%s/%s]: %s - %s",
                    i + 1,
                    len(file_paths),
                    file_paths[i].name,
                    result,
                )
            elif result is False:
                logger.error(
                    "Upload failed [%s/%s]: %s",
                    i + 1,
                    len(file_paths),
                    file_paths[i].name,
                )
            else:
                success_count += 1
        return success_count
    
    async def _upload_single_file(
        self, file_path: Path, upload_url: str, config: Dict[str, Any]
    ) -> bool:
        """Upload a single file to the given OSS URL."""

        def _put():
            with open(file_path, "rb") as f:
                return _get_session().put(
                    upload_url, data=f, timeout=config.get("timeout", 300)
                )

        try:
            response = await asyncio.to_thread(_put)
            if response.status_code != 200:
                logger.error(
                    "Upload failed: HTTP %s, URL: %s",
                    response.status_code,
                    upload_url,
                )
                return False
            logger.info("Upload success: %s", file_path.name)
            return True
        except Exception as e:
            logger.error("Upload error: %s - %s", file_path.name, e)
            return False
    
    async def _poll_batch_results(
        self,
        batch_id: str,
        config: Dict[str, Any],
        expected_success_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Poll batch results; stop when done_count + failed_count >= expected_success_count."""
        url = f"{self.api_base_url}/extract-results/batch/{batch_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_token']}",
        }
        max_poll_time = config.get("max_poll_time", 600)
        poll_interval = config.get("poll_interval", 5)
        start_time = time.time()
        logger.info("Polling batch results batch_id=%s expected=%s", batch_id, expected_success_count)

        while time.time() - start_time < max_poll_time:
            try:
                response = await asyncio.to_thread(
                    _get_session().get,
                    url,
                    headers=headers,
                    timeout=config.get("timeout", 300),
                )
                if response.status_code != 200:
                    logger.warning("Poll result failed: HTTP %s", response.status_code)
                    await asyncio.sleep(poll_interval)
                    continue

                result = response.json()
                if result.get("code") != 0:
                    logger.warning(
                        "Poll result failed: %s",
                        result.get("msg", "Unknown error"),
                    )
                    await asyncio.sleep(poll_interval)
                    continue

                extract_results = result.get("data", {}).get("extract_result", [])
                if not extract_results:
                    logger.warning("No extract results")
                    await asyncio.sleep(poll_interval)
                    continue

                done_count = sum(1 for item in extract_results if item.get("state") == "done")
                failed_count = sum(1 for item in extract_results if item.get("state") == "failed")
                total_count = len(extract_results)
                all_done = all(
                    item.get("state", "") in ("done", "failed") for item in extract_results
                )
                if (
                    expected_success_count is not None
                    and expected_success_count > 0
                    and done_count + failed_count >= expected_success_count
                ):
                    logger.info(
                        "Poll done batch_id=%s final_count=%s expected=%s",
                        batch_id,
                        done_count + failed_count,
                        expected_success_count,
                    )
                    return extract_results
                if all_done:
                    logger.info("Batch parse done batch_id=%s", batch_id)
                    return extract_results

                logger.info(
                    "Parse progress: %s/%s done, %s failed",
                    done_count,
                    total_count,
                    failed_count,
                )
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.warning("Poll error: %s", e)
                await asyncio.sleep(poll_interval)

        raise ToolExecutionError(
            message=f"Parse timeout: not completed within {max_poll_time}s",
            tool_name=self.get_name(),
            details={"batch_id": batch_id, "max_poll_time": max_poll_time},
        )

    async def _process_extract_results(
        self,
        file_paths: List[Path],
        extract_results: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Tuple[List[Document], List[Tuple[str, str]]]:
        """Process extract results: download and extract Markdown; return (documents, parse_failed).
        Match by result index (extract_results[i] -> file_paths[i]) first, then fallback to file_name.
        """
        documents: List[Document] = []
        parse_failed: List[Tuple[str, str]] = []
        file_map = {file_path.name: file_path for file_path in file_paths}

        for i, result in enumerate(extract_results):
            file_name = result.get("file_name", "")
            state = result.get("state", "")
            err_msg = result.get("err_msg", "")
            full_zip_url = result.get("full_zip_url", "")
            data_id = result.get("data_id", "")

            file_path = file_paths[i] if i < len(file_paths) else None
            if file_path is None:
                file_path = file_map.get(file_name)
            if not file_path:
                logger.warning("File not in batch: %s", file_name)
                parse_failed.append((file_name, "file_name not in batch"))
                continue

            path_name = file_path.name
            if state == "failed":
                reason = err_msg or "state=failed"
                logger.error("Parse failed: %s - %s", path_name, reason)
                parse_failed.append((path_name, reason))
                continue

            if state != "done":
                logger.warning("File state not done: %s - %s", path_name, state)
                parse_failed.append((path_name, f"state={state}"))
                continue

            if not full_zip_url:
                logger.error("No result URL: %s", path_name)
                parse_failed.append((path_name, "no full_zip_url"))
                continue

            try:
                process_result = await self._download_and_process_zip(
                    full_zip_url, config, file_path.stem
                )
                markdown_content = process_result["content"]
                saved_path = process_result["saved_path"]
                assets = process_result["assets"]
                extracted_files = process_result["extracted_files"]

                logger.info("Parse result saved to temp dir: %s", saved_path)
                document_id = self._generate_document_id(file_path, markdown_content)
                # Only include parse-semantic config in metadata (whitelist). Avoid dumping
                # full config (api_base_url, timeout, batch_size, etc.) to reduce storage and
                # avoid leaking operational/sensitive details into chunks/vector store.
                _parse_meta = {
                    k: config[k]
                    for k in ("model_version", "language", "enable_ocr", "enable_formula", "enable_table")
                    if k in config
                }
                metadata = {
                    "source": str(file_path),
                    "filename": path_name,
                    "file_size": file_path.stat().st_size,
                    "file_type": file_path.suffix.lower(),
                    "parsed_at": time.time(),
                    "parser": "MinerU",
                    "batch_index": i,
                    "total_in_batch": len(extract_results),
                    "data_id": data_id,
                    "state": state,
                    "assets_dir": saved_path,
                    "assets": assets,
                    "assets_count": len(assets),
                    "extracted_files_count": len(extracted_files),
                    **_parse_meta,
                }
                document = Document(
                    id=document_id,
                    content=markdown_content,
                    metadata=metadata,
                )
                documents.append(document)
                logger.info(
                    "Document parsed [%d/%d]: %s, assets: %d",
                    i + 1,
                    len(extract_results),
                    path_name,
                    len(assets),
                )
            except Exception as e:
                logger.error("Process result failed: %s - %s", path_name, e)
                parse_failed.append((path_name, str(e)))

        return (documents, parse_failed)
    
    async def _download_and_process_zip(self, zip_url: str, config: Dict[str, Any],
                                       file_name: Optional[str] = None) -> Dict[str, Any]:
        """Download MinerU result ZIP and extract Markdown and assets. Returns dict: content, saved_path, assets, extracted_files."""
        try:
            response = await asyncio.to_thread(
                _get_session().get, zip_url, timeout=config.get("timeout", 300)
            )
            if response.status_code != 200:
                raise ToolExecutionError(
                    message=f"Download result failed: HTTP {response.status_code}",
                    tool_name=self.get_name(),
                    details={"zip_url": zip_url}
                )

            temp_dir = config.get("temp_dir")
            if temp_dir:
                base_temp_dir = Path(temp_dir)
                base_temp_dir.mkdir(parents=True, exist_ok=True)
                save_dir = Path(tempfile.mkdtemp(dir=str(base_temp_dir)))
                logger.info("Using configured temp dir: %s", save_dir)
            else:
                save_dir = Path(tempfile.mkdtemp())
                logger.info("Using system temp dir: %s", save_dir)

            if file_name is None:
                file_name = f"result_{int(time.time())}"

            zip_data = io.BytesIO(response.content)
            markdown_content = ""
            extracted_files = []
            assets = []

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.endswith('/'):
                        continue

                    extract_path = save_dir / zip_info.filename
                    extract_path.parent.mkdir(parents=True, exist_ok=True)

                    with zip_ref.open(zip_info) as source, open(extract_path, 'wb') as target:
                        target.write(source.read())

                    extracted_files.append(zip_info.filename)

                    file_ext = Path(zip_info.filename).suffix.lower()
                    if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.tiff', '.ico']:
                        assets.append(zip_info.filename)

                    if zip_info.filename.endswith('.md') and not markdown_content:
                        with open(extract_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()

            logger.info("ZIP extracted files=%d assets=%d", len(extracted_files), len(assets))

            if not markdown_content:
                raise ToolExecutionError(
                    message="No Markdown file in ZIP",
                    tool_name=self.get_name(),
                    details={"zip_url": zip_url, "extracted_files": extracted_files}
                )

            logger.info("Markdown content length: %d", len(markdown_content))
            
            return {
                "content": markdown_content,
                "saved_path": str(save_dir),
                "assets": assets,
                "extracted_files": extracted_files
            }
            
        except zipfile.BadZipFile as e:
            raise ToolExecutionError(
                message="Downloaded file is not a valid ZIP",
                tool_name=self.get_name(),
                cause=e,
                details={"zip_url": zip_url},
            ) from e
        except Exception as e:
            raise ToolExecutionError(
                message=f"Download or process ZIP failed: {str(e)}",
                tool_name=self.get_name(),
                cause=e,
                details={"zip_url": zip_url},
            ) from e

    def _generate_data_id(self, file_path: Path) -> str:
        """Stable data id from file name, size, mtime and time."""
        file_stat = file_path.stat()
        content = f"{file_path.name}_{file_stat.st_size}_{file_stat.st_mtime}_{time.time()}"
        data_id = hashlib.md5(content.encode()).hexdigest()[:32]
        
        return f"deerflow_{data_id}"
    
    def _generate_document_id(self, file_path: Path, content: str) -> str:
        """Stable document id from path and content hash."""
        file_stat = file_path.stat()
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        
        return f"doc_{file_path.stem}_{file_stat.st_size}_{file_stat.st_mtime}_{content_hash}"
    
    def get_config_schema(self) -> Dict[str, Any]:
        schema = super().get_config_schema()
        schema.update({
            "api_base_url": {"type": "string", "default": "https://mineru.net/api/v4", "description": "MinerU API base URL (override for different environments)"},
            "api_token": {"type": "string", "default": None, "description": "MinerU API token"},
            "model_version": {"type": "string", "default": "pipeline", "description": "Model version", "enum": ["pipeline", "vlm", "MinerU-HTML"]},
            "enable_ocr": {"type": "boolean", "default": False, "description": "Enable OCR"},
            "enable_formula": {"type": "boolean", "default": True, "description": "Enable formula recognition"},
            "enable_table": {"type": "boolean", "default": True, "description": "Enable table recognition"},
            "language": {"type": "string", "default": "ch", "description": "Document language (ch/en)"},
            "timeout": {"type": "integer", "default": 300, "description": "API timeout (seconds)"},
            "max_file_size": {"type": "integer", "default": 200 * 1024 * 1024, "description": "Max file size (bytes)"},
            "poll_interval": {"type": "integer", "default": 5, "description": "Poll interval (seconds)"},
            "max_poll_time": {"type": "integer", "default": 600, "description": "Max poll time (seconds)"},
            "batch_size": {"type": "integer", "default": 200, "description": "Max files per batch"},
            "callback": {"type": "string", "default": None, "description": "Callback URL"},
            "seed": {"type": "string", "default": None, "description": "Callback signature seed"},
            "extra_formats": {"type": "array", "default": [], "description": "Extra export formats"},
            "page_ranges": {"type": "string", "default": "", "description": "Page ranges e.g. 1-10,15-20"},
            "temp_dir": {"type": "string", "default": None, "description": "Temp directory path"},
        })
        return schema

    def __repr__(self) -> str:
        has_token = bool(self.config.get("api_token"))
        token_info = "with token" if has_token else "no token"
        return f"DocumentParser(model={self.config.get('model_version', 'pipeline')}, {token_info})"



class MinerUParser:
    """Wrapper used by get_parser; exposes parse_files for pipeline batch flow."""

    def __init__(self, **kwargs):
        self._parser = DocumentParser(**kwargs)

    async def parse_files(
        self, paths: List[Path], **kwargs
    ) -> Tuple[List[Document], List[Tuple[str, str]]]:
        """Delegate to DocumentParser.parse_files; used by pipeline."""
        return await self._parser.parse_files(paths, **kwargs)
