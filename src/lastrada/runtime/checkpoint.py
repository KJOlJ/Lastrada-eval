"""Resolve Sony weights using the existing production download behavior."""

import os
import hashlib
import tempfile
import threading
import time
from pathlib import Path
from functools import lru_cache
import streamlit as st

_SAMPLEID_MIN_BYTES = 100 * 1024 * 1024
EXPECTED_MD5 = "c50d08e54b37e49fbcf8904b1f0f5ca3"
_DOWNLOAD_LOCK = threading.Lock()
# Public provider URL, not a credential. Local previews do not inherit Cloud Run env vars.
DEFAULT_CHECKPOINT_URL = (
    "https://zenodo.org/records/17413869/files/sampleid-best.ckpt?download=1"
)


def _runtime_setting(name):
    value = os.getenv(name)
    if value:
        return str(value).strip()
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


@lru_cache(maxsize=16)
def _verified_content(path, size, modified, checksum):
    # Cache only unchanged files; size and nanosecond mtime invalidate replacements.
    import zipfile

    if not zipfile.is_zipfile(path):
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == checksum


def _valid_sampleid_checkpoint(path):
    if not path:
        return False
    try:
        candidate = Path(path).expanduser().resolve()
        stat = candidate.stat()
        if not candidate.is_file() or stat.st_size < _SAMPLEID_MIN_BYTES:
            return False
        return _verified_content(str(candidate), stat.st_size, stat.st_mtime_ns, EXPECTED_MD5)
    except OSError:
        return False


def _download_sampleid_checkpoint(url):
    import requests

    cache_root = Path(_runtime_setting("LASTRADA_CACHE_DIR") or "cache") / "sampleid"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / "sampleid-best.ckpt"
    with _DOWNLOAD_LOCK:
        if _valid_sampleid_checkpoint(destination):
            return destination.resolve()
        for attempt in range(3):
            # Never share, truncate, or unlink another request's partial file.
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".part", delete=False) as temp:
                partial = Path(temp.name)
            try:
                with requests.get(
                    url, stream=True, timeout=(15, 300),
                    headers={"User-Agent": "Lastrada-SampleID/1.0", "Accept-Encoding": "identity"},
                ) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if not _valid_sampleid_checkpoint(partial):
                    raise RuntimeError("Sony checkpoint integrity verification failed")
                partial.replace(destination)
                return destination.resolve()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
            finally:
                partial.unlink(missing_ok=True)


def resolve_sampleid_checkpoint(override=None, allow_download=False):
    candidates = []
    if override and str(override).strip():
        candidates.append(Path(str(override).strip()).expanduser())
    configured_path = _runtime_setting("LASTRADA_SAMPLEID_CHECKPOINT")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.extend(
        [
            Path("checkpoints/sampleid-best.ckpt"),
            Path.home() / ".lastrada_sampleid" / "sampleid-best.ckpt",
            Path(_runtime_setting("LASTRADA_CACHE_DIR") or "cache") / "sampleid" / "sampleid-best.ckpt",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid_sampleid_checkpoint(resolved):
            return str(resolved), None

    download_url = _runtime_setting("LASTRADA_SAMPLEID_CHECKPOINT_URL") or DEFAULT_CHECKPOINT_URL
    require_installed = _runtime_setting("LASTRADA_REQUIRE_PREINSTALLED_MODEL") == "1"
    if allow_download and download_url and not require_installed:
        try:
            downloaded = _download_sampleid_checkpoint(download_url)
            return str(downloaded), None
        except Exception as exc:
            return None, f"Sony checkpoint download failed: {exc}"

    return None, (
        "A verified Sony checkpoint is not installed. For local use, place sampleid-best.ckpt in "
        "~/.lastrada_sampleid/ or checkpoints/. For Streamlit Cloud, configure "
        "LASTRADA_SAMPLEID_CHECKPOINT_URL as a direct-download secret."
    )
