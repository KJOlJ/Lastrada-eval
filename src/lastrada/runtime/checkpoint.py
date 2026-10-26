"""Resolve Sony weights using the existing production download behavior."""

import os
from pathlib import Path
import streamlit as st

_SAMPLEID_MIN_BYTES = 100 * 1024 * 1024
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


def _valid_sampleid_checkpoint(path):
    if not path:
        return False
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.stat().st_size < _SAMPLEID_MIN_BYTES:
        return False
    try:
        import zipfile

        return zipfile.is_zipfile(candidate)
    except OSError:
        return False


def _download_sampleid_checkpoint(url):
    import requests

    cache_root = Path(_runtime_setting("LASTRADA_CACHE_DIR") or "cache") / "sampleid"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / "sampleid-best.ckpt"
    partial = destination.with_suffix(".ckpt.part")
    if _valid_sampleid_checkpoint(destination):
        return destination.resolve()

    partial.unlink(missing_ok=True)
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 300),
            headers={"User-Agent": "Lastrada-SampleID/1.0"},
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not _valid_sampleid_checkpoint(partial):
            raise RuntimeError(
                "downloaded file is incomplete or is not a valid PyTorch checkpoint"
            )
        partial.replace(destination)
        return destination.resolve()
    except Exception:
        partial.unlink(missing_ok=True)
        raise


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
    if allow_download and download_url:
        try:
            downloaded = _download_sampleid_checkpoint(download_url)
            return str(downloaded), None
        except Exception as exc:
            return None, f"Sony checkpoint download failed: {exc}"

    return None, (
        "Sony checkpoint not found. For local use, place sampleid-best.ckpt in "
        "~/.lastrada_sampleid/ or checkpoints/. For Streamlit Cloud, configure "
        "LASTRADA_SAMPLEID_CHECKPOINT_URL as a direct-download secret."
    )
