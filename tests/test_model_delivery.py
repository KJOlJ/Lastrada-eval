from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import zipfile
from pathlib import Path

import pytest

from lastrada.runtime import checkpoint


@pytest.fixture
def model_bytes(monkeypatch, tmp_path):
    content = BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("weights", "test fixture, not real weights")
    data = content.getvalue()
    monkeypatch.setattr(checkpoint, "_SAMPLEID_MIN_BYTES", 1)
    monkeypatch.setattr(checkpoint, "EXPECTED_MD5", hashlib.md5(data).hexdigest())
    monkeypatch.setattr(checkpoint, "_runtime_setting", lambda name: str(tmp_path) if name == "LASTRADA_CACHE_DIR" else None)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda seconds: None)
    checkpoint._verified_content.cache_clear()
    return data


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def test_simultaneous_users_share_one_complete_download(model_bytes, monkeypatch):
    import requests

    calls = []

    def get(*args, **kwargs):
        calls.append(True)
        return Response(model_bytes)

    monkeypatch.setattr(requests, "get", get)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(checkpoint._download_sampleid_checkpoint, ["https://test/model"] * 4))
    assert len(calls) == 1
    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == model_bytes
    assert not list(paths[0].parent.glob("*.part"))


def test_corrupt_download_is_retried_without_publishing(model_bytes, monkeypatch, tmp_path):
    import requests

    calls = []

    def get(*args, **kwargs):
        calls.append(True)
        return Response(b"incomplete")

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(RuntimeError, match="integrity"):
        checkpoint._download_sampleid_checkpoint("https://test/model")
    assert len(calls) == 3
    assert not (tmp_path / "sampleid" / "sampleid-best.ckpt").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_corrupt_first_attempt_recovers(model_bytes, monkeypatch):
    import requests

    responses = iter([b"bad", model_bytes])
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response(next(responses)))
    assert checkpoint._download_sampleid_checkpoint("https://test/model").read_bytes() == model_bytes


def test_valid_zip_with_wrong_checksum_is_rejected(model_bytes, tmp_path):
    target = tmp_path / "bad.ckpt"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("weights", "different weights")
    assert not checkpoint._valid_sampleid_checkpoint(target)


def test_production_mode_never_downloads(monkeypatch):
    monkeypatch.setattr(checkpoint, "_runtime_setting", lambda name: "1" if name == "LASTRADA_REQUIRE_PREINSTALLED_MODEL" else None)
    monkeypatch.setattr(checkpoint, "_valid_sampleid_checkpoint", lambda path: False)

    def unexpected(*args):
        pytest.fail("Production must not download during a user's analysis")

    monkeypatch.setattr(checkpoint, "_download_sampleid_checkpoint", unexpected)
    path, error = checkpoint.resolve_sampleid_checkpoint(allow_download=True)
    assert path is None
    assert "verified Sony checkpoint" in error


def test_failed_startup_check_never_launches_server(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import start_app

    def failed_check():
        raise RuntimeError("missing model")

    monkeypatch.setattr(start_app, "verify_model", failed_check)
    monkeypatch.setattr(start_app.os, "execv", lambda *args: pytest.fail("Server must not start"))
    previous = Path.cwd()
    try:
        with pytest.raises(RuntimeError, match="missing model"):
            start_app.main()
    finally:
        monkeypatch.chdir(previous)
