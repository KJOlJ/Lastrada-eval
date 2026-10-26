from lastrada.runtime import checkpoint


def test_local_preview_downloads_public_model_without_cloud_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_runtime_setting", lambda name: None)
    monkeypatch.setattr(checkpoint, "_valid_sampleid_checkpoint", lambda path: False)
    urls = []

    def download(url):
        urls.append(url)
        return tmp_path / "model.ckpt"

    monkeypatch.setattr(checkpoint, "_download_sampleid_checkpoint", download)
    path, error = checkpoint.resolve_sampleid_checkpoint(allow_download=True)
    assert error is None
    assert path == str(tmp_path / "model.ckpt")
    assert urls == [checkpoint.DEFAULT_CHECKPOINT_URL]


def test_configured_url_takes_precedence_and_download_remains_opt_in(monkeypatch, tmp_path):
    url = "https://example.test/custom.ckpt"
    monkeypatch.setattr(checkpoint, "_runtime_setting", lambda name: url if name.endswith("_URL") else None)
    monkeypatch.setattr(checkpoint, "_valid_sampleid_checkpoint", lambda path: False)
    urls = []

    def download(value):
        urls.append(value)
        return tmp_path / "model.ckpt"

    monkeypatch.setattr(checkpoint, "_download_sampleid_checkpoint", download)
    assert checkpoint.resolve_sampleid_checkpoint(allow_download=False)[0] is None
    assert not urls
    checkpoint.resolve_sampleid_checkpoint(allow_download=True)
    assert urls == [url]


def test_cached_checkpoint_is_resolved_without_downloading(monkeypatch, tmp_path):
    cached = tmp_path / "sampleid" / "sampleid-best.ckpt"
    monkeypatch.setattr(checkpoint, "_runtime_setting", lambda name: str(tmp_path) if name == "LASTRADA_CACHE_DIR" else None)
    monkeypatch.setattr(checkpoint, "_valid_sampleid_checkpoint", lambda path: path == cached)
    assert checkpoint.resolve_sampleid_checkpoint() == (str(cached), None)
