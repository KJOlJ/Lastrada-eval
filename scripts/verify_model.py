"""Offline deployment gate: verify weights and run one generated-audio inference."""

import os
from pathlib import Path
import tempfile

from lastrada.audio.synthetic import demo_pair
from lastrada.recognition.sampleid_adapter import SampleIDAdapter
from lastrada.runtime.checkpoint import _valid_sampleid_checkpoint


def verify_model():
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(os.environ.get(
        "LASTRADA_SAMPLEID_CHECKPOINT", root / "checkpoints" / "sampleid-best.ckpt"
    )).resolve()
    if not _valid_sampleid_checkpoint(checkpoint):
        raise RuntimeError("Bundled Sony model is missing or fails the pinned checksum")
    with tempfile.TemporaryDirectory(prefix="lastrada-model-check-") as cache:
        adapter = SampleIDAdapter(checkpoint, cache_dir=cache, batch_size=1)
        embeddings = adapter.embed(demo_pair()[0])
        if embeddings.values.shape[0] != 1 or not embeddings.values.size:
            raise RuntimeError("Sony synthetic inference did not return an embedding")
    print("Bundled Sony checkpoint checksum and offline inference: OK", flush=True)
    return checkpoint


if __name__ == "__main__":
    verify_model()
