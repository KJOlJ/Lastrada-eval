# Sony model delivery

Production users only upload two recordings and select Analyze. Model installation
is a deployment responsibility, not a user action.

`cloudbuild.json` downloads Sony's public pinned checkpoint into `checkpoints/`
before building the image. `scripts/fetch_sampleid.py` checks provider metadata,
each byte range and the complete pinned checksum. Failures stop the build. Model
weights, recordings and caches are not committed to Git.

The image is tested using Docker with `--network=none`. `scripts/verify_model.py`
requires the bundled checkpoint to match the pinned checksum and performs an
actual inference on generated audio. Import-only validation is insufficient.

The entrypoint is `python scripts/start_app.py`. Startup repeats the checksum and
generated-audio inference before opening a listening port, then execs Streamlit.
It sets the exact checkpoint path and disables request-time downloads. An instance
without a working model fails its startup probe instead of showing unusable
analysis results to users. No input recordings are involved in readiness checks.

The local-development fallback uses a process-wide download lock, unique partial
files, bounded retries, checksum validation, and atomic publication. Concurrent
processes may download independently but never share a partial file. Production
does not use that fallback.

Configure the existing Cloud Build trigger to read `cloudbuild.json` from the
deployment branch, retaining its GitHub connection, deployment substitutions,
Python 3.12/pip settings, service identity, and Cloud Run session settings. The
legacy inline import-only build is not sufficient for this delivery contract.

Rollback: restore the previous Cloud Run revision and trigger configuration from
the saved deployment snapshot. A rolled-back older revision can still have the
old request-time download defect, so prefer a new corrected build.
