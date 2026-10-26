# Sony SampleID + guarded DTW in the main Lastrada workflow

The normal `app.py` flow now treats Sony SampleID as the default master-timing
candidate generator. The separate **DTW Lab** Streamlit page is removed from the
deployed product navigation.

## Analysis sequence

1. PF-NMF remains available as broad spectral-overlap context.
2. Sony SampleID retrieves paired original-track and new-track candidate regions
   using top-ranked matches plus temporal consistency.
3. Direct waveform alignment is attempted first for near-identical reuse.
4. Guarded chroma/subsequence DTW is attempted for moderate pitch or timing
   transformations.
5. A DTW proposal is accepted only when it overlaps the Sony candidate and its
   derivative/source duration ratio stays within the configured guardrails.
6. The resulting paired intervals appear directly in **Step 3 · Listen & check**.
7. The reviewer accepts, edits, or rejects the candidate timings before they are
   used in the Nature of Use document.

All automated results remain review-required candidate evidence.

## Local checkpoint setup

Place the checkpoint at either:

```text
checkpoints/sampleid-best.ckpt
```

or:

```text
~/.lastrada_sampleid/sampleid-best.ckpt
```

The optional environment variable `LASTRADA_SAMPLEID_CHECKPOINT` may point to a
custom local path.

## Streamlit Community Cloud

A Windows path such as `C:\Users\...` cannot work on the Linux cloud server.
Configure a Streamlit secret containing a stable direct-download URL instead:

```toml
LASTRADA_SAMPLEID_CHECKPOINT_URL = "https://your-storage.example/sampleid-best.ckpt"
```

The app downloads the checkpoint atomically into its runtime cache and validates
that it is large enough and is a PyTorch ZIP checkpoint before loading it.

The production requirements use `-e git+https://...#egg=sampleid` to install Sony
SampleID from the pinned upstream source in editable mode. A plain
`sampleid @ git+https://...` builds a regular wheel, which omits the nested
`sampleid/src/` model modules. Editable installation is required because the Sony checkpoint refers to
`src.networks.*` modules that are not included in the upstream wheel package.

## Cloud Run

The `lastrada-eval-git` service in project `lastrada-eval-8f795` uses Cloud Build
buildpacks to install `requirements.txt`. Set
`LASTRADA_SAMPLEID_CHECKPOINT_URL` as a Cloud Run environment variable. A valid
checkpoint does not compensate for missing Python model modules: both the
checkpoint and the editable Sony source installation are required.

The **Validate Sony production runtime** workflow checks the editable installation
and imports `src.networks.resnet_ibn` on pull requests targeting
`feat/sony-dtw-main-flow`. It does not download weights or prove full inference.

Pushing or merging into `feat/sony-dtw-main-flow` triggers a production deployment.
Prepare fixes on a separate branch, review the PR and its validation results, and
approve deployment before merging. After deployment, start a new analysis and
confirm Sony runs before evaluating its candidate timestamps.
