# Lastrada · Nature of Use assistant

Local Python/Streamlit v0 for analyzing a known original/new-recording pair and drafting documentation for expert review. This is a documentation assistant, not a licensing rate calculator or legal clearance decision.

## Run

Python 3.12+ and `uv`:

```bash
cd /workspace/Lastrada
bash scripts/install.sh
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0
```

Alternatively, create a virtual environment with `python -m venv .venv`, install `requirements.lock` with pip, then install this project using `pip install --no-deps -e .` inside that environment. FFmpeg is recommended for compressed formats; soundfile supports WAV/FLAC and many MP3 files. No paid APIs are required.

## Deploy on Streamlit Community Cloud

Use repository `tankr98/Lastrada`, branch `feat/lastrada-v0`, entrypoint `app.py`, and Python 3.12. Production dependencies and native audio packages are declared in `requirements.txt` and `packages.txt`. Set app sharing to **Only specific people can view this app** before client uploads. See [deployment instructions](docs/DEPLOYMENT.md).

## A four-step creator workflow

1. **Add your tracks:** upload the original and your new track, enter both titles/artists, and choose whether you sampled the recording, replayed/re-sang the music, or need both paths checked.
2. **Listen & explore:** compare playback and suggested moments. Estimated audio overlap is a listening aid, not confirmed sample coverage. Technical details are tucked away.
3. **Describe your sample:** identify the sound or musical part, enter its appearances in your track, name sections, explain how you used it, repetition and its role. Generate and edit the Nature of Use draft. Optional original timestamps, lyrics and pitch details are retained when known.
4. **Your Nature of Use:** read the document, see any missing details and download HTML or text. HTML can be printed to PDF. The analysis JSON and automated/reviewed versions remain optional supporting material.

**Next** saves draft corrections; **Back** preserves edits. Marking a document as reviewed requires confirmed findings, checked details and a reviewer name. Incomplete drafts remain downloadable and clearly list missing inputs. Recording reuse and compositional reuse stay separate throughout.

Playback and analysis run in the hosting workspace. There is no external model API. Uploads and reviews are session-local, not a persistent case database. Downloaded documents contain case information and need approved private storage.

## Capability status

| Capability | v0 status |
|---|---|
| Audio validation, mono normalization, resampling, playback | Functional; 100 MB / 10 minute input limits |
| PF-NMF spectral activation and union coverage | Experimental, synthetic-validated baseline; not a detector |
| SampleID CPU chunk embeddings, caching, grouped candidates | CPU inference verified on generated audio; optional Sony package/checkpoint required; see setup guide |
| Publishing interpolation localization | Expert input required; not automatically implemented |
| Placement | Conservative proposals; section labels and final class need expert review |
| Element, lyrics, transformations, musical role | Unknown automatically; expert correction supported |
| Nature of Use drafting, separate reviewed versions, JSON/HTML export | Functional deterministic templates |
| Historical retrieval | Local optional spreadsheet; all reserved cases excluded |
| Real-case evaluation | Harness implemented; requires audio, confirmed labels and version checks |

Similarity and PF-NMF activation never create a verified sample claim. `unknown`, `not_verified`, `not_detected`, `candidate`, and `verified` remain separate. Lack of recording matches does not establish lack of compositional reuse.

## Validate

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/diagnose.py
.venv/bin/python scripts/synthetic_smoke.py
```

See [setup](docs/SETUP.md), [architecture and decisions](docs/TECHNICAL_DECISIONS.md), [project plan](docs/PROJECT_PLAN.md), [data dictionary](docs/DATA_DICTIONARY.md), [audit procedure](docs/DATA_AUDIT.md), [evaluation](docs/EVALUATION.md), and [open questions](docs/OPEN_QUESTIONS.md).

## Confidentiality

Client documents, historical workbooks, commercial recordings, checkpoints, caches and real analysis outputs are not repository assets. Keep them outside this public checkout or under ignored `data/private/`, `outputs/`, and `cache/`. Only synthetic fixtures belong in source control. `.gitignore` helps prevent accidents; inspect staged changes before publishing.
