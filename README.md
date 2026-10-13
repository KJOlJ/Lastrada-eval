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

## A seven-screen creator workflow

The workspace opens in navy dark mode with a light-mode toggle, progress at the top and a compact collapsible sidebar. Each screen focuses on one task, with compact buttons and a subtle transition that respects reduced-motion preferences.

1. **Add your tracks:** upload the original and your new recording.
2. **Track details:** enter titles and artists, then choose recording samples, replayed music or both.
3. **Listen & check:** play both tracks and edit the suggested timestamps. Sound comparisons and technical details are optional.
4. **Describe the sound:** name what you used, how you used it, repetition and its musical role.
5. **Place the sample:** name song sections, overall placement and any known pitch change.
6. **Review your draft:** generate and edit the Nature of Use text, add your name and save or mark it as reviewed.
7. **Your Nature of Use:** preview and download HTML or text. HTML can be printed to PDF; supporting JSON is optional.

Estimated master-analysis times populate the editable table; publishing never inherits them. Timing patterns prompt questions about recurring use, melody and short loops. Only a creator-confirmed answer fills musical details, without replacing existing edits. Suggested times need a separate listening confirmation.

**Back** and **Next** preserve session inputs; the final Next saves the review draft. Marking a document as reviewed requires confirmed findings, checked details and a reviewer name. Incomplete drafts remain downloadable and list missing inputs. Recording reuse and compositional reuse stay separate.

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
