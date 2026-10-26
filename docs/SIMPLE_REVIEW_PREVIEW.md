# Upload-first interface preview

Run `streamlit run simple_app.py` to review the new interface locally. The existing
`app.py` production entrypoint and its seven-step review workflow are unchanged.

Local runs do not inherit Cloud Run environment variables. The preview defaults
to the existing public Sony checkpoint URL on Zenodo, downloads it only when a
real analysis is requested, and reuses the local cache afterward. The initial
download is approximately 768 MB. `LASTRADA_SAMPLEID_CHECKPOINT_URL` and
`LASTRADA_SAMPLEID_CHECKPOINT` still override the defaults. Failed Sony runs show
the underlying reason directly above playback, not only in technical details.

The new flow requires only an original recording and a new recording. The user
clicks **Analyze tracks** once to run the existing Sony + guarded DTW pipeline.
Filename credits are inferred where recognizable; otherwise the filename and an
explicit `Unknown` artist are used. No title, artist, rights, musical-description,
or reviewer form blocks analysis.

Results show candidate timestamps and independent paired audio excerpts. Full
recordings remain playable even if no candidates are returned. Spectral charts,
method diagnostics, and JSON export are optional. Selecting excerpts, generating
or editing a draft, and downloading results do not rerun analysis. Changing the
uploads hides old results until the new pair is analyzed.

**Create Nature of Use draft** is optional and produces an editable preliminary
text based on available candidate evidence. It does not approve findings, infer
instruments or song sections, or convert spectral overlap to verified reuse.
Publishing analysis is explicitly outside this minimal recording-match workflow.
The synthetic demo illustrates known inserted regions, not model accuracy.

The analysis engine is unchanged. Demucs and stem-level evidence in the planning
diagram remain separate technical work; this preview does not claim to implement
them. Uploads, audio clips, and results stay in the current Streamlit session.

Review the preview before changing Cloud Build's `GOOGLE_ENTRYPOINT` from `app.py`
to `simple_app.py`. No production entrypoint change is included here.
