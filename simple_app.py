"""Upload-first review preview. Run separately until the interface is approved."""

import hashlib
import os

import streamlit as st

from lastrada.audio.io import Audio, decode, wav_bytes
from lastrada.audio.synthetic import demo_pair
from lastrada.config import load_settings
from lastrada.pipeline import analyze_pair
from lastrada.recognition.sampleid_adapter import SampleIDAdapter, SampleIDUnavailable
from lastrada.reporting.comparison import comparison_figure, spectral_view
from lastrada.reporting.document import timestamp
from lastrada.reporting.quick_review import match_rows, quick_draft, quick_intake
from lastrada.runtime.checkpoint import resolve_sampleid_checkpoint
from lastrada.schemas.models import Interval


st.set_page_config(page_title="LaStrada · Compare tracks", page_icon="🎵", layout="centered")
st.markdown("""
<style>
.block-container {max-width:1000px;padding-top:2.5rem;padding-bottom:4rem;}
h1 {letter-spacing:-.045em;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
</style>
""", unsafe_allow_html=True)


def reset():
    for key in list(st.session_state):
        if key.startswith("simple_"):
            del st.session_state[key]


def identity(original, new, demo):
    if demo:
        return "synthetic"
    if not original or not new:
        return None
    digest = hashlib.sha256()
    for file in (original, new):
        digest.update(file.name.encode())
        digest.update(hashlib.sha256(file.getvalue()).digest())
    return digest.hexdigest()


def clip(audio, start, end):
    samples = audio.samples[int(start * audio.sample_rate):int(end * audio.sample_rate)]
    return wav_bytes(Audio(samples, audio.sample_rate, len(samples) / audio.sample_rate))


brand, restart = st.columns([5, 1])
brand.caption("LASTRADA · AUDIO COMPARISON")
restart.button("New pair", on_click=reset, width="stretch")
st.title("Two tracks. Find the connection.")
st.write("Upload the original and the new recording. We’ll suggest matching moments to compare.")

with st.expander("Try a demo"):
    demo = st.checkbox("Use synthetic demonstration audio", key="simple_demo")

has_result = "simple_result" in st.session_state
uploads = st.expander("Your tracks · change or analyze again", expanded=not has_result)
with uploads:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1 · Original")
        st.caption("The potential source recording.")
        original_file = st.file_uploader(
            "Original track", type=["mp3", "wav", "flac", "m4a", "ogg"],
            key="simple_original", help="Up to 100 MB and 10 minutes per file.",
        )
    with right:
        st.subheader("2 · New recording")
        st.caption("The track that may contain the sample.")
        new_file = st.file_uploader(
            "New track", type=["mp3", "wav", "flac", "m4a", "ogg"],
            key="simple_new", help="Up to 100 MB and 10 minutes per file.",
        )
    if demo:
        st.info("Demo uses generated audio. It does not run Sony or demonstrate real-song accuracy.")
    st.caption("Only two audio files are required. MP3, WAV, FLAC, M4A or OGG · max 10 min each.")
    run = st.button(
        "Analyze tracks", type="primary", width="stretch",
        disabled=not demo and (original_file is None or new_file is None),
    )
    st.caption("Recordings are processed on the app server. Download results before leaving this session.")

current_identity = identity(original_file, new_file, demo)
if run:
    try:
        with st.status("Preparing your tracks…", expanded=True) as progress:
            if demo:
                original, new, demo_regions = demo_pair()
            else:
                original = decode(original_file.getvalue())
                new = decode(new_file.getvalue())
            intake = quick_intake(
                original_file.name if original_file and not demo else "Synthetic source",
                new_file.name if new_file and not demo else "Synthetic mix", synthetic=demo,
            )
            settings = load_settings()
            adapter, setup_error = None, None
            if not demo:
                progress.update(label="Preparing Sony analysis…")
                st.write("The first run may take longer while the model downloads.")
                checkpoint, setup_error = resolve_sampleid_checkpoint(allow_download=True)
                if checkpoint:
                    try:
                        adapter = SampleIDAdapter(
                            checkpoint, os.path.join(os.getenv("LASTRADA_CACHE_DIR", "cache"), "sampleid"),
                            batch_size=settings.sampleid_batch_size,
                        )
                    except SampleIDUnavailable as exc:
                        setup_error = str(exc)
            progress.update(label="Finding and aligning candidate matches…")
            result = analyze_pair(intake, original, new, adapter=adapter, synthetic=demo, settings=settings)
            if demo:
                # Known insertions in generated signals illustrate the listening UI;
                # they are never substituted for failed inference on real uploads.
                result.master.locate.sample_occurrences = [
                    Interval(new_start=start, new_end=end, original_start=0,
                             original_end=original.duration, source="synthetic_known_insertion",
                             status="candidate")
                    for start, end in demo_regions
                ]
                result.master.locate.detection_status = "candidate"
                result.master.locate.sample_detected = None
                result.methods["demo_regions"] = "known_insertions_not_model_predictions"
            if setup_error:
                result.methods["sampleid"] = f"unavailable: {setup_error}"
                result.warnings.append(setup_error)
            # Replace the previous result only after the entire run succeeds.
            st.session_state.simple_result = result
            st.session_state.simple_audio = (original, new)
            st.session_state.simple_identity = current_identity
            st.session_state.simple_views = (spectral_view(original), spectral_view(new))
            st.session_state.simple_clips = {}
            st.session_state.pop("simple_draft", None)
            st.session_state.pop("simple_match", None)
            progress.update(label="Comparison ready", state="complete", expanded=False)
        st.rerun()
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Analysis could not finish: {exc}")
        st.caption("Try another file or run again. Any previous result is retained for its original pair.")

result = st.session_state.get("simple_result")
if result and current_identity != st.session_state.simple_identity:
    st.info("Your track selection changed. Click Analyze tracks to get results for this pair.")
    result = None

if result:
    st.divider()
    original, new = st.session_state.simple_audio
    candidates = result.master.locate.sample_occurrences
    sony_ran = result.methods.get("sampleid") == "local_cpu_chunk_embeddings"
    st.subheader("Listen to the suggested matches" if candidates else "Your comparison")
    if result.synthetic:
        st.info("Synthetic preview · excerpts show known insertions in generated audio, not Sony predictions.")
    elif not sony_ran:
        st.warning("Sony matching did not complete. No Sony matches are available for this run.")
        reason = result.methods.get("sampleid", "").removeprefix("unavailable:").strip()
        st.error(reason or "The Sony model was not available for this analysis.")
        st.caption("Open Your tracks above and click Analyze tracks to retry. The first model download is about 768 MB.")
    elif not candidates:
        st.info("No paired matches found under the current settings. This does not rule out reuse.")
    else:
        st.caption(f"{len(candidates)} candidate match(es) · automated suggestions, not confirmed reuse")
    left, right = st.columns(2)
    for side, col, work, audio in (
        ("original", left, result.input.original_work, original),
        ("new", right, result.input.new_work, new),
    ):
        col.markdown(f"**{work.title}**")
        col.caption(f"{work.artist} · {timestamp(audio.duration)}")
        # Full-track players are useful even when Sony returns no candidate.
        cache = st.session_state.simple_clips
        key = f"full-{side}"
        if key not in cache:
            cache[key] = wav_bytes(audio)
        col.audio(cache[key], format="audio/wav")

    if candidates:
        st.dataframe(match_rows(result), hide_index=True, width="stretch")
        index = st.selectbox(
            "Listen to a match", range(len(candidates)), key="simple_match",
            format_func=lambda i: f"Match {i + 1} · new track {timestamp(candidates[i].new_start)}–{timestamp(candidates[i].new_end)}",
        )
        item = candidates[index]
        left, right = st.columns(2)
        for label, col, audio, start, end in (
            ("Original excerpt", left, original, item.original_start, item.original_end),
            ("New recording excerpt", right, new, item.new_start, item.new_end),
        ):
            col.markdown(f"**{label}**")
            if start is not None:
                col.caption(f"{timestamp(start)} – {timestamp(end)}")
                key = (label, start, end)
                if key not in cache:
                    cache[key] = clip(audio, start, end)
                col.audio(cache[key], format="audio/wav")
            else:
                col.caption("Source timestamps unavailable for this candidate.")
        st.caption("Listen to both excerpts before drawing a conclusion. Playback is independent.")

    with st.expander("Spectral comparison · optional"):
        st.caption("These views show sound patterns. Spectral overlap is not confirmed sample use.")
        st.plotly_chart(
            comparison_figure(*st.session_state.simple_views, candidates), width="stretch"
        )

    st.divider()
    st.subheader("Keep your results")
    st.download_button(
        "Download analysis (.json)", result.model_dump_json(indent=2),
        file_name="lastrada-analysis.json", mime="application/json",
    )
    st.caption("Optional: turn the available evidence into a preliminary written draft.")
    if st.button("Create Nature of Use draft"):
        st.session_state.simple_draft = quick_draft(result)
    if "simple_draft" in st.session_state:
        st.warning("Preliminary draft · not reviewed. Unestablished musical details remain unknown.")
        draft = st.text_area("Your draft · optional edits", key="simple_draft", height=300)
        st.download_button(
            "Download draft (.txt)", draft, file_name="lastrada-draft.txt", mime="text/plain",
        )
        st.caption("Editing this text does not mark the automated findings as verified.")

    with st.expander("Analysis details"):
        st.write("Recording matching only. Replayed melody and lyrics require separate review.")
        st.json({"methods": result.methods, "timings_seconds": result.timings_seconds})
        for warning in result.warnings:
            st.caption(warning)
