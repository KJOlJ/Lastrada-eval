"""Local reviewer interface. UI state is session-local; exports contain no audio."""

import os
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lastrada.audio.io import decode, wav_bytes
from lastrada.audio.synthetic import demo_pair
from lastrada.pipeline import analyze_pair
from lastrada.schemas.models import Intake, Work, PLACEMENT_CLASSES, Feature, Interval, Placement
from lastrada.reporting.review import reviewed_copy
from lastrada.reporting.export import json_report, html_report
from lastrada.synthesis.narrative import draft
from lastrada.alignment.intervals import coverage
from lastrada.classification.placement import classify
from lastrada.recognition.sampleid_adapter import SampleIDAdapter, SampleIDUnavailable
from lastrada.ingestion.dataset import load_dataset, retrieve_examples
from lastrada.config import load_settings

settings = load_settings()

st.set_page_config(page_title="Lastrada · Nature of Use", page_icon="🎵", layout="wide")
st.markdown(
    """
<style>
.stApp {background:#f8f7fb;}
.block-container {max-width:1200px;padding-top:2.5rem;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stSidebar"] {background:#efebf6;border-right:1px solid #e0d9ec;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
[data-testid="stFileUploader"] {border-radius:14px;}
[data-baseweb="tab-list"] {gap:24px;margin-bottom:24px;border-bottom:1px solid #e3ddea;}
[data-baseweb="tab"] {padding:14px 0;font-weight:600;}
.stButton>button,.stFormSubmitButton>button {border-radius:10px;min-height:44px;}
.eyebrow {color:#7950b5;font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;}
.hero {margin:0 0 24px;}
.hero h1 {font-size:42px;line-height:1.12;margin:8px 0 12px;}
.hero p {color:#746b80;font-size:17px;max-width:670px;line-height:1.6;}
</style>
<div class="hero"><div class="eyebrow">Lastrada / Nature of Use</div>
<h1>Start with the music.</h1>
<p>Add the original and new recordings. Build a draft of how the music is used, then review the evidence before sharing.</p></div>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### Lastrada")
    st.caption("NATURE OF USE WORKSPACE")
    st.divider()
    st.markdown("**01 · Add your recordings**")
    st.caption("Original work, new work and requested rights.")
    st.markdown("**02 · Explore the evidence**")
    st.caption("Listen, inspect timelines and check uncertain findings.")
    st.markdown("**03 · Review the description**")
    st.caption("Add musical context and correct the draft.")
    st.markdown("**04 · Export your report**")
    st.caption("Keep the automated evidence and your reviewed version.")
    st.divider()
    st.caption("Draft assistant · Expert review required")
    with st.expander("Demo and advanced options"):
        synthetic = st.checkbox("Use a clearly labeled synthetic demonstration", key="demo")
        use_sampleid = st.checkbox("Include optional SampleID candidate matching", value=False)
        st.caption(
            "Experimental matching needs calibration. Scores do not establish accurate sample timestamps."
        )


def chart(audio, intervals=(), color="#7950B5"):
    step = max(1, len(audio.samples) // 3000)
    fig = go.Figure(
        go.Scatter(
            x=np.arange(0, len(audio.samples), step) / audio.sample_rate,
            y=audio.samples[::step],
            line={"color": color, "width": 1},
            name="Audio",
        )
    )
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                x0=interval.new_start,
                x1=interval.new_end,
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0),
            )
            for interval in intervals
        ]
    )
    fig.update_layout(
        height=230,
        xaxis_title="Seconds",
        yaxis_title="Amplitude",
        margin={"l": 20, "r": 20, "t": 10, "b": 30},
    )
    return fig


views = st.tabs(
    ["1 · Intake", "2 · Locate and measure", "3 · Classify and describe", "4 · Review and export"]
)
with views[0]:
    st.subheader("Create a new analysis")
    st.caption("Provide both recordings and tell us which rights you want to examine.")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("#### 01 · Original recording")
        st.caption("The existing work that is being sampled or re-performed.")
        original_file = st.file_uploader(
            "Original audio",
            type=["wav", "flac", "mp3", "m4a", "ogg"],
            help="MP3, WAV, FLAC, M4A or OGG · up to 100 MB · maximum 10 minutes",
        )
        if original_file:
            st.audio(original_file.getvalue(), format=original_file.type or "audio/mpeg")
            st.caption(f"{original_file.name} · {original_file.size / 1024 / 1024:.1f} MB")
    with right, st.container(border=True):
        st.markdown("#### 02 · New recording")
        st.caption("The new work that incorporates material from the original.")
        new_file = st.file_uploader(
            "New audio",
            type=["wav", "flac", "mp3", "m4a", "ogg"],
            help="MP3, WAV, FLAC, M4A or OGG · up to 100 MB · maximum 10 minutes",
        )
        if new_file:
            st.audio(new_file.getvalue(), format=new_file.type or "audio/mpeg")
            st.caption(f"{new_file.name} · {new_file.size / 1024 / 1024:.1f} MB")
    if synthetic:
        st.info("Synthetic demo selected. Generated audio will be used instead of your uploads.")
    with st.form("intake", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            original_title = st.text_input(
                "Original title", "Synthetic source" if synthetic else "", placeholder="Song title"
            )
            original_artist = st.text_input(
                "Original artist",
                "Generated tones" if synthetic else "",
                placeholder="Artist or group",
            )
        with right:
            new_title = st.text_input(
                "New title", "Synthetic mix" if synthetic else "", placeholder="Song title"
            )
            new_artist = st.text_input(
                "New artist", "Generated tones" if synthetic else "", placeholder="Artist or group"
            )
        st.divider()
        st.markdown("#### 03 · Requested rights analysis")
        scope = st.radio(
            "Requested analysis",
            ["Both", "Master", "Publishing"],
            horizontal=True,
            help="Choose the scope supplied by the requester; this is not a finding about the audio.",
        )
        st.caption(
            "Master: audio from an existing recording. Publishing: melody, harmony or lyrics, including replayed or re-sung material. Both: separate findings for each path."
        )
        with st.expander("Case reference and recording versions · optional"):
            case_id = st.text_input(
                "Case identifier",
                "synthetic-demo" if synthetic else "",
                placeholder="Your internal reference",
            )
            original_warning = st.text_input(
                "Original audio-version warning (optional)",
                placeholder="For example, radio edit or different mix",
            )
            new_warning = st.text_input(
                "New audio-version warning (optional)",
                placeholder="For example, unreleased edit or extended intro",
            )
        st.caption(
            "Your recordings are processed in this workspace. Analysis produces an experimental draft that needs expert review."
        )
        submitted = st.form_submit_button("Start analysis", type="primary", width="stretch")
    if submitted:
        try:
            if not synthetic and (not original_file or not new_file):
                raise ValueError("Add both the original and new recordings to start.")
            if not all(
                value.strip() for value in (original_title, original_artist, new_title, new_artist)
            ):
                raise ValueError("Enter a title and artist for both recordings.")
            intake = Intake(
                case_id=case_id.strip() or "local-case",
                original_work=Work(
                    title=original_title.strip(),
                    artist=original_artist.strip(),
                    audio_file=original_file.name if original_file else None,
                    audio_version_warning=original_warning or None,
                ),
                new_work=Work(
                    title=new_title.strip(),
                    artist=new_artist.strip(),
                    audio_file=new_file.name if new_file else None,
                    audio_version_warning=new_warning or None,
                ),
                analysis_scope=["publishing", "master"] if scope == "Both" else [scope.lower()],
            )
            analysis_start = perf_counter()
            t = perf_counter()
            if synthetic:
                original, new, _ = demo_pair()
            else:
                if not original_file or not new_file:
                    raise ValueError("Upload both original and new recordings")
                original, new = decode(original_file.getvalue()), decode(new_file.getvalue())
            preparation_time = perf_counter() - t
            adapter_setup_start = perf_counter()
            adapter = None
            adapter_error = None
            checkpoint = os.getenv("LASTRADA_SAMPLEID_CHECKPOINT") or (
                "checkpoints/sampleid-best.ckpt"
                if Path("checkpoints/sampleid-best.ckpt").is_file()
                else None
            )
            if use_sampleid and "master" in intake.analysis_scope:
                try:
                    adapter = SampleIDAdapter(
                        checkpoint,
                        os.getenv("LASTRADA_CACHE_DIR", "cache") + "/sampleid",
                        batch_size=settings.sampleid_batch_size,
                    )
                except SampleIDUnavailable as exc:
                    adapter_error = str(exc)
                    st.warning(adapter_error)
            adapter_setup_time = perf_counter() - adapter_setup_start
            with st.spinner("Measuring audio locally…"):
                result = analyze_pair(
                    intake, original, new, adapter=adapter, synthetic=synthetic, settings=settings
                )
            if adapter_error:
                result.methods["sampleid"] = f"unavailable: {adapter_error}"
                result.warnings.append(adapter_error)
            result.timings_seconds["audio_preparation"] = preparation_time
            result.timings_seconds["model_setup"] = adapter_setup_time
            result.timings_seconds["total"] = perf_counter() - analysis_start
            st.session_state.analysis = result
            st.session_state.original_audio = original
            st.session_state.new_audio = new
            # Clear prior case's review widgets and exports.
            for key in list(st.session_state):
                if key.startswith("review_"):
                    del st.session_state[key]
            st.success("Analysis complete. Continue to Locate and measure, then review the draft.")
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if "analysis" in st.session_state:
        active = st.session_state.analysis
        st.success(
            f"Current analysis: {active.input.new_work.title} · Continue to Locate and measure."
        )
        with st.expander("Listen to the analyzed recordings"):
            st.caption("Original recording")
            st.audio(wav_bytes(st.session_state.original_audio), format="audio/wav")
            st.caption("New recording")
            st.audio(wav_bytes(st.session_state.new_audio), format="audio/wav")

result = st.session_state.get("analysis")
with views[1]:
    if result is None:
        st.write("Provide a pair in Intake, or try the synthetic demonstration.")
    else:
        if result.synthetic:
            st.warning(
                "Synthetic demonstration · No historical audio or pretrained inference is implied."
            )
        for rights in result.input.analysis_scope:
            path = getattr(result, rights)
            st.subheader(rights.capitalize())
            st.write(f"Finding status: **{path.locate.detection_status.replace('_', ' ')}**")
            if rights == "master":
                seconds, percent = (
                    path.locate.total_covered_seconds,
                    path.locate.coverage_percentage,
                )
                st.metric(
                    "Experimental spectral activation coverage", f"{percent:.1f}% · {seconds:.2f}s"
                )
                st.caption(
                    "Shaded PF-NMF regions measure spectral activation, not verified recording reuse."
                )
                st.plotly_chart(
                    chart(st.session_state.new_audio, path.locate.measurement_intervals),
                    width="stretch",
                    key="new_measurement_chart",
                )
                st.write("Candidate recording matches")
                st.dataframe([i.model_dump() for i in path.locate.sample_occurrences])
                st.write("Experimental measurement regions")
                st.dataframe([i.model_dump() for i in path.locate.measurement_intervals])
                st.write("Original reference waveform")
                source_intervals = [
                    i.model_copy(update={"new_start": i.original_start, "new_end": i.original_end})
                    for i in path.locate.sample_occurrences
                    if i.original_start is not None
                ]
                st.plotly_chart(
                    chart(st.session_state.original_audio, source_intervals, "#3F7A89"),
                    width="stretch",
                    key="original_chart",
                )
            else:
                st.write(
                    "Composition localization is unavailable automatically. Add verified publishing "
                    "occurrences in review; master candidates are not copied into this path."
                )
            for limitation in path.describe.limitations:
                st.caption(limitation)
        st.json(result.methods)
        st.write("Processing time (seconds)", result.timings_seconds)

with views[2]:
    if result is None:
        st.write("Analyze a pair to create a structured draft.")
    else:
        st.subheader("Automated draft")
        st.write(result.nature_of_use.text)
        for rights in result.input.analysis_scope:
            path = getattr(result, rights)
            st.write(
                f"**{rights.capitalize()} placement:** "
                f"{path.classification.placement.value if path.classification.placement else 'Needs review'}"
            )
            st.caption(path.classification.reason)
        st.subheader("Expert correction")
        st.caption(
            "Enter listening-based findings independently for each rights path. "
            "Saving keeps the automated predictions alongside a separate reviewed version."
        )
        reviewed_paths = {}
        for rights in result.input.analysis_scope:
            path = getattr(result, rights).model_copy(deep=True)
            with st.expander(f"{rights.capitalize()} findings", expanded=True):
                key = f"review_{rights}_"
                status = st.selectbox(
                    "Reuse finding",
                    ["not_verified", "unknown", "candidate", "not_detected", "verified"],
                    index=[
                        "not_verified",
                        "unknown",
                        "candidate",
                        "not_detected",
                        "verified",
                    ].index(path.locate.detection_status),
                    key=key + "status",
                )
                st.caption(
                    "Verified means you have independently confirmed reuse. Not detected means no "
                    "reuse was found by the stated review, not a guarantee of absence."
                )
                rows = [
                    {
                        "new_start": i.new_start,
                        "new_end": i.new_end,
                        "original_start": i.original_start,
                        "original_end": i.original_end,
                    }
                    for i in path.locate.sample_occurrences
                ]
                intervals = st.data_editor(
                    pd.DataFrame(
                        rows, columns=["new_start", "new_end", "original_start", "original_end"]
                    ),
                    num_rows="dynamic",
                    key=key + "intervals",
                    column_config={
                        c: st.column_config.NumberColumn(c, min_value=0.0, step=0.01)
                        for c in ["new_start", "new_end", "original_start", "original_end"]
                    },
                )
                element = st.text_input(
                    "Musical element / instrumentation (only if identified)", key=key + "element"
                )
                lyrics = st.text_input("Reused lyrics (only if identified)", key=key + "lyrics")
                reuse_type = st.text_input(
                    "Reuse method (e.g. replayed, resung, direct recording reuse)",
                    key=key + "method",
                )
                repetition = st.text_input("Repetition / looping pattern", key=key + "repetition")
                role = st.text_input("Role in the new recording", key=key + "role")
                pitch = st.number_input(
                    "Pitch alteration in semitones (only if verified)", value=0.0, key=key + "pitch"
                )
                pitch_known = st.checkbox(
                    "Pitch alteration independently verified", key=key + "pitch_known"
                )
                sections = st.multiselect(
                    "Sections",
                    ["intro", "verse", "chorus", "bridge", "outro"],
                    key=key + "sections",
                )
                placement = st.selectbox(
                    "Placement class", ["Needs review"] + PLACEMENT_CLASSES, key=key + "placement"
                )
                confirmed = st.checkbox(
                    "I have checked this path’s findings and classification", key=key + "checked"
                )
                try:
                    occurrences = []
                    for row in intervals.to_dict("records"):
                        if pd.isna(row["new_start"]) and pd.isna(row["new_end"]):
                            continue
                        occurrences.append(
                            Interval(
                                new_start=row["new_start"],
                                new_end=row["new_end"],
                                original_start=None
                                if pd.isna(row["original_start"])
                                else row["original_start"],
                                original_end=None
                                if pd.isna(row["original_end"])
                                else row["original_end"],
                                source="expert_review",
                                status="expert_verified",
                            )
                        )
                    path.locate.sample_occurrences = occurrences
                    path.locate.detection_status = status
                    path.locate.sample_detected = (
                        True
                        if status == "verified"
                        else False
                        if status == "not_detected"
                        else None
                    )
                    path.locate.pitch_shift_semitones = pitch if pitch_known else None
                    path.locate.review_required = not confirmed
                    if occurrences and status == "verified":
                        sec, pct = coverage(occurrences, result.durations["new"])
                        path.locate.total_covered_seconds, path.locate.coverage_percentage = (
                            sec,
                            pct,
                        )
                    elif status == "not_detected":
                        if occurrences:
                            raise ValueError("Remove reuse occurrences for a not-detected finding")
                        path.locate.total_covered_seconds, path.locate.coverage_percentage = (
                            0.0,
                            0.0,
                        )
                    else:
                        # Unverified measurements remain in the automated snapshot, not reviewed truth.
                        path.locate.total_covered_seconds = None
                        path.locate.coverage_percentage = None
                    path.describe.elements = [
                        Feature(
                            feature=f,
                            value=v,
                            source="expert_review",
                            status="expert_verified",
                            review_required=not confirmed,
                        )
                        for f, v in [("musical_element", element), ("lyrics", lyrics)]
                        if v
                    ]
                    path.describe.sections, path.describe.reuse_type = sections, reuse_type or None
                    path.describe.repetition, path.describe.role_in_new_song = (
                        repetition or None,
                        role or None,
                    )
                    path.classification = classify(
                        path.locate.coverage_percentage, occurrences, sections, expert=True
                    )
                    path.classification.placement = (
                        None if placement == "Needs review" else Placement(placement)
                    )
                    path.classification.source = "expert_review"
                    path.classification.review_required = not confirmed or (
                        status == "verified" and placement == "Needs review"
                    )
                    reviewed_paths[rights] = path.model_dump(mode="json")
                except ValueError as exc:
                    st.error(f"{rights.capitalize()}: {exc}")
        candidate = result.model_copy(deep=True)
        for rights, value in reviewed_paths.items():
            from lastrada.schemas.models import PathResult

            setattr(candidate, rights, PathResult.model_validate(value))
        generated = draft(candidate).text
        if st.button("Fill narrative from reviewed findings"):
            st.session_state.review_narrative = generated
        narrative = st.text_area(
            "Editable Nature of Use",
            value=result.nature_of_use.text,
            height=180,
            key="review_narrative",
        )
        reviewer = st.text_input("Reviewer name", key="review_name")
        notes = st.text_area("Review notes / remaining uncertainty", key="review_notes")
        left, right = st.columns(2)
        save = left.button("Save corrections")
        approve = right.button("Approve reviewed draft", type="primary")
        if save or approve:
            try:
                if len(reviewed_paths) != len(result.input.analysis_scope):
                    raise ValueError("Resolve invalid fields before saving")
                st.session_state.analysis = reviewed_copy(
                    result, reviewed_paths, narrative, reviewer, approve=approve, notes=notes
                )
                st.success(
                    "Reviewed version saved for this session. Export it in Review and export."
                )
                result = st.session_state.analysis
            except ValueError as exc:
                st.error(str(exc))
        with st.expander("Optional local historical references"):
            st.caption(
                "Reference text is separate from predictions. Reserved test cases are always excluded. "
                "Client data stays local and must not be committed."
            )
            spreadsheet = st.file_uploader(
                "Local reference spreadsheet", type=["xlsx"], key="review_reference_upload"
            )
            if spreadsheet:
                try:
                    reference_frame = load_dataset(spreadsheet)
                    selection = st.selectbox(
                        "Retrieve by reviewed placement",
                        PLACEMENT_CLASSES,
                        key="review_reference_class",
                    )
                    st.write(
                        retrieve_examples(reference_frame, selection, exclude_ids=[result.case_id])
                    )
                except ValueError as exc:
                    st.error(str(exc))

with views[3]:
    if result is None:
        st.write("An analysis is required before export.")
    else:
        st.write(f"Review status: **{result.review.status}**")
        st.write(
            "Download reports before leaving the session. They contain confidential case information; "
            "keep them in your approved private storage."
        )
        with st.expander("Original automated output"):
            st.json(result.model_dump(mode="json", exclude={"review"}))
        with st.expander(
            "Separate expert corrections", expanded=result.review.corrected is not None
        ):
            st.json(result.review.model_dump(mode="json"))
        # Fixed filenames avoid accidental path/HTML injection from user-provided identifiers.
        st.download_button(
            "Export JSON", json_report(result), "lastrada-analysis.json", "application/json"
        )
        st.download_button(
            "Export HTML report", html_report(result), "lastrada-report.html", "text/html"
        )
