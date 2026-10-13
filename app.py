"""Local reviewer interface. UI state is session-local; exports contain no audio."""

import os
from copy import deepcopy
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
from lastrada.reporting.export import json_report
from lastrada.reporting.document import (
    document_text,
    document_html,
    missing_document_inputs,
    creator_narrative,
)
from lastrada.alignment.intervals import coverage
from lastrada.classification.placement import classify
from lastrada.recognition.sampleid_adapter import SampleIDAdapter, SampleIDUnavailable
from lastrada.ingestion.dataset import load_dataset, retrieve_examples
from lastrada.config import load_settings
from lastrada.reporting.suggestions import suggested_intervals, timing_questions, confirmed_prefills

settings = load_settings()

st.set_page_config(
    page_title="Lastrada · Nature of Use",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)
WORKFLOW_STEPS = [
    "1 · Add your tracks",
    "2 · Listen & explore",
    "3 · Describe your sample",
    "4 · Your Nature of Use",
]


def go_to_step(index):
    # Callbacks execute before rendering; they can safely update the tab widget.
    st.session_state.workflow_tab = WORKFLOW_STEPS[index if "analysis" in st.session_state else 0]


def validate_step():
    if "analysis" not in st.session_state:
        st.session_state.workflow_tab = WORKFLOW_STEPS[0]


if "pending_narrative" in st.session_state:
    st.session_state.review_narrative = st.session_state.pop("pending_narrative")
if "pending_workflow_tab" in st.session_state:
    st.session_state.workflow_tab = st.session_state.pop("pending_workflow_tab")
legacy_steps = [
    "1 · Intake",
    "2 · Locate and measure",
    "3 · Classify and describe",
    "4 · Review and export",
]
if st.session_state.get("workflow_tab") not in WORKFLOW_STEPS:
    previous = st.session_state.get("workflow_tab")
    st.session_state.workflow_tab = WORKFLOW_STEPS[
        legacy_steps.index(previous) if previous in legacy_steps else 0
    ]
validate_step()

brand, appearance = st.columns([5, 1])
brand.caption("LASTRADA · NATURE OF USE")
with appearance:
    dark_mode = st.toggle("Dark mode", value=True, key="dark_mode")
current_step = WORKFLOW_STEPS.index(st.session_state.get("workflow_tab", WORKFLOW_STEPS[0]))
st.progress(
    (current_step + 1) / 4,
    text=f"STEP {current_step + 1} OF 4 · {WORKFLOW_STEPS[current_step].split(' · ')[1]}",
)
palette = (
    dict(
        bg="#091426",
        surface="#14243b",
        field="#1b2e49",
        text="#f5f8ff",
        muted="#b9c7db",
        border="#334963",
    )
    if dark_mode
    else dict(
        bg="#f7f9fc",
        surface="#ffffff",
        field="#edf2f8",
        text="#14243b",
        muted="#52647a",
        border="#d5dfeb",
    )
)
st.markdown(
    f"""
<style>
.stApp,[data-testid="stHeader"] {{background:{palette["bg"]};color:{palette["text"]};}}
.block-container {{max-width:1180px;padding-top:4rem;}}
h1,h2,h3,h4 {{letter-spacing:-.035em;color:{palette["text"]} !important;}}
[data-testid="stSidebar"] {{background:{palette["surface"]};border:0;}}
[data-testid="stSidebar"][aria-expanded="true"] {{width:230px !important;min-width:230px !important;max-width:230px !important;}}
[data-testid="stSidebar"][aria-expanded="false"] {{width:0 !important;min-width:0 !important;max-width:0 !important;}}
[data-testid="stHeader"] button {{color:{palette["text"]} !important;}}
.stCaption,.stCaption p,[data-testid="stCaption"] p,small {{color:{palette["muted"]} !important;}}
[data-testid="stFileUploaderDropzone"] small {{color:{palette["muted"]} !important;}}
[data-testid="stAlert"] {{background:{palette["surface"]};color:{palette["text"]};border:1px solid {palette["border"]};}}
[data-testid="stSidebar"] p {{font-size:.9rem;}}
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],label {{color:{palette["text"]} !important;}}
[data-testid="stCaptionContainer"] p {{color:{palette["muted"]} !important;}}
[data-testid="stVerticalBlockBorderWrapper"]>div {{border-color:{palette["border"]} !important;border-radius:16px;}}
[data-testid="stFileUploaderDropzone"],input,textarea,[data-baseweb="select"]>div {{background:{palette["field"]} !important;color:{palette["text"]} !important;border-color:{palette["border"]} !important;}}
input::placeholder,textarea::placeholder {{color:{palette["muted"]} !important;}}
[data-baseweb="popover"],[role="listbox"],[role="option"] {{background:{palette["surface"]} !important;color:{palette["text"]} !important;}}
[data-testid="stExpander"] details {{background:{palette["surface"]};border-color:{palette["border"]};}}
[data-baseweb="tab-list"] {{gap:20px;margin-bottom:20px;border-bottom:1px solid {palette["border"]};}}
[data-baseweb="tab"] {{color:{palette["muted"]};font-weight:700;}}
[data-baseweb="tab"][aria-selected="true"] {{color:{palette["text"]};}}
button {{border-radius:10px !important;}}
button[kind="primary"],button[kind="primaryFormSubmit"] {{background:#dceaff !important;color:#091426 !important;font-weight:700;border-color:#dceaff;}}
[data-testid="stBaseButton-secondary"] {{background:{palette["surface"]};color:{palette["text"]};border-color:{palette["border"]};}}
.hero h1 {{font-size:42px;line-height:1.1;margin:22px 0 10px;}}
.hero p {{color:{palette["muted"]};font-size:17px;max-width:680px;line-height:1.6;}}
.st-key-nature_document {{background:#fff;color:#14243b;border-radius:16px;}}
.st-key-nature_document h3,.st-key-nature_document p,.st-key-nature_document [data-testid="stMarkdownContainer"] {{color:#14243b !important;}}
.st-key-nature_document [data-testid="stCaptionContainer"] p {{color:#52647a !important;}}
</style>
<div class="hero"><h1>Your sample. Your document.</h1>
<p>Upload two tracks. Check the sample. Tell us how you used it. Download your Nature of Use.</p></div>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### Lastrada")
    st.caption("NATURE OF USE WORKSPACE")
    st.divider()
    st.markdown("**01 · Add your tracks**")
    st.caption("Original track. Your track. Titles and artists.")
    st.markdown("**02 · Listen & explore**")
    st.caption("Listen to both tracks. Check the suggested times.")
    st.markdown("**03 · Describe your sample**")
    st.caption("Confirm the times. Tell us about the sound.")
    st.markdown("**04 · Your Nature of Use**")
    st.caption("Review the description and download your Nature of Use document.")
    st.divider()
    st.caption("Made for creators · Review before sharing")
    with st.expander("Demo and advanced options"):
        synthetic = st.checkbox("Use a clearly labeled synthetic demonstration", key="demo")
        use_sampleid = st.checkbox("Include optional SampleID candidate matching", value=False)
        st.caption(
            "Experimental matching needs calibration. Scores do not establish accurate sample timestamps."
        )


def apply_suggestion(rights, kind):
    prefix = f"review_{rights}_"
    for field, value in confirmed_prefills(kind, rights).items():
        existing = st.session_state.get(prefix + field)
        if existing in (None, "", "Needs review"):
            st.session_state[prefix + field] = value


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
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=palette["bg"],
        plot_bgcolor=palette["surface"],
    )
    return fig


views = st.tabs(WORKFLOW_STEPS, key="workflow_tab", on_change=validate_step)

with views[0]:
    st.subheader("Step 1 · Add your tracks")
    with st.expander("What you’ll need for a complete Nature of Use"):
        st.markdown(
            "**Now:** two audio files, titles, artists and type of reuse.\n\n**After listening:** the sound or musical part you used, times it appears in your track, song sections, how you used it, repetition, and its role in the track.\n\n**To finish:** a checked description and a reviewer’s name. You can download an unfinished draft; missing details stay clearly listed."
        )
    st.caption(
        "You’ll need the original track, your new track, and the title and artist for each. MP3 files work."
    )
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("#### The original track")
        st.caption("The track you sampled, or the song whose melody or lyrics you replayed.")
        original_file = st.file_uploader(
            "Original audio",
            type=["wav", "flac", "mp3", "m4a", "ogg"],
            help="MP3, WAV, FLAC, M4A or OGG · up to 100 MB · maximum 10 minutes",
        )
        if original_file:
            st.audio(original_file.getvalue(), format=original_file.type or "audio/mpeg")
            st.caption(f"{original_file.name} · {original_file.size / 1024 / 1024:.1f} MB")
    with right, st.container(border=True):
        st.markdown("#### Your new track")
        st.caption("Your finished track or current mix containing the sample.")
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
        st.markdown("#### What kind of reuse are you documenting?")
        scope = st.radio(
            "Type of reuse",
            ["Both", "Master", "Publishing"],
            horizontal=True,
            format_func=lambda x: {
                "Master": "Sampled the recording",
                "Publishing": "Replayed or re-sang it",
                "Both": "Both / check both",
            }[x],
            help="Using the actual recording involves master rights. Replaying a melody or re-singing lyrics involves publishing rights. These are separate findings.",
        )
        st.caption(
            "Not sure? Choose both. Recording samples and replayed melodies or lyrics are documented separately."
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
        submitted = st.form_submit_button("Find suggested moments", type="primary", width="stretch")
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
            st.success(
                "Your tracks are ready. Next, listen and explore, then tell us how you used the sample."
            )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if "analysis" in st.session_state:
        active = st.session_state.analysis
        st.success(
            f"Your track: {active.input.new_work.title} · Next, listen and explore the suggested moments."
        )
        with st.expander("Listen to the analyzed recordings"):
            st.caption("Original recording")
            st.audio(wav_bytes(st.session_state.original_audio), format="audio/wav")
            st.caption("New recording")
            st.audio(wav_bytes(st.session_state.new_audio), format="audio/wav")

    st.divider()
    st.button(
        "Next: Listen & explore →",
        key="next_intake",
        type="primary",
        width="stretch",
        disabled="analysis" not in st.session_state,
        on_click=go_to_step,
        args=(1,),
    )
    if "analysis" not in st.session_state:
        st.caption("Start an analysis above to continue.")

result = st.session_state.get("analysis")
with views[1]:
    st.subheader("Step 2 · Listen & explore")
    st.caption(
        "Compare the tracks and check the suggested moments. The analysis is a starting point, not a substitute for listening."
    )
    if result is None:
        st.write("Provide a pair in Intake, or try the synthetic demonstration.")
    else:
        if result.synthetic:
            st.warning(
                "Synthetic demonstration · No historical audio or pretrained inference is implied."
            )
        for rights in result.input.analysis_scope:
            path = getattr(result, rights)
            st.subheader("Recorded audio" if rights == "master" else "Melody & lyrics")
            st.caption(
                {
                    "candidate": "Suggested matches · check by listening",
                    "not_verified": "Waiting for your listening check",
                    "unknown": "More information needed",
                    "not_detected": "No matches found by this check",
                    "verified": "Checked by the reviewer",
                }.get(path.locate.detection_status, "Not requested")
            )
            if rights == "master":
                seconds, percent = (
                    path.locate.total_covered_seconds,
                    path.locate.coverage_percentage,
                )
                st.metric(
                    "Estimated audio overlap — experimental", f"{percent:.1f}% · {seconds:.2f}s"
                )
                st.caption(
                    "Shading marks moments with similar sound patterns. Shared instruments can also create overlap, so this is not a confirmed sample percentage."
                )
                st.plotly_chart(
                    chart(st.session_state.new_audio, path.locate.measurement_intervals),
                    width="stretch",
                    key="new_measurement_chart",
                )
                st.write("Suggested moments to listen to")
                if path.locate.sample_occurrences:
                    st.dataframe(
                        [
                            {
                                "Your track starts (s)": i.new_start,
                                "Ends (s)": i.new_end,
                                "Original starts (s)": i.original_start,
                                "Original ends (s)": i.original_end,
                            }
                            for i in path.locate.sample_occurrences
                        ]
                    )
                else:
                    st.caption(
                        "No sample timestamps have been confirmed. Add the moments you hear in step 3."
                    )
                with st.expander("See estimated overlap times"):
                    st.dataframe(
                        [
                            {"Start in your track (s)": i.new_start, "End (s)": i.new_end}
                            for i in path.locate.measurement_intervals
                        ]
                    )
                st.write("The original track")
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
                    "Replayed melodies and re-sung lyrics need your listening notes. The tool cannot automatically locate them yet. Describe them separately from any recording sample in step 3."
                )
        with st.expander("How the analysis works · technical details"):
            st.caption(
                "The tool compares sound patterns. It cannot reliably identify instruments, lyrics or musical sections for you yet. Your listening notes complete the document."
            )
            st.json(result.methods)
            st.write("Processing time (seconds)", result.timings_seconds)
            for rights in result.input.analysis_scope:
                st.write(getattr(result, rights).describe.limitations)

    st.divider()
    back, forward = st.columns(2)
    back.button("← Back: Add your tracks", key="back_evidence", on_click=go_to_step, args=(0,))
    forward.button(
        "Next: Describe your sample →",
        key="next_evidence",
        type="primary",
        width="stretch",
        disabled=result is None,
        on_click=go_to_step,
        args=(2,),
    )

with views[2]:
    st.subheader("Step 3 · Describe your sample")
    if result is None:
        st.write("Analyze a pair to create a structured draft.")
    else:
        st.subheader("Your working description")
        st.write(creator_narrative(result))
        st.caption(
            "Tell us what a listener should know: what you took, where it appears, how it repeats, and the role it plays in your track. These details shape the final document."
        )
        reviewed_paths = {}
        for rights in result.input.analysis_scope:
            path = getattr(result, rights).model_copy(deep=True)
            with st.container(border=True):
                st.subheader(
                    "Your recording sample"
                    if rights == "master"
                    else "Your replayed melody / lyrics"
                )
                key = f"review_{rights}_"
                status = st.selectbox(
                    "Have you checked this use?",
                    ["not_verified", "unknown", "candidate", "not_detected", "verified"],
                    index=[
                        "not_verified",
                        "unknown",
                        "candidate",
                        "not_detected",
                        "verified",
                    ].index(path.locate.detection_status),
                    key=key + "status",
                    format_func=lambda x: {
                        "not_verified": "I haven’t checked it yet",
                        "unknown": "I’m not sure",
                        "candidate": "Possible use — needs checking",
                        "not_detected": "I didn’t find reuse in this path",
                        "verified": "I’ve confirmed this use",
                    }[x],
                )
                st.caption(
                    "Choose ‘I’ve confirmed this use’ only after listening. Similarity scores cannot confirm a sample for you."
                )
                with st.expander("1 · Check the times", expanded=True):
                    st.markdown(
                        "**Where can we hear it?** Add a row for each appearance in your track. Enter seconds (1:15 = 75). Original-track times are optional when you haven’t located them yet."
                    )
                    if key + "seed_intervals" not in st.session_state:
                        st.session_state[key + "seed_intervals"] = suggested_intervals(
                            result, rights
                        )
                    seeds = st.session_state[key + "seed_intervals"]
                    if seeds:
                        st.caption(
                            "Estimated times are filled in below. Listen, edit or delete them. They are not confirmed sample times."
                        )
                    rows = [
                        {
                            "new_start": i.new_start,
                            "new_end": i.new_end,
                            "original_start": i.original_start,
                            "original_end": i.original_end,
                        }
                        for i in seeds
                    ]
                    intervals = st.data_editor(
                        pd.DataFrame(
                            rows, columns=["new_start", "new_end", "original_start", "original_end"]
                        ),
                        num_rows="dynamic",
                        disabled=status == "not_detected",
                        key=key + "intervals",
                        column_config={
                            c: st.column_config.NumberColumn(
                                {
                                    "new_start": "Your track · start (s)",
                                    "new_end": "Your track · end (s)",
                                    "original_start": "Original · start (s)",
                                    "original_end": "Original · end (s)",
                                }[c],
                                min_value=0.0,
                                step=0.01,
                            )
                            for c in ["new_start", "new_end", "original_start", "original_end"]
                        },
                    )
                    times_confirmed = st.checkbox(
                        "I’ve listened and confirmed these times", key=key + "times_confirmed"
                    )
                    if status == "not_detected":
                        st.caption(
                            "No-use finding selected. Estimated times will not be included as sample use."
                        )
                    try:
                        current_times = [
                            Interval(
                                new_start=r["new_start"],
                                new_end=r["new_end"],
                                source="timing_question",
                            )
                            for r in intervals.to_dict("records")
                            if not pd.isna(r["new_start"]) and not pd.isna(r["new_end"])
                        ]
                        questions = timing_questions(current_times, result.durations["new"])
                    except ValueError:
                        questions = []
                    for suggestion in questions if status != "not_detected" else []:
                        answer = st.radio(
                            suggestion.question,
                            ["Not sure", "Yes", "No"],
                            horizontal=True,
                            key=key + "suggestion_" + suggestion.kind,
                        )
                        if answer == "Yes":
                            st.button(
                                "Use this answer to fill the details",
                                key=key + "apply_" + suggestion.kind,
                                on_click=apply_suggestion,
                                args=(rights, suggestion.kind),
                            )
                    st.caption(
                        "A timing pattern cannot identify a melody or prove a loop. Suggestions are added only when you say yes."
                    )
                with st.expander("2 · Describe the sound", expanded=times_confirmed):
                    element = st.text_input(
                        "What did you take?",
                        placeholder="For example: a vocal phrase, drum break or guitar melody",
                        key=key + "element",
                    )
                    lyrics = st.text_input("Any reused lyrics? (optional)", key=key + "lyrics")
                    reuse_type = st.text_input(
                        "How did you use it?",
                        key=key + "method",
                        placeholder="For example: chopped the original recording, replayed the melody",
                    )
                    repetition = st.text_input(
                        "Does it loop or repeat?",
                        placeholder="For example: a four-bar loop under each verse",
                        key=key + "repetition",
                    )
                    role = st.text_input(
                        "What does it do in your track?",
                        placeholder="For example: the main hook, or a texture under new vocals",
                        key=key + "role",
                    )
                with st.expander("3 · Add sections and changes", expanded=False):
                    pitch = st.number_input(
                        "Did you pitch it up or down? (semitones, optional)",
                        value=0.0,
                        key=key + "pitch",
                    )
                    pitch_known = st.checkbox(
                        "I know this pitch change is correct", key=key + "pitch_known"
                    )
                    sections = st.multiselect(
                        "Which sections of your track?",
                        ["intro", "verse", "chorus", "bridge", "outro"],
                        key=key + "sections",
                    )
                    placement = st.selectbox(
                        "Where does it appear overall?",
                        ["Needs review"] + PLACEMENT_CLASSES,
                        key=key + "placement",
                        format_func=lambda x: {
                            "Needs review": "Still deciding",
                            "throughout": "Throughout the track",
                            "intro_outro_with_chorus": "Intro / outro and chorus",
                            "chorus_only": "Choruses only",
                            "intro_outro_without_chorus": "Intro / outro, outside the chorus",
                            "verses": "Verses",
                            "one_time": "One appearance",
                        }[x],
                    )
                    confirmed = st.checkbox(
                        "I’ve listened and checked these details", key=key + "checked"
                    )
                try:
                    occurrences = []
                    for row in intervals.to_dict("records"):
                        if status == "not_detected":
                            continue
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
                                source="expert_review"
                                if times_confirmed
                                else "unconfirmed_timing_suggestion",
                                status="expert_verified" if times_confirmed else "estimated",
                            )
                        )
                    path.locate.sample_occurrences = occurrences
                    effective_status = (
                        "candidate"
                        if status == "verified" and not times_confirmed and occurrences
                        else status
                    )
                    path.locate.detection_status = effective_status
                    path.locate.sample_detected = (
                        True
                        if effective_status == "verified"
                        else False
                        if status == "not_detected"
                        else None
                    )
                    path.locate.pitch_shift_semitones = pitch if pitch_known else None
                    path.locate.review_required = not confirmed
                    if occurrences and effective_status == "verified":
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
        remaining = missing_document_inputs(candidate)
        if remaining:
            with st.expander(f"What’s still needed? · {len(remaining)} details"):
                for item in remaining:
                    st.write("• " + item)
        else:
            st.success(
                "Your musical details are complete. Generate the draft and give it a final read."
            )
        generated = creator_narrative(candidate)
        if st.button("Generate Nature of Use draft", type="primary"):
            st.session_state.review_narrative = generated
            st.session_state.review_generated_text = generated
        narrative = st.text_area(
            "Edit your Nature of Use draft",
            value=creator_narrative(result),
            height=180,
            key="review_narrative",
        )
        reviewer = st.text_input(
            "Your name / reviewer (required to mark as reviewed)", key="review_name"
        )
        notes = st.text_area("Anything still to check? (optional)", key="review_notes")
        left, right = st.columns(2)
        save = left.button("Save draft")
        approve = right.button("Mark document as reviewed", type="primary")
        st.divider()
        st.caption(
            "Next saves your description as a draft. Marking it as reviewed is a separate action."
        )
        back, forward = st.columns(2)
        back.button("← Back: Listen & explore", key="back_draft", on_click=go_to_step, args=(1,))
        proceed = forward.button(
            "Next: Your Nature of Use →", key="next_draft", type="primary", width="stretch"
        )
        if save or approve or proceed:
            try:
                if len(reviewed_paths) != len(result.input.analysis_scope):
                    raise ValueError("Resolve invalid fields before saving")
                if narrative in (
                    st.session_state.get("review_generated_text"),
                    creator_narrative(result),
                ):
                    narrative = generated
                updated = reviewed_copy(
                    result, reviewed_paths, narrative, reviewer, approve=approve, notes=notes
                )
                if approve and missing_document_inputs(updated):
                    raise ValueError(
                        "Complete the musical details before marking this document as reviewed: "
                        + "; ".join(missing_document_inputs(updated))
                    )
                if proceed and result.review.status == "approved":
                    previous_payload = deepcopy(result.review.corrected)
                    updated_payload = deepcopy(updated.review.corrected)
                    for payload in (previous_payload, updated_payload):
                        payload["nature_of_use"].pop("review_required", None)
                    if (
                        previous_payload == updated_payload
                        and reviewer == result.review.reviewer
                        and notes == result.review.notes
                    ):
                        updated = result
                st.session_state.analysis = updated
                st.success("Draft saved for this session. Your document is in step 4.")
                result = st.session_state.analysis
                if proceed:
                    st.session_state.pending_narrative = narrative
                    st.session_state.pending_workflow_tab = WORKFLOW_STEPS[3]
                    st.rerun()
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
    st.subheader("Step 4 · Your Nature of Use")
    st.caption(
        "The finished product: a clear account of what you took and how you used it. Review the text, then download your document."
    )
    if result is None:
        st.write("Add your tracks and work through the first three steps to create your document.")
    else:
        missing = missing_document_inputs(result)
        if missing:
            st.warning("This is an unfinished draft. These details still need your input:")
            for item in missing:
                st.write("• " + item)
        elif result.review.status != "approved":
            st.info(
                "Your details are filled in. Give the description a final listening check and mark it as reviewed in step 3."
            )
        else:
            st.success("Reviewed Nature of Use · ready to download")
        source = result.review.corrected or result.model_dump(mode="json")
        with st.container(border=True, key="nature_document"):
            st.caption(
                "NATURE OF USE · " + ("REVIEWED" if result.review.status == "approved" else "DRAFT")
            )
            st.subheader(result.input.new_work.title + " — " + result.input.new_work.artist)
            st.caption(
                "Original: "
                + result.input.original_work.title
                + " — "
                + result.input.original_work.artist
            )
            st.write(source["nature_of_use"]["text"])
        left, right = st.columns(2)
        left.download_button(
            "Download Nature of Use · HTML",
            document_html(result),
            "nature-of-use.html",
            "text/html",
            type="primary",
            width="stretch",
        )
        right.download_button(
            "Download Nature of Use · Text",
            document_text(result),
            "nature-of-use.txt",
            "text/plain",
            width="stretch",
        )
        st.caption(
            "The HTML document can be opened in a browser and printed to PDF. Download before leaving; drafts are saved for this session only."
        )
        with st.expander("Supporting analysis and saved versions"):
            st.json(result.model_dump(mode="json", exclude={"review"}))
            st.json(result.review.model_dump(mode="json"))
            st.download_button(
                "Download analysis data · JSON",
                json_report(result),
                "lastrada-analysis.json",
                "application/json",
            )
    st.divider()
    st.button("← Back: Describe your sample", key="back_export", on_click=go_to_step, args=(2,))
