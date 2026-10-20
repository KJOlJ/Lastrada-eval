"""Apply the Sony SampleID -> guarded DTW flow directly to ``app.py``.

This migration is intentionally deterministic and idempotent. It updates the
normal seven-step Lastrada workflow rather than exposing the engineering test
bench as a separate Streamlit page.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
DTW_PAGE = ROOT / "pages" / "8_DTW_Lab.py"
MARKER = "# LASTRADA_SONY_DTW_MAIN_FLOW_V1"

SETTINGS_ANCHOR = "settings = load_settings()\n\nst.set_page_config(\n"
SETTINGS_REPLACEMENT = '''settings = load_settings()

# LASTRADA_SONY_DTW_MAIN_FLOW_V1
_SAMPLEID_MIN_BYTES = 100 * 1024 * 1024


def _runtime_setting(name):
    value = os.getenv(name)
    if value:
        return str(value).strip()
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


def _valid_sampleid_checkpoint(path):
    if not path:
        return False
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.stat().st_size < _SAMPLEID_MIN_BYTES:
        return False
    try:
        import zipfile

        return zipfile.is_zipfile(candidate)
    except OSError:
        return False


def _download_sampleid_checkpoint(url):
    import requests

    cache_root = Path(_runtime_setting("LASTRADA_CACHE_DIR") or "cache") / "sampleid"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / "sampleid-best.ckpt"
    partial = destination.with_suffix(".ckpt.part")
    if _valid_sampleid_checkpoint(destination):
        return destination.resolve()

    partial.unlink(missing_ok=True)
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 300),
            headers={"User-Agent": "Lastrada-SampleID/1.0"},
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not _valid_sampleid_checkpoint(partial):
            raise RuntimeError(
                "downloaded file is incomplete or is not a valid PyTorch checkpoint"
            )
        partial.replace(destination)
        return destination.resolve()
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def resolve_sampleid_checkpoint(override=None, allow_download=False):
    candidates = []
    if override and str(override).strip():
        candidates.append(Path(str(override).strip()).expanduser())
    configured_path = _runtime_setting("LASTRADA_SAMPLEID_CHECKPOINT")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.extend(
        [
            Path("checkpoints/sampleid-best.ckpt"),
            Path.home() / ".lastrada_sampleid" / "sampleid-best.ckpt",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid_sampleid_checkpoint(resolved):
            return str(resolved), None

    download_url = _runtime_setting("LASTRADA_SAMPLEID_CHECKPOINT_URL")
    if allow_download and download_url:
        try:
            downloaded = _download_sampleid_checkpoint(download_url)
            return str(downloaded), None
        except Exception as exc:
            return None, f"Sony checkpoint download failed: {exc}"

    return None, (
        "Sony checkpoint not found. For local use, place sampleid-best.ckpt in "
        "~/.lastrada_sampleid/ or checkpoints/. For Streamlit Cloud, configure "
        "LASTRADA_SAMPLEID_CHECKPOINT_URL as a direct-download secret."
    )


st.set_page_config(
'''

SIDEBAR_ANCHOR = '''    with st.expander("Demo and advanced options"):
        synthetic = st.checkbox("Use a clearly labeled synthetic demonstration", key="demo")
        use_sampleid = st.checkbox("Include optional SampleID candidate matching", value=False)
        st.caption(
            "Experimental matching needs calibration. Scores do not establish accurate sample timestamps."
        )
'''

SIDEBAR_REPLACEMENT = '''    with st.expander("Demo and advanced options"):
        synthetic = st.checkbox("Use a clearly labeled synthetic demonstration", key="demo")
        use_sampleid = st.checkbox(
            "Use Sony SampleID + guarded DTW timing analysis",
            value=True,
            help=(
                "Sony proposes paired regions in the original and new track. "
                "Direct waveform matching or guarded DTW then refines those timings."
            ),
        )
        detected_checkpoint, checkpoint_note = resolve_sampleid_checkpoint(
            allow_download=False
        )
        sampleid_checkpoint_override = st.text_input(
            "Sony checkpoint path · optional local override",
            value=detected_checkpoint or "",
            placeholder=r"C:\\Users\\you\\.lastrada_sampleid\\sampleid-best.ckpt",
            disabled=not use_sampleid,
        )
        if use_sampleid and detected_checkpoint:
            st.caption("Sony checkpoint found. SampleID will run before DTW.")
        elif use_sampleid and _runtime_setting("LASTRADA_SAMPLEID_CHECKPOINT_URL"):
            st.caption(
                "The server will download and cache the configured Sony checkpoint "
                "when analysis starts."
            )
        elif use_sampleid:
            st.caption(checkpoint_note)
        st.caption(
            "Automated matches remain candidate evidence and require listening review."
        )
'''

CHECKPOINT_ANCHOR = '''                checkpoint = os.getenv("LASTRADA_SAMPLEID_CHECKPOINT") or (
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
'''

CHECKPOINT_REPLACEMENT = '''                checkpoint, checkpoint_note = resolve_sampleid_checkpoint(
                    sampleid_checkpoint_override,
                    allow_download=use_sampleid,
                )
                if use_sampleid and "master" in intake.analysis_scope:
                    if checkpoint is None:
                        adapter_error = checkpoint_note
                        st.warning(
                            "Sony + DTW could not start for this run. "
                            + str(adapter_error)
                            + " PF-NMF will remain available as fallback context."
                        )
                    else:
                        try:
                            adapter = SampleIDAdapter(
                                checkpoint,
                                os.getenv("LASTRADA_CACHE_DIR", "cache") + "/sampleid",
                                batch_size=settings.sampleid_batch_size,
                            )
                        except SampleIDUnavailable as exc:
                            adapter_error = str(exc)
                            st.warning(
                                "Sony + DTW could not start for this run: "
                                + adapter_error
                                + ". PF-NMF will remain available as fallback context."
                            )
'''

MASTER_ANCHOR = '''                    if rights == "master":
                        seconds, percent = (
'''

MASTER_REPLACEMENT = '''                    if rights == "master":
                        automated_occurrences = list(path.locate.sample_occurrences)
                        sampleid_method = result.methods.get("sampleid", "")
                        alignment_method = result.methods.get("alignment", "")
                        if automated_occurrences:
                            method_labels = {
                                "sampleid_rank_sequence": "Sony paired candidate",
                                "dtw_chroma_local_alignment": "DTW-refined Sony candidate",
                                "direct_waveform_local_alignment": "direct waveform refinement",
                                "sampleid_chunk_cosine": "coarse Sony candidate",
                            }
                            source_names = [item.source for item in automated_occurrences]
                            summary = ", ".join(
                                f"{method_labels.get(name, name)}: {source_names.count(name)}"
                                for name in dict.fromkeys(source_names)
                            )
                            st.success(
                                "Sony paired candidate regions in the original and new "
                                "track. Guarded DTW was applied automatically where a "
                                "plausible refinement passed the timing guardrails."
                            )
                            st.caption(
                                "Automated timing evidence · "
                                + summary
                                + (f" · pipeline: {alignment_method}" if alignment_method else "")
                            )
                            with st.expander("Review automated timing provenance"):
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "method": method_labels.get(item.source, item.source),
                                                "original_start": item.original_start,
                                                "original_end": item.original_end,
                                                "new_start": item.new_start,
                                                "new_end": item.new_end,
                                            }
                                            for item in automated_occurrences
                                        ]
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        elif sampleid_method.startswith("unavailable"):
                            reason = sampleid_method.removeprefix("unavailable:").strip()
                            st.warning(
                                "Sony + DTW did not run. The moments below are PF-NMF "
                                "fallback estimates and may not include original-track "
                                "timestamps. Reason: "
                                + (reason or "unknown SampleID setup error")
                            )
                        elif sampleid_method == "local_cpu_chunk_embeddings":
                            st.info(
                                "Sony completed but did not return a coherent paired "
                                "candidate under the current rank and timing rules. PF-NMF "
                                "is shown below as broader context."
                            )
                        else:
                            st.info(
                                "Sony + DTW were not used for this run. PF-NMF is shown "
                                "below as broader spectral context."
                            )

                        seconds, percent = (
'''

CHART_CAPTION_ANCHOR = '''                        st.caption(
                            "PF-NMF highlights similar sound in your track; it does not tell us which passage of the original was used. Source times appear only when available from matching or your review. Each track’s colour scale is relative to its own peak."
                        )
'''

CHART_CAPTION_REPLACEMENT = '''                        st.caption(
                            "PF-NMF provides broad spectral-overlap context. When Sony "
                            "finds paired source/new-track candidates, direct waveform "
                            "alignment or guarded DTW supplies the original-track times "
                            "shown in the review table. Each track’s colour scale is "
                            "relative to its own peak."
                        )
'''

SEEDS_ANCHOR = '''                    seeds = st.session_state[key + "seed_intervals"]
                    agreement = st.radio(
'''

SEEDS_REPLACEMENT = '''                    seeds = st.session_state[key + "seed_intervals"]
                    if rights == "master" and seeds:
                        paired = [
                            item
                            for item in seeds
                            if item.original_start is not None
                            and item.original_end is not None
                        ]
                        if paired:
                            st.info(
                                f"{len(paired)} paired original ↔ new-track timing "
                                "suggestion(s) came from the Sony/refinement pipeline. "
                                "Listen before accepting or editing them."
                            )
                            with st.expander("Timing methods"):
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "method": item.source,
                                                "original_start": item.original_start,
                                                "original_end": item.original_end,
                                                "new_start": item.new_start,
                                                "new_end": item.new_end,
                                            }
                                            for item in paired
                                        ]
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    agreement = st.radio(
'''


def replace_once(text: str, label: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not apply {label}: anchor not found")
    return text.replace(old, new, 1)


def main() -> None:
    text = APP.read_text(encoding="utf-8")
    if MARKER not in text:
        replacements = [
            ("checkpoint helpers", SETTINGS_ANCHOR, SETTINGS_REPLACEMENT),
            ("default sidebar analysis", SIDEBAR_ANCHOR, SIDEBAR_REPLACEMENT),
            ("checkpoint resolution", CHECKPOINT_ANCHOR, CHECKPOINT_REPLACEMENT),
            ("Step 3 analysis evidence", MASTER_ANCHOR, MASTER_REPLACEMENT),
            ("spectral caption", CHART_CAPTION_ANCHOR, CHART_CAPTION_REPLACEMENT),
            ("review timing provenance", SEEDS_ANCHOR, SEEDS_REPLACEMENT),
        ]
        for label, old, new in replacements:
            text = replace_once(text, label, old, new)
        compile(text, str(APP), "exec")
        APP.write_text(text, encoding="utf-8")

    if DTW_PAGE.exists():
        DTW_PAGE.unlink()

    print("Main Lastrada flow now defaults to Sony SampleID followed by guarded DTW.")


if __name__ == "__main__":
    main()
