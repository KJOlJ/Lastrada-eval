"""Small presentation helpers for the upload-first analysis interface."""

from pathlib import PurePath

from lastrada.ingestion.filenames import filename_credits
from lastrada.reporting.document import timestamp
from lastrada.schemas.models import Intake, Work


def track_work(filename, fallback):
    credits = filename_credits(filename) if filename else None
    title, artist = credits or (PurePath(filename).stem if filename else fallback, "Unknown")
    return Work(title=title.strip() or fallback, artist=artist, audio_file=filename or None)


def quick_intake(original_name, new_name, synthetic=False):
    return Intake(
        case_id="synthetic-demo" if synthetic else "audio-comparison",
        original_work=track_work(original_name, "Original track"),
        new_work=track_work(new_name, "New track"),
        analysis_scope=["master"],
    )


def match_rows(analysis):
    return [
        {
            "Match": index,
            "Original": f"{timestamp(item.original_start)} – {timestamp(item.original_end)}",
            "New track": f"{timestamp(item.new_start)} – {timestamp(item.new_end)}",
            "Status": "Candidate · listen to check",
        }
        for index, item in enumerate(analysis.master.locate.sample_occurrences, 1)
    ]


def quick_draft(analysis):
    """Describe candidate evidence without converting it to reviewed reuse."""
    source, new = analysis.input.original_work, analysis.input.new_work
    lines = [
        "NATURE OF USE — PRELIMINARY DRAFT",
        "Not reviewed. Automated candidates do not establish reuse or clearance.",
        "",
        f"Original: {source.title} — {source.artist}",
        f"New recording: {new.title} — {new.artist}",
        "",
    ]
    candidates = analysis.master.locate.sample_occurrences
    if candidates:
        lines.append(
            f"Automated recording analysis identified {len(candidates)} candidate matching "
            "region(s). Listening and expert review are needed to confirm any reuse."
        )
        for row in match_rows(analysis):
            lines.append(
                f"Candidate {row['Match']}: original {row['Original']}; "
                f"new recording {row['New track']}."
            )
    elif analysis.methods.get("sampleid") == "local_cpu_chunk_embeddings":
        lines.append(
            "Sony analysis returned no paired candidate regions under the current settings. "
            "This does not establish absence of reuse."
        )
    else:
        lines.append(
            "Sony analysis did not complete. No Sony-based reuse description is available; "
            "PF-NMF spectral overlap alone cannot establish a recording sample."
        )
    lines += [
        "",
        "The reused instrument or vocal, looping, song sections, transformations, and "
        "musical role have not been established by this analysis.",
        "Replayed melody and lyrics (publishing) have not been analyzed in this workflow.",
    ]
    if analysis.synthetic:
        lines += ["", "SYNTHETIC DEMONSTRATION — generated audio, not a real music case."]
    return "\n".join(lines)
