"""Keep the cached PF-NMF preview when Sony cannot start."""

from pathlib import Path


app = Path(__file__).resolve().parents[1] / "app.py"
text = app.read_text(encoding="utf-8")

result_anchor = '''                with st.spinner("Measuring audio locally…"):
                    result = (
                        reuse_preview(st.session_state.preview_analysis, intake)
                        if reusable and not use_sampleid
                        else analyze_pair(
                            intake,
                            original,
                            new,
                            adapter=adapter,
                            synthetic=synthetic,
                            settings=settings,
                        )
                    )
'''
result_replacement = '''                reused_preview = reusable and (not use_sampleid or adapter is None)
                with st.spinner("Measuring audio locally…"):
                    result = (
                        reuse_preview(st.session_state.preview_analysis, intake)
                        if reused_preview
                        else analyze_pair(
                            intake,
                            original,
                            new,
                            adapter=adapter,
                            synthetic=synthetic,
                            settings=settings,
                        )
                    )
'''

timing_anchor = '''                if reusable and not use_sampleid:
                    result.timings_seconds["intake_attachment"] = attachment_seconds
                    result.timings_seconds["total"] += attachment_seconds
                else:
                    result.timings_seconds["total"] = attachment_seconds
'''
timing_replacement = '''                if reused_preview:
                    result.timings_seconds["intake_attachment"] = attachment_seconds
                    result.timings_seconds["total"] += attachment_seconds
                else:
                    result.timings_seconds["total"] = attachment_seconds
'''

if result_anchor not in text or timing_anchor not in text:
    raise RuntimeError("Preview reuse anchors were not found")
text = text.replace(result_anchor, result_replacement, 1)
text = text.replace(timing_anchor, timing_replacement, 1)
compile(text, str(app), "exec")
app.write_text(text, encoding="utf-8")
print("PF-NMF preview reuse preserved when Sony is unavailable")
