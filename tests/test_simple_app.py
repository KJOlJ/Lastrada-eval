from pathlib import Path

from streamlit.testing.v1 import AppTest

from lastrada.reporting.quick_review import quick_draft, quick_intake


APP = Path(__file__).resolve().parents[1] / "simple_app.py"


def by_label(elements, label):
    return next(item for item in elements if item.label == label)


def test_upload_first_requires_pair_without_metadata_form():
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert by_label(app.button, "Analyze tracks").disabled
    assert not app.text_input
    assert not app.radio
    assert "simple_result" not in app.session_state


def test_demo_analysis_optional_draft_and_reset(monkeypatch):
    from lastrada import pipeline

    calls = []
    original_analyze = pipeline.analyze_pair

    def counted_analysis(*args, **kwargs):
        calls.append(True)
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_pair", counted_analysis)
    app = AppTest.from_file(APP, default_timeout=30).run()
    by_label(app.checkbox, "Use synthetic demonstration audio").check().run()
    by_label(app.button, "Analyze tracks").click().run()
    assert not app.exception
    result = app.session_state["simple_result"]
    assert result.synthetic
    assert result.input.analysis_scope == ["master"]
    assert result.master.locate.sample_detected is None
    assert len(result.master.locate.sample_occurrences) == 2
    assert all(i.source == "synthetic_known_insertion" for i in result.master.locate.sample_occurrences)
    assert "simple_draft" not in app.session_state
    by_label(app.selectbox, "Listen to a match").set_value(1).run()
    assert len(calls) == 1
    by_label(app.button, "Create Nature of Use draft").click().run()
    assert not app.exception
    assert "PRELIMINARY DRAFT" in app.session_state["simple_draft"]
    assert "SYNTHETIC DEMONSTRATION" in app.session_state["simple_draft"]
    assert app.session_state["simple_result"].review.status == "pending"
    assert len(calls) == 1
    by_label(app.text_area, "Your draft · optional edits").set_value("My draft edit").run()
    assert app.session_state["simple_draft"] == "My draft edit"
    assert len(calls) == 1
    by_label(app.button, "New pair").click().run()
    assert not app.exception
    assert "simple_result" not in app.session_state
    assert "simple_draft" not in app.session_state
    assert by_label(app.button, "Analyze tracks").disabled


def test_generic_filenames_do_not_require_guessed_credits():
    intake = quick_intake("source.mp3", "mix.mp3")
    assert intake.original_work.title == "source"
    assert intake.original_work.artist == "Unknown"
    known = quick_intake("None of Ur Friends Business by Ginuwine.mp3", "mix.mp3")
    assert known.original_work.artist == "Ginuwine"


def test_draft_does_not_turn_fallback_into_matching_evidence(intake):
    from lastrada.schemas.models import Analysis

    result = Analysis(case_id=intake.case_id, input=intake)
    text = quick_draft(result)
    assert "Sony analysis did not complete" in text
    assert "PF-NMF spectral overlap alone cannot establish" in text
    assert result.review.status == "pending"
