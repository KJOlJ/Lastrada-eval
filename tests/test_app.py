from pathlib import Path
from streamlit.testing.v1 import AppTest


def by_label(elements, label):
    return next(e for e in elements if e.label == label)


def test_streamlit_synthetic_workflow_and_review_export():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert not app.exception
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.button, "Start analysis").click().run()
    assert not app.exception
    assert app.session_state["analysis"].synthetic
    assert app.session_state["analysis"].master.locate.sample_detected is None
    assert app.session_state["analysis"].publishing.locate.detection_status == "not_verified"
    by_label(app.text_input, "Reviewer name").set_value("Synthetic reviewer")
    by_label(app.text_area, "Editable Nature of Use").set_value("Synthetic review draft")
    by_label(app.button, "Save corrections").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "edited"
    assert (
        app.session_state["analysis"].review.corrected["nature_of_use"]["text"]
        == "Synthetic review draft"
    )
    assert app.session_state["analysis"].nature_of_use.text != "Synthetic review draft"
    by_label(app.button, "Approve reviewed draft").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "edited"
    assert any("resolve uncertain" in e.value for e in app.error)


def test_intake_requires_paired_uploads_and_starts_without_results():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert by_label(app.radio, "Requested analysis").value == "Both"
    assert "analysis" not in app.session_state
    by_label(app.button, "Start analysis").click().run()
    assert not app.exception
    assert any("Add both the original and new recordings" in e.value for e in app.error)
    assert "analysis" not in app.session_state


def test_demo_requires_metadata_and_respects_master_scope():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.text_input, "New artist").set_value("   ")
    by_label(app.button, "Start analysis").click().run()
    assert any("Enter a title and artist" in e.value for e in app.error)
    by_label(app.text_input, "New artist").set_value("Generated tones")
    by_label(app.radio, "Requested analysis").set_value("Master")
    by_label(app.button, "Start analysis").click().run()
    assert not app.exception
    assert app.session_state["analysis"].input.analysis_scope == ["master"]
    assert app.session_state["analysis"].publishing.locate.detection_status == "not_requested"


def test_next_back_flow_saves_draft_and_preserves_edits():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert by_label(app.button, "Next: Locate and measure →").disabled
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.button, "Start analysis").click().run()
    by_label(app.button, "Next: Locate and measure →").click().run()
    assert app.session_state["workflow_tab"] == "2 · Locate and measure"
    by_label(app.button, "Next: Classify and describe →").click().run()
    assert app.session_state["workflow_tab"] == "3 · Classify and describe"
    by_label(app.text_area, "Editable Nature of Use").set_value("Unsaved synthetic draft")
    by_label(app.text_input, "Reviewer name").set_value("Reviewer")
    by_label(app.button, "Next: Review and export →").click().run()
    assert not app.exception
    assert app.session_state["workflow_tab"] == "4 · Review and export"
    assert app.session_state["analysis"].review.status == "edited"
    assert (
        app.session_state["analysis"].review.corrected["nature_of_use"]["text"]
        == "Unsaved synthetic draft"
    )
    by_label(app.button, "← Back: Classify and describe").click().run()
    assert by_label(app.text_area, "Editable Nature of Use").value == "Unsaved synthetic draft"
    assert by_label(app.text_input, "Reviewer name").value == "Reviewer"
    by_label(app.button, "← Back: Locate and measure").click().run()
    by_label(app.button, "← Back: Intake").click().run()
    assert app.session_state["workflow_tab"] == "1 · Intake"
    assert app.session_state["analysis"].synthetic


def test_manual_step_cannot_bypass_intake():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.session_state["workflow_tab"] = "4 · Review and export"
    app.run()
    # Defensive guard runs before the tab widget as well as on manual tab changes.
    assert app.session_state["workflow_tab"] == "1 · Intake"


def test_next_does_not_revoke_unchanged_approval():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.button, "Start analysis").click().run()
    for element in app.selectbox:
        if element.label == "Reuse finding":
            element.set_value("not_detected")
    for element in app.checkbox:
        if element.label == "I have checked this path’s findings and classification":
            element.check()
    by_label(app.text_input, "Reviewer name").set_value("Reviewer")
    by_label(app.text_area, "Editable Nature of Use").set_value("Reviewed synthetic finding.")
    by_label(app.button, "Approve reviewed draft").click().run()
    assert app.session_state["analysis"].review.status == "approved"
    by_label(app.button, "Next: Review and export →").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "approved"
    by_label(app.button, "← Back: Classify and describe").click().run()
    by_label(app.text_area, "Editable Nature of Use").set_value("Changed synthetic finding.")
    by_label(app.button, "Next: Review and export →").click().run()
    assert app.session_state["analysis"].review.status == "edited"
