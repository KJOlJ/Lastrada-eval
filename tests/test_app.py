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
    by_label(app.button, "Analyze pair").click().run()
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
