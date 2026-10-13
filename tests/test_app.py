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
    by_label(app.button, "Find suggested moments").click().run()
    assert not app.exception
    assert app.session_state["analysis"].synthetic
    assert app.session_state["analysis"].master.locate.sample_detected is None
    assert app.session_state["analysis"].publishing.locate.detection_status == "not_verified"
    by_label(app.text_input, "Your name / reviewer (required to mark as reviewed)").set_value(
        "Synthetic reviewer"
    )
    by_label(app.text_area, "Edit your Nature of Use draft").set_value("Synthetic review draft")
    by_label(app.button, "Save draft").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "edited"
    assert (
        app.session_state["analysis"].review.corrected["nature_of_use"]["text"]
        == "Synthetic review draft"
    )
    assert app.session_state["analysis"].nature_of_use.text != "Synthetic review draft"
    by_label(app.button, "Mark document as reviewed").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "edited"
    assert any("resolve uncertain" in e.value for e in app.error)


def test_intake_requires_paired_uploads_and_starts_without_results():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert by_label(app.radio, "Type of reuse").value == "Both"
    assert "analysis" not in app.session_state
    by_label(app.button, "Find suggested moments").click().run()
    assert not app.exception
    assert any("Add both the original and new recordings" in e.value for e in app.error)
    assert "analysis" not in app.session_state


def test_demo_requires_metadata_and_respects_master_scope():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.text_input, "New artist").set_value("   ")
    by_label(app.button, "Find suggested moments").click().run()
    assert any("Enter a title and artist" in e.value for e in app.error)
    by_label(app.text_input, "New artist").set_value("Generated tones")
    by_label(app.radio, "Type of reuse").set_value("Master")
    by_label(app.button, "Find suggested moments").click().run()
    assert not app.exception
    assert app.session_state["analysis"].input.analysis_scope == ["master"]
    assert app.session_state["analysis"].publishing.locate.detection_status == "not_requested"


def test_next_back_flow_saves_draft_and_preserves_edits():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert by_label(app.button, "Next: Listen & explore →").disabled
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.button, "Find suggested moments").click().run()
    by_label(app.button, "Next: Listen & explore →").click().run()
    assert app.session_state["workflow_tab"] == "2 · Listen & explore"
    by_label(app.button, "Next: Describe your sample →").click().run()
    assert app.session_state["workflow_tab"] == "3 · Describe your sample"
    by_label(app.text_area, "Edit your Nature of Use draft").set_value("Unsaved synthetic draft")
    by_label(app.text_input, "Your name / reviewer (required to mark as reviewed)").set_value(
        "Reviewer"
    )
    by_label(app.button, "Next: Your Nature of Use →").click().run()
    assert not app.exception
    assert app.session_state["workflow_tab"] == "4 · Your Nature of Use"
    assert app.session_state["analysis"].review.status == "edited"
    assert (
        app.session_state["analysis"].review.corrected["nature_of_use"]["text"]
        == "Unsaved synthetic draft"
    )
    by_label(app.button, "← Back: Describe your sample").click().run()
    assert (
        by_label(app.text_area, "Edit your Nature of Use draft").value == "Unsaved synthetic draft"
    )
    assert (
        by_label(app.text_input, "Your name / reviewer (required to mark as reviewed)").value
        == "Reviewer"
    )
    by_label(app.button, "← Back: Listen & explore").click().run()
    by_label(app.button, "← Back: Add your tracks").click().run()
    assert app.session_state["workflow_tab"] == "1 · Add your tracks"
    assert app.session_state["analysis"].synthetic


def test_manual_step_cannot_bypass_intake():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.session_state["workflow_tab"] = "4 · Your Nature of Use"
    app.run()
    # Defensive guard runs before the tab widget as well as on manual tab changes.
    assert app.session_state["workflow_tab"] == "1 · Add your tracks"


def test_next_does_not_revoke_unchanged_approval():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.button, "Find suggested moments").click().run()
    for element in app.selectbox:
        if element.label == "Have you checked this use?":
            element.set_value("not_detected")
    for element in app.checkbox:
        if element.label == "I’ve listened and checked these details":
            element.check()
    by_label(app.text_input, "Your name / reviewer (required to mark as reviewed)").set_value(
        "Reviewer"
    )
    by_label(app.text_area, "Edit your Nature of Use draft").set_value(
        "Reviewed synthetic finding."
    )
    by_label(app.button, "Mark document as reviewed").click().run()
    assert app.session_state["analysis"].review.status == "approved"
    by_label(app.button, "Next: Your Nature of Use →").click().run()
    assert not app.exception
    assert app.session_state["analysis"].review.status == "approved"
    by_label(app.button, "← Back: Describe your sample").click().run()
    by_label(app.text_area, "Edit your Nature of Use draft").set_value("Changed synthetic finding.")
    by_label(app.button, "Next: Your Nature of Use →").click().run()
    assert app.session_state["analysis"].review.status == "edited"


def test_prefilled_times_require_confirmation_and_loop_answers_fill_only_blanks():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    assert app.session_state["dark_mode"] is True
    by_label(app.toggle, "Dark mode").set_value(False).run()
    assert app.session_state["dark_mode"] is False
    by_label(app.checkbox, "Use a clearly labeled synthetic demonstration").check().run()
    by_label(app.radio, "Type of reuse").set_value("Master")
    by_label(app.button, "Find suggested moments").click().run()
    by_label(app.selectbox, "Have you checked this use?").set_value("verified").run()
    assert len(app.dataframe[0].value) > 0
    by_label(app.button, "Next: Your Nature of Use →").click().run()
    saved = app.session_state["analysis"].review.corrected["master"]["locate"]
    assert saved["detection_status"] == "candidate"
    assert saved["sample_detected"] is None
    assert saved["sample_occurrences"][0]["status"] == "estimated"
    by_label(app.button, "← Back: Describe your sample").click().run()
    question = next(r for r in app.radio if "repeating loop?" in r.label)
    question.set_value("Yes").run()
    by_label(app.button, "Use this answer to fill the details").click().run()
    assert (
        by_label(app.text_input, "Does it loop or repeat?").value
        == "A short loop repeated in the track"
    )
    by_label(app.text_input, "Does it loop or repeat?").set_value("My own loop description").run()
    by_label(app.button, "Use this answer to fill the details").click().run()
    assert by_label(app.text_input, "Does it loop or repeat?").value == "My own loop description"
    by_label(app.checkbox, "I’ve listened and confirmed these times").check().run()
    by_label(app.button, "Next: Your Nature of Use →").click().run()
    saved = app.session_state["analysis"].review.corrected["master"]["locate"]
    assert saved["detection_status"] == "verified"
    assert saved["sample_occurrences"][0]["source"] == "expert_review"
