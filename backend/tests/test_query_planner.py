from app.services.query_planner import planner


def test_whole_document_question_uses_document_wide_strategy():
    plan = planner.plan("Summarize the whole guideline and all major recommendations", "auto")
    assert plan["whole_document"] is True
    assert plan["strategy"] == "document-wide"


def test_simple_question_stays_focused():
    plan = planner.plan("What is the target blood pressure?", "fast")
    assert plan["strategy"] == "focused"
