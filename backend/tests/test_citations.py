from app.services.citations import build_evidence_context, validate_citations


def evidence():
    return [
        {"id": "a", "text": "Blood pressure should be monitored.", "source": "a.pdf", "page": 2, "section": "Monitoring"},
        {"id": "b", "text": "Follow-up intervals depend on control.", "source": "a.pdf", "page": 3, "section": "Follow-up"},
    ]


def test_context_assigns_stable_source_ids():
    context, selected = build_evidence_context(evidence(), 10000)
    assert "[S1]" in context and "[S2]" in context
    assert selected[0]["source_id"] == "S1"


def test_invalid_citations_are_removed():
    _, selected = build_evidence_context(evidence(), 10000)
    result = validate_citations("Monitoring is recommended. [S1] Unsupported. [S99]", selected)
    assert "[S99]" not in result["answer"]
    assert result["valid_citations"] == ["S1"]


def test_uncited_bullet_is_detected():
    _, selected = build_evidence_context(evidence(), 10000)
    result = validate_citations("- Blood pressure should be monitored regularly.\n- Follow-up depends on control. [S2]", selected)
    assert result["coverage"] < 1.0
    assert result["uncited_blocks"] == 1
