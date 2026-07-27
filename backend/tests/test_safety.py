from app.services.safety import emergency_notice, safe_history


def test_emergency_notice():
    assert emergency_notice("I have severe chest pain and cannot breathe")
    assert emergency_notice("What does the guideline say about screening?") is None


def test_history_is_bounded():
    history = [{"role": "user", "content": "x" * 2000} for _ in range(10)]
    cleaned = safe_history(history, max_turns=2, max_chars=1000)
    assert sum(len(item["content"]) for item in cleaned) <= 1000
