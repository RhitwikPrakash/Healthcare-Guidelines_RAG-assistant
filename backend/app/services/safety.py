from __future__ import annotations

import re


SAFETY_NOTE = (
    "Educational document assistant only. It does not diagnose, prescribe, or replace a qualified clinician. "
    "Verify decisions against the original guideline and local clinical policy."
)

_EMERGENCY_RE = re.compile(
    r"\b(?:chest pain|difficulty breathing|cannot breathe|severe bleeding|unconscious|stroke symptoms|suicid|overdose|anaphylaxis|seizure|medical emergency)\b",
    re.IGNORECASE,
)


def emergency_notice(question: str) -> str | None:
    if _EMERGENCY_RE.search(question):
        return (
            "Your question may describe an urgent situation. Seek immediate local emergency medical help; "
            "do not rely on this document assistant for emergency triage."
        )
    return None


def safe_history(history: list[dict[str, str]], max_turns: int = 4, max_chars: int = 3500) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    used = 0
    for item in reversed(history[-max_turns * 2 :]):
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = content[:remaining]
        cleaned.append({"role": role, "content": content})
        used += len(content)
    return list(reversed(cleaned))
