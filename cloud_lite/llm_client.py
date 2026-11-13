from __future__ import annotations

import os
from typing import Optional, Tuple

import requests


RATE_LIMIT_MESSAGE = (
    "The AI providers are rate-limited or their API credit/quota is exhausted. "
    "Please check your provider usage limits and retry shortly."
)


class ProviderRateLimitError(RuntimeError):
    """Raised when provider rate limits or quota prevent answering."""


class ProviderUnavailableError(RuntimeError):
    """Raised when the configured providers cannot complete the answer."""


def get_secret(name: str) -> Optional[str]:
    """Read secrets from environment variables or Streamlit Secrets."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass

    return None


def get_openai_api_key() -> Optional[str]:
    return get_secret("OPENAI_API_KEY")


def get_gemini_api_key() -> Optional[str]:
    # Both spellings are supported for existing Streamlit deployments.
    return get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")


def _parse_openai_text(data: dict) -> str:
    """Extract visible text from the OpenAI Responses API result."""
    pieces = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text"):
                pieces.append(part["text"])

    # The official Python SDK synthesizes output_text; REST may include it too.
    return "\n".join(pieces).strip() or str(data.get("output_text") or "").strip()


def call_openai(prompt: str, max_tokens: int = 1600) -> Tuple[str, str]:
    """Primary answer provider: OpenAI Responses API (no server-side storage)."""
    key = get_openai_api_key()
    if not key:
        raise ProviderUnavailableError("OpenAI API key is missing.")

    model = get_secret("OPENAI_MODEL") or "gpt-5.4-mini"
    payload = {
        "model": model,
        "instructions": (
            "You are a careful healthcare-guideline RAG assistant. Answer the "
            "current question using ONLY the supplied document evidence. "
            "Distinguish disease stage, patient group and treatment context; do not "
            "combine recommendations from unrelated sections. If evidence is "
            "insufficient, say so. Do not invent citations or add inline source labels. "
            "This is educational information, not medical advice."
        ),
        "input": prompt,
        "max_output_tokens": max_tokens,
        # Avoid retaining response objects in the provider's Responses store.
        "store": False,
    }
    # GPT-5.4 mini supports disabling reasoning for direct grounded answers.
    if model.startswith("gpt-5.4"):
        payload["reasoning"] = {"effort": "none"}

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 110),
        )
    except requests.RequestException as error:
        raise ProviderUnavailableError("OpenAI request failed due to a connection error.") from error

    if response.status_code in (402, 429):
        raise ProviderRateLimitError(
            "OpenAI API is rate-limited or has insufficient credit/quota."
        )
    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"OpenAI API request failed (HTTP {response.status_code})."
        )

    try:
        data = response.json()
        answer = _parse_openai_text(data)
        status = data.get("status")
    except (ValueError, AttributeError, TypeError) as error:
        raise ProviderUnavailableError("OpenAI returned an invalid response.") from error

    if status == "incomplete":
        raise ProviderUnavailableError("OpenAI returned an incomplete answer.")
    if not answer:
        raise ProviderUnavailableError("OpenAI returned an empty answer.")

    return answer, model


def call_gemini(prompt: str, max_tokens: int = 1600) -> Tuple[str, str]:
    """Fallback answer provider: Google Gemini."""
    key = get_gemini_api_key()
    if not key:
        raise ProviderUnavailableError("Gemini fallback API key is missing.")

    model = get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.9,
            "maxOutputTokens": max_tokens,
        },
    }

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 110),
        )
    except requests.RequestException as error:
        raise ProviderUnavailableError("Gemini fallback is temporarily unavailable.") from error

    if response.status_code == 429:
        raise ProviderRateLimitError("Gemini fallback is rate-limited.")
    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"Gemini fallback request failed (HTTP {response.status_code})."
        )

    try:
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates")
        candidate = candidates[0]
        text = "\n".join(
            p["text"] for p in candidate.get("content", {}).get("parts", [])
            if isinstance(p, dict) and p.get("text")
        ).strip()
        finish = candidate.get("finishReason")
    except (ValueError, AttributeError, KeyError, TypeError) as error:
        raise ProviderUnavailableError("Gemini fallback returned an invalid response.") from error

    if finish == "MAX_TOKENS":
        raise ProviderUnavailableError("Gemini fallback reached its output limit.")
    if not text:
        raise ProviderUnavailableError("Gemini fallback returned an empty answer.")
    return text, model


def generate_answer(prompt: str, max_tokens: int = 1600) -> Tuple[str, str, str]:
    """Return (answer, provider label, model); OpenAI first, Gemini on failure."""
    primary_error: Optional[Exception] = None
    try:
        text, model = call_openai(prompt, max_tokens=max_tokens)
        return text, "OpenAI", model
    except (ProviderUnavailableError, ProviderRateLimitError) as error:
        primary_error = error

    try:
        text, model = call_gemini(prompt, max_tokens=max_tokens)
        return text, "Gemini fallback", model
    except ProviderRateLimitError as fallback_error:
        if isinstance(primary_error, ProviderRateLimitError):
            raise ProviderRateLimitError(RATE_LIMIT_MESSAGE) from fallback_error
        raise ProviderUnavailableError(
            "OpenAI failed and Gemini fallback is rate-limited. "
            "Check the API key, credits and provider usage quotas."
        ) from fallback_error
    except ProviderUnavailableError as fallback_error:
        if isinstance(primary_error, ProviderRateLimitError):
            raise ProviderRateLimitError(
                "OpenAI is rate-limited/out of credit, and Gemini fallback "
                f"also failed: {fallback_error}"
            ) from fallback_error
        raise ProviderUnavailableError(
            f"Both providers failed. OpenAI: {primary_error}; Gemini: {fallback_error}"
        ) from fallback_error
