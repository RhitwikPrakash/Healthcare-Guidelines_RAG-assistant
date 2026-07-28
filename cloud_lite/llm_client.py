from __future__ import annotations

import os
from typing import Optional, Tuple

import requests


RATE_LIMIT_MESSAGE = "Please wait about 40–60 seconds and try again."


class ProviderRateLimitError(RuntimeError):
    """Raised when the available provider has temporarily exhausted its free quota."""


class ProviderUnavailableError(RuntimeError):
    """Raised when neither configured provider can complete the request."""


class ProviderIncompleteError(ProviderUnavailableError):
    """Raised when a provider stops before producing a complete answer."""


def get_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return str(value).strip()

    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value is not None:
                return str(value).strip()
    except Exception:
        pass

    return None


def get_gemini_api_key() -> Optional[str]:
    """Accept either Gemini key name used by the project."""
    return get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")


def _looks_complete(text: str) -> bool:
    clean = (text or "").strip()
    if len(clean) < 40:
        return False

    bad_endings = (
        "including",
        "such as",
        "for example",
        "and",
        "or",
        "with",
        "because",
        "therefore",
        "including:",
        "such as:",
    )
    return clean[-1] in ".!?)]}" and not clean.casefold().endswith(bad_endings)


def call_gemini(prompt: str, max_tokens: int = 1800) -> Tuple[str, str]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ProviderUnavailableError("Gemini API key is missing.")

    model = get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.9,
            "maxOutputTokens": max_tokens,
            # RAG answers need concise synthesis, not a large hidden thinking budget.
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=90)
    except requests.RequestException as error:
        raise ProviderUnavailableError("Gemini is temporarily unavailable.") from error

    if response.status_code == 429:
        raise ProviderRateLimitError(RATE_LIMIT_MESSAGE)
    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"Gemini request failed with status {response.status_code}."
        )

    try:
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderUnavailableError("Gemini returned no answer candidate.")

        candidate = candidates[0]
        finish_reason = str(candidate.get("finishReason") or "").upper()
        parts = candidate.get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts).strip()
    except ProviderUnavailableError:
        raise
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderUnavailableError("Gemini returned an invalid response.") from error

    if finish_reason == "MAX_TOKENS":
        raise ProviderIncompleteError("Gemini stopped at its output-token limit.")
    if finish_reason and finish_reason != "STOP":
        raise ProviderUnavailableError(
            f"Gemini stopped before completion ({finish_reason})."
        )
    if not text:
        raise ProviderUnavailableError("Gemini returned an empty response.")
    if not _looks_complete(text):
        raise ProviderIncompleteError("Gemini returned an incomplete answer.")

    return text, model


def call_groq(prompt: str, max_tokens: int = 900) -> Tuple[str, str]:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise ProviderUnavailableError("Groq API key is missing.")

    model = get_secret("GROQ_MODEL") or "llama-3.1-8b-instant"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful healthcare-document RAG assistant. "
                    "Use only the supplied evidence, answer concisely, finish every "
                    "sentence, and do not add inline citation labels."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as error:
        raise ProviderUnavailableError("Groq is temporarily unavailable.") from error

    if response.status_code == 429:
        raise ProviderRateLimitError(RATE_LIMIT_MESSAGE)
    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"Groq request failed with status {response.status_code}."
        )

    try:
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"].strip()
        finish_reason = str(choice.get("finish_reason") or "").lower()
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderUnavailableError("Groq returned an invalid response.") from error

    if finish_reason == "length":
        raise ProviderIncompleteError("Groq stopped at its output-token limit.")
    if finish_reason and finish_reason != "stop":
        raise ProviderUnavailableError(
            f"Groq stopped before completion ({finish_reason})."
        )
    if not text:
        raise ProviderUnavailableError("Groq returned an empty response.")
    if not _looks_complete(text):
        raise ProviderIncompleteError("Groq returned an incomplete answer.")

    return text, model


def generate_answer(prompt: str) -> Tuple[str, str, str]:
    """Use Gemini first, then Groq; never return a visibly truncated answer."""
    gemini_error: Optional[Exception] = None

    try:
        text, model = call_gemini(prompt, max_tokens=1800)
        return text, "Gemini", model
    except Exception as error:  # fallback is intentional
        gemini_error = error

    try:
        text, model = call_groq(prompt, max_tokens=900)
        return text, "Groq fallback", model
    except ProviderRateLimitError:
        raise
    except Exception as groq_error:
        if isinstance(gemini_error, ProviderRateLimitError):
            raise ProviderRateLimitError(RATE_LIMIT_MESSAGE) from groq_error
        raise ProviderUnavailableError(
            "The AI providers could not complete the answer. Please try again shortly."
        ) from groq_error