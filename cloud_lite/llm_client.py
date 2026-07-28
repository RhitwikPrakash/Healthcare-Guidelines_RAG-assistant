import os
from typing import Tuple, Optional

import requests


def get_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return None


def call_gemini(prompt: str, max_tokens: int = 1800) -> Tuple[str, str]:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

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
        },
    }

    response = requests.post(url, json=payload, timeout=90)

    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error: {response.status_code} {response.text[:300]}")

    data = response.json()
    candidates = data.get("candidates", [])

    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts).strip()

    if not text:
        raise RuntimeError("Gemini returned empty text.")

    return text, model


def call_groq(prompt: str, max_tokens: int = 1800) -> Tuple[str, str]:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

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
                "content": "You are a careful healthcare RAG assistant. Use only the provided evidence.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=90)

    if response.status_code >= 400:
        raise RuntimeError(f"Groq API error: {response.status_code} {response.text[:300]}")

    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()

    if not text:
        raise RuntimeError("Groq returned empty text.")

    return text, model


def generate_answer(prompt: str, max_tokens: int = 1800) -> Tuple[str, str, str]:
    gemini_error = None

    try:
        text, model = call_gemini(prompt, max_tokens=max_tokens)
        return text, "Gemini", model
    except Exception as error:
        gemini_error = str(error)

    try:
        text, model = call_groq(prompt, max_tokens=max_tokens)
        return text, "Groq fallback", model
    except Exception as groq_error:
        raise RuntimeError(
            "Both Gemini and Groq failed.\n\n"
            f"Gemini error: {gemini_error}\n\n"
            f"Groq error: {groq_error}"
        )