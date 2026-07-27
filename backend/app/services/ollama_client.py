from __future__ import annotations
import re
import time
from typing import Any
import requests
from app.config import get_settings

_REASONING_PATTERNS = (
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<unused94>.*?<unused95>", re.IGNORECASE | re.DOTALL),
)

def _strip_private_reasoning(text: str) -> str:
    cleaned = text
    for pattern in _REASONING_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()

class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    def tags(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=8)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        try:
            payload = self.tags()
            models = [item.get("name", "") for item in payload.get("models", [])]
            return {"reachable": True, "models": models}
        except Exception as exc:  # noqa: BLE001
            return {"reachable": False, "models": [], "error": str(exc)}

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        selected_model = model or self.settings.llm_model
        fresh_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                raise ValueError("Each Ollama message must have a valid role and non-empty content")
            fresh_messages.append({"role": role, "content": content})

        # /api/chat is stateless when only `messages` are supplied. We deliberately
        # never send an Ollama `context` value or reuse a prior response payload.
        body: dict[str, Any] = {
            "model": selected_model,
            "messages": fresh_messages,
            "stream": False,
            "keep_alive": "20m",
            "options": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.settings.max_output_tokens,
                "num_ctx": 16384,
                "num_keep": 0,
                "repeat_penalty": 1.08,
            },
        }
        if json_mode:
            body["format"] = "json"
            # Structured verifier/planner calls need final JSON immediately.
            # Prevent thinking tokens from consuming the output allowance.
            body["think"] = False

        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_content = str((payload.get("message") or {}).get("content") or "")
        content = _strip_private_reasoning(raw_content)
        if not content:
            raise RuntimeError(f"Ollama model {selected_model} returned an empty response")
        return {
            "content": content,
            "model": payload.get("model", selected_model),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "eval_count": payload.get("eval_count"),
            "prompt_eval_count": payload.get("prompt_eval_count"),
        }

    def chat_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        preferred_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        first = preferred_model or self.settings.llm_model
        models = [first]
        if self.settings.llm_fallback_model and self.settings.llm_fallback_model not in models:
            models.append(self.settings.llm_fallback_model)
        errors: list[str] = []
        for model in models:
            try:
                result = self.chat(
                    messages,
                    model=model,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result["fallback_used"] = model != first
                return result
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: {exc}")
        raise RuntimeError("All Ollama models failed. " + " | ".join(errors))

ollama = OllamaClient()