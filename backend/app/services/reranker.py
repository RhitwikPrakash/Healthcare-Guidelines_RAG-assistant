from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.config import get_settings


class Reranker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _load(self, model_name: str) -> tuple[str, Any]:
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            if model_name == "ncbi/MedCPT-Cross-Encoder":
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.hf_home),
                    trust_remote_code=False,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.hf_home),
                    trust_remote_code=False,
                )
                model.to("cpu")
                model.eval()
                loaded = ("medcpt", (tokenizer, model))
            else:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(
                    model_name,
                    max_length=512,
                    device="cpu",
                    trust_remote_code=False,
                )
                loaded = ("sentence_transformers", model)
            self._models[model_name] = loaded
            return loaded

    @staticmethod
    def _predict_medcpt(query: str, candidates: list[dict[str, Any]], payload: Any) -> np.ndarray:
        import torch

        tokenizer, model = payload
        scores: list[float] = []
        batch_size = 8
        pairs = [[query, candidate["text"]] for candidate in candidates]
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                encoded = tokenizer(
                    pairs[start : start + batch_size],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                logits = model(**encoded).logits.squeeze(dim=-1)
                scores.extend(float(value) for value in logits.detach().cpu().reshape(-1).tolist())
        return np.asarray(scores, dtype=float)

    def _predict(self, model_name: str, query: str, candidates: list[dict[str, Any]]) -> np.ndarray:
        model_type, payload = self._load(model_name)
        if model_type == "medcpt":
            return self._predict_medcpt(query, candidates, payload)
        pairs = [(query, candidate["text"]) for candidate in candidates]
        return np.asarray(payload.predict(pairs, batch_size=8, show_progress_bar=False), dtype=float)

    @staticmethod
    def _apply_scores(candidates: list[dict[str, Any]], raw: np.ndarray, model_name: str) -> list[dict[str, Any]]:
        minimum = float(raw.min()) if raw.size else 0.0
        maximum = float(raw.max()) if raw.size else 1.0
        denominator = maximum - minimum or 1.0
        for candidate, score in zip(candidates, raw.tolist()):
            candidate["rerank_score"] = (float(score) - minimum) / denominator
            candidate["reranker_model"] = model_name
            candidate["final_score"] = 0.76 * candidate["rerank_score"] + 0.24 * candidate.get("fusion_score", 0.0)
        return sorted(candidates, key=lambda item: item.get("final_score", 0.0), reverse=True)

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not self.settings.enable_reranker or self.settings.testing:
            return sorted(candidates, key=lambda item: item.get("fusion_score", 0.0), reverse=True)

        errors: list[str] = []
        model_names = [self.settings.reranker_model]
        fallback = self.settings.reranker_fallback_model
        if fallback and fallback not in model_names:
            model_names.append(fallback)
        for model_name in model_names:
            try:
                raw = self._predict(model_name, query, candidates)
                if raw.size != len(candidates):
                    raise RuntimeError("Reranker returned an unexpected number of scores")
                ranked = self._apply_scores(candidates, raw, model_name)
                if errors:
                    for candidate in ranked:
                        candidate["reranker_warning"] = " | ".join(errors)
                return ranked
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_name}: {exc}")

        for candidate in candidates:
            candidate["reranker_warning"] = " | ".join(errors)
        return sorted(candidates, key=lambda item: item.get("fusion_score", 0.0), reverse=True)


reranker = Reranker()
