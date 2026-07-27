from __future__ import annotations

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.embedding_backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        print(f"Caching embedding model: {settings.embedding_model}")
        SentenceTransformer(
            settings.embedding_model,
            cache_folder=str(settings.hf_home),
            device="cpu",
            trust_remote_code=False,
        )

    if settings.enable_reranker:
        if settings.reranker_model == "ncbi/MedCPT-Cross-Encoder":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            print(f"Caching medical reranker: {settings.reranker_model}")
            AutoTokenizer.from_pretrained(
                settings.reranker_model,
                cache_dir=str(settings.hf_home),
                trust_remote_code=False,
            )
            AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model,
                cache_dir=str(settings.hf_home),
                trust_remote_code=False,
            )
        else:
            from sentence_transformers import CrossEncoder

            print(f"Caching reranker: {settings.reranker_model}")
            CrossEncoder(
                settings.reranker_model,
                device="cpu",
                trust_remote_code=False,
            )
    print("Medical retrieval models are cached.")


if __name__ == "__main__":
    main()
