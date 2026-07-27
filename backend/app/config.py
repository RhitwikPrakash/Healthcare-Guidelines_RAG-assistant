from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Healthcare Guidelines RAG Assistant"
    model_profile: Literal["medical_quality", "balanced", "ultra_light"] = "medical_quality"

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "medgemma1.5:4b"
    llm_fallback_model: str = "qwen3.5:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"

    embedding_backend: Literal["sentence_transformers", "ollama"] = "sentence_transformers"
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    enable_reranker: bool = True

    chunk_size: int = Field(default=1100, ge=400, le=2500)
    chunk_overlap: int = Field(default=160, ge=0, le=500)
    dense_k: int = Field(default=24, ge=4, le=100)
    lexical_k: int = Field(default=24, ge=4, le=100)
    rerank_k: int = Field(default=14, ge=4, le=40)
    final_k: int = Field(default=8, ge=2, le=20)
    max_context_chars: int = Field(default=18000, ge=4000, le=60000)
    max_upload_mb: int = Field(default=80, ge=5, le=300)
    max_files_per_session: int = Field(default=12, ge=1, le=50)

    planner_mode: Literal["auto", "always", "off"] = "auto"
    ollama_timeout_seconds: int = Field(default=300, ge=30, le=900)
    temperature: float = Field(default=0.12, ge=0, le=1)
    max_output_tokens: int = Field(default=1600, ge=256, le=4096)

    # Authenticated history settings. If JWT_SECRET is omitted, a strong secret
    # is generated once in data/.jwt_secret and reused on later starts.
    auth_enabled: bool = True
    jwt_secret: str = ""
    jwt_issuer: str = "healthcare-guidelines-rag"
    jwt_expire_days: int = Field(default=7, ge=1, le=90)
    chat_retention_months: int = Field(default=5, ge=1, le=24)
    chat_cleanup_interval_hours: int = Field(default=24, ge=1, le=168)
    chat_encryption_key: str = ""

    data_dir: Path = Path("data")
    hf_home: Path = Path("models/huggingface")
    log_level: str = "INFO"
    frontend_origin: str = "http://localhost:8501"
    testing: bool = False

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def chat_database_path(self) -> Path:
        return self.data_dir / "chat_history.db"

    @property
    def jwt_secret_path(self) -> Path:
        return self.data_dir / ".jwt_secret"

    @property
    def chat_encryption_key_path(self) -> Path:
        return self.data_dir / ".history_key"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.chroma_dir, self.sessions_dir, self.hf_home):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings