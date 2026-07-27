from typing import Any, Literal
from pydantic import BaseModel, Field

class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    display_name: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)

class UserView(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserView


class ConversationCreate(BaseModel):
    title: str = Field(default="New research chat", min_length=1, max_length=120)

class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)

class ConversationView(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageView(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class QueryRequest(BaseModel):
    session_id: str = Field(min_length=4, max_length=120)
    request_id: str = Field(min_length=16, max_length=80, pattern=r"^[A-Za-z0-9-]+$")
    question: str = Field(min_length=2, max_length=6000)
    mode: Literal["auto", "deep", "fast"] = "auto"
    # Retained for compatibility. The server never trusts client-supplied history.
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)


class JobCreated(BaseModel):
    job_id: str
    status: str = "queued"
    request_id: str | None = None
    conversation_id: str | None = None
    user_message_id: str | None = None
    question_hash: str | None = None
    document_set_hash: str | None = None


class DocumentInfo(BaseModel):
    file_name: str
    pages: int
    chunks: int
    size_bytes: int
    sha256: str


class JobView(BaseModel):
    job_id: str
    status: str
    kind: str
    progress: float
    phase: str
    detail: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    request_id: str | None = None
    conversation_id: str | None = None
    user_message_id: str | None = None
    question_hash: str | None = None
    document_set_hash: str | None = None