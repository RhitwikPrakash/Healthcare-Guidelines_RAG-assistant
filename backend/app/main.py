from __future__ import annotations

import hashlib
import logging
import re
import threading
from difflib import SequenceMatcher
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.database import db
from app.schemas import (
    ConversationCreate,
    ConversationRename,
    ConversationView,
    JobCreated,
    JobView,
    LoginRequest,
    MessageView,
    QueryRequest,
    RegisterRequest,
    TokenResponse,
    UserView,
)
from app.services.answer_engine import QUERY_STEPS, answer_job
from app.services.auth_service import auth, get_current_user
from app.services.bm25_store import bm25_store
from app.services.chat_store import chat_store
from app.services.document_store import store
from app.services.history_cleanup import history_cleanup
from app.services.ingestion import UPLOAD_STEPS, ingest_job
from app.services.job_manager import jobs
from app.services.ollama_client import ollama
from app.services.vector_store import vector_store

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("healthcare-rag")
_query_lock = threading.RLock()

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
_CITATION_RE = re.compile(r"\[(S\d+)\]")
_REFERENTIAL_RE = re.compile(
    r"\b(?:it|its|that|those|this|these|they|them|their|there|above|earlier|previous|former|latter|same|"
    r"what about|how about|why so|explain further|elaborate|continue|compare with that)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about",
    "according",
    "answer",
    "asked",
    "based",
    "could",
    "document",
    "early",
    "explain",
    "give",
    "guide",
    "guideline",
    "important",
    "include",
    "level",
    "main",
    "page",
    "pages",
    "question",
    "relevant",
    "should",
    "show",
    "tell",
    "their",
    "there",
    "these",
    "this",
    "those",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
_FOCUS_GENERIC = {
    "cancer", "diagnosi", "early", "document", "evidence", "guideline", "health", "provide",
    "finding", "question", "answer", "information", "programme", "program", "system",
}
_NUMERIC_REQUEST_RE = re.compile(
    r"\b(?:numerical|numeric|number|numbers|percentage|percent|rate|ratio|statistic|statistics|how many|how much)\b",
    re.IGNORECASE,
)
_ANSWER_NUMBER_RE = re.compile(r"(?:\b\d+(?:\.\d+)?\b|\d+\s*%|\bpercent\b)", re.IGNORECASE)


_INTENT_RULES: tuple[tuple[re.Pattern[str], set[str]], ...] = (
    (
        re.compile(r"\b(?:barrier|barriers|obstacle|obstacles|delay|delays)\b", re.IGNORECASE),
        {"barrier", "obstacle", "delay", "stigma", "access", "financial", "geographic", "logistical", "coordination"},
    ),
    (
        re.compile(r"\b(?:symptom|symptoms|sign|signs|warning)\b", re.IGNORECASE),
        {"symptom", "sign", "warning", "lump", "bleeding", "pain", "lesion", "discharge"},
    ),
    (
        re.compile(r"\b(?:screening|screen)\b", re.IGNORECASE),
        {"screening", "screen", "asymptomatic", "target", "population"},
    ),
    (
        re.compile(r"\b(?:indicator|indicators|monitoring|evaluate|evaluation|target|targets|metric|metrics)\b", re.IGNORECASE),
        {"indicator", "monitor", "evaluation", "target", "metric", "proportion", "interval"},
    ),
    (
        re.compile(r"\b(?:treatment|therapy|treat)\b", re.IGNORECASE),
        {"treatment", "therapy", "treat", "care", "access"},
    ),
    (
        re.compile(r"\b(?:compare|difference|different|contrast)\b", re.IGNORECASE),
        {"whereas", "while", "contrast", "difference", "compared", "step", "screening", "diagnosis"},
    ),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.initialise()
    history_cleanup.start()
    yield
    history_cleanup.stop()


app = FastAPI(
    title=settings.app_name,
    version="3.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_conversation(user_id: str, conversation_id: str) -> dict[str, Any]:
    conversation = chat_store.get_conversation(user_id, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _normalise_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalise_question(question).encode("utf-8")).hexdigest()


def _document_set_hash(documents: list[dict[str, Any]]) -> str:
    parts = []
    for item in documents:
        parts.append(
            "|".join(
                (
                    str(item.get("sha256") or ""),
                    str(item.get("file_name") or ""),
                    str(item.get("pages") or ""),
                    str(item.get("chunks") or ""),
                )
            )
        )
    canonical = "\n".join(sorted(parts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stem(word: str) -> str:
    lowered = word.casefold().strip("'-")
    for suffix in ("ations", "ation", "ments", "ment", "ingly", "edly", "ing", "ies", "ers", "ed", "es", "s"):
        if len(lowered) > len(suffix) + 3 and lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _content_terms(text: str) -> set[str]:
    return {_stem(word) for word in _WORD_RE.findall(text) if word.casefold() not in _STOPWORDS}


def _needs_history(question: str) -> bool:
    clean = question.strip()
    if not clean:
        return False
    if _REFERENTIAL_RE.search(clean):
        return True
    words = clean.split()
    return len(words) <= 7 and bool(re.match(r"^(?:why|how|and|also|then|more)\b", clean, re.IGNORECASE))


def _answer_blocks(answer: str) -> list[str]:
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and any(re.match(r"^(?:[-*•]|\d+[.)])\s+", line) for line in lines):
            candidates = lines
        else:
            candidates = [" ".join(lines)] if lines else []
        for candidate in candidates:
            clean = re.sub(r"^(?:#{1,6}\s*|[-*•]\s+|\d+[.)]\s+)", "", candidate).strip()
            clean = re.sub(r"\[(?:S\d+)\]", "", clean).strip()
            if len(_WORD_RE.findall(clean)) >= 6:
                blocks.append(clean)
    return blocks


def _focus_terms(question: str) -> set[str]:
    terms = _content_terms(question) - _FOCUS_GENERIC
    for pattern, required_terms in _INTENT_RULES:
        if pattern.search(question):
            terms.update(_stem(term) for term in required_terms)
    return terms or _content_terms(question)


def _copied_from_previous(answer: str, previous_answers: list[str]) -> dict[str, Any]:
    answer_tokens = [_stem(word) for word in _WORD_RE.findall(answer)]
    answer_blocks = _answer_blocks(answer)
    strongest = {"matched_tokens": 0, "answer_ratio": 0.0, "previous_ratio": 0.0, "paragraph_match": 0.0}
    contaminated = False
    for previous in previous_answers:
        previous_tokens = [_stem(word) for word in _WORD_RE.findall(previous)]
        if len(previous_tokens) < 20 or len(answer_tokens) < 20:
            continue
        match = SequenceMatcher(None, answer_tokens, previous_tokens, autojunk=False).find_longest_match()
        answer_ratio = match.size / max(1, len(answer_tokens))
        previous_ratio = match.size / max(1, len(previous_tokens))
        if match.size > strongest["matched_tokens"]:
            strongest.update(
                {
                    "matched_tokens": match.size,
                    "answer_ratio": round(answer_ratio, 3),
                    "previous_ratio": round(previous_ratio, 3),
                }
            )
        if match.size >= 40 and (answer_ratio >= 0.12 or previous_ratio >= 0.20):
            contaminated = True

        previous_blocks = _answer_blocks(previous)
        for block in answer_blocks:
            block_terms = _content_terms(block)
            if len(block_terms) < 8:
                continue
            for old_block in previous_blocks:
                old_terms = _content_terms(old_block)
                if len(old_terms) < 8:
                    continue
                similarity = len(block_terms & old_terms) / max(1, len(block_terms | old_terms))
                strongest["paragraph_match"] = round(max(float(strongest["paragraph_match"]), similarity), 3)
                if similarity >= 0.82 and min(len(block_terms), len(old_terms)) >= 12:
                    contaminated = True
    return {"passed": not contaminated, **strongest}


def _result_quality(
    question: str,
    result: dict[str, Any],
    previous_answers: list[str] | None = None,
) -> dict[str, Any]:
    answer = str(result.get("answer") or "").strip()
    question_terms = _content_terms(question)
    answer_terms = _content_terms(answer)
    overlap_terms = sorted(question_terms & answer_terms)
    overlap_count = len(overlap_terms)
    overlap_ratio = overlap_count / max(1, len(question_terms))

    if len(question_terms) <= 3:
        alignment_passed = overlap_count >= 1
    elif len(question_terms) <= 7:
        alignment_passed = overlap_count >= 2 or overlap_ratio >= 0.30
    else:
        alignment_passed = overlap_count >= 3 or overlap_ratio >= 0.24

    intent_checks: list[dict[str, Any]] = []
    for pattern, required_terms in _INTENT_RULES:
        if not pattern.search(question):
            continue
        stemmed_required = {_stem(term) for term in required_terms}
        matched = sorted(answer_terms & stemmed_required)
        intent_checks.append({"matched": matched, "passed": bool(matched)})
    intent_passed = all(item["passed"] for item in intent_checks)

    focus = _focus_terms(question)
    blocks = _answer_blocks(answer)
    block_checks: list[dict[str, Any]] = []
    for block in blocks:
        block_terms = _content_terms(block)
        matched = sorted(block_terms & focus)
        block_checks.append({"preview": block[:140], "matched": matched, "passed": bool(matched)})
    aligned_blocks = sum(1 for item in block_checks if item["passed"])
    block_alignment_ratio = aligned_blocks / max(1, len(block_checks))
    block_alignment_passed = not block_checks or block_alignment_ratio >= 0.72

    numerical_requested = bool(_NUMERIC_REQUEST_RE.search(question))
    numerical_passed = not numerical_requested or bool(_ANSWER_NUMBER_RE.search(answer))

    copied_check = _copied_from_previous(answer, previous_answers or [])

    cited_ids = set(_CITATION_RE.findall(answer))
    card_ids = {str(item.get("id") or "") for item in result.get("citations", []) if item.get("id")}
    citations_match = bool(cited_ids) and cited_ids == card_ids

    trace = result.get("trace") or {}
    citation_coverage = float(trace.get("citation_coverage") or 0.0)
    support_ratio = float(trace.get("citation_support_ratio") or 0.0)
    citation_quality_passed = citations_match and citation_coverage >= 0.60 and support_ratio >= 0.45

    subqueries = [str(item) for item in trace.get("subqueries", []) if str(item).strip()]
    planner_terms = _content_terms(" ".join(subqueries))
    planner_overlap = len(question_terms & planner_terms) / max(1, len(question_terms)) if subqueries else 1.0
    planner_passed = planner_overlap >= 0.20

    semantic = trace.get("semantic_verifier") or {}
    semantic_verifier_passed = not semantic or bool(semantic.get("passed", False))

    answer_length_passed = len(answer) >= 60
    passed = all(
        (
            answer_length_passed,
            alignment_passed,
            intent_passed,
            block_alignment_passed,
            numerical_passed,
            copied_check["passed"],
            planner_passed,
        )
    )
    return {
        "passed": passed,
        "answer_length_passed": answer_length_passed,
        "alignment_passed": alignment_passed,
        "intent_passed": intent_passed,
        "block_alignment_passed": block_alignment_passed,
        "block_alignment_ratio": round(block_alignment_ratio, 3),
        "block_checks": block_checks,
        "numerical_requested": numerical_requested,
        "numerical_passed": numerical_passed,
        "previous_answer_contamination_passed": copied_check["passed"],
        "previous_answer_similarity": copied_check,
        "citation_quality_passed": citation_quality_passed,
        "planner_passed": planner_passed,
        "semantic_verifier_passed": semantic_verifier_passed,
        "question_term_overlap": overlap_terms,
        "question_term_overlap_ratio": round(overlap_ratio, 3),
        "intent_checks": intent_checks,
        "cited_ids": sorted(cited_ids),
        "citation_card_ids": sorted(card_ids),
        "citation_coverage": round(citation_coverage, 3),
        "citation_support_ratio": round(support_ratio, 3),
        "planner_overlap_ratio": round(planner_overlap, 3),
        "semantic_verifier": semantic,
    }


def _answer_and_save(
    job_id: str,
    user_id: str,
    conversation_id: str,
    request_id: str,
    user_message_id: str,
    question_hash: str,
    document_set_hash: str,
    question: str,
    mode: str,
    followup_question: str | None,
    previous_answers: list[str],
) -> dict[str, Any]:
    expected_binding = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "question_hash": question_hash,
        "document_set_hash": document_set_hash,
    }
    jobs.assert_binding(job_id, **expected_binding)
    if not chat_store.message_exists(user_id, conversation_id, user_message_id, role="user"):
        raise RuntimeError("The originating question was removed before the answer completed")

    result = answer_job(job_id, conversation_id, question, mode, followup_question)
    quality = _result_quality(question, result, previous_answers)
    retry_used = False

    if not quality["passed"]:
        retry_used = True
        jobs.update(
            job_id,
            step=4,
            phase="Rechecking answer alignment",
            detail="The first draft did not match the current question strongly enough; regenerating without prior-chat context",
            progress=0.66,
        )
        strict_question = (
            f"{question}\n\n"
            "Answer only the current question above. Ignore unrelated earlier topics. "
            "Use only the uploaded PDF evidence and include exact evidence citations for every factual point."
        )
        retry_mode = "deep" if mode == "deep" else "auto"
        result = answer_job(job_id, conversation_id, strict_question, retry_mode, None)
        quality = _result_quality(question, result, previous_answers)

    quality["retry_used"] = retry_used
    if not quality["passed"]:
        logger.warning("Answer rejected by final quality gate: %s", quality)
        raise RuntimeError(
            "The generated response did not pass the final question-alignment and citation checks, so it was not saved or shown. "
            "Please retry the question in Smart Auto or Deep Research mode."
        )

    jobs.assert_binding(job_id, **expected_binding)
    if _document_set_hash(store.list_documents(conversation_id)) != document_set_hash:
        raise RuntimeError("The processed PDF set changed while this answer was being generated")
    if not chat_store.message_exists(user_id, conversation_id, user_message_id, role="user"):
        raise RuntimeError("The originating question no longer exists")

    result.update(expected_binding)
    result["quality"] = quality
    assistant_message = chat_store.add_bound_assistant_message(
        user_id,
        conversation_id,
        str(result["answer"]),
        {
            "document_set_hash": document_set_hash,
            "citations": result.get("citations", []),
            "confidence": result.get("confidence"),
            "trace": result.get("trace"),
            "safety_note": result.get("safety_note"),
            "quality": quality,
        },
        parent_message_id=user_message_id,
        request_id=request_id,
        question_hash=question_hash,
    )
    result["assistant_message_id"] = assistant_message["id"]
    return result


@app.get("/")
def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, Any]:
    ollama_state = ollama.health()
    return {
        "status": "ok",
        "backend": "ready",
        "ollama": ollama_state,
        "auth_enabled": settings.auth_enabled,
        "chat_retention_months": settings.chat_retention_months,
        "profile": settings.model_profile,
        "primary_model": settings.llm_model,
        "fallback_model": settings.llm_fallback_model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model if settings.embedding_backend == "sentence_transformers" else settings.ollama_embedding_model,
        "reranker": settings.reranker_model if settings.enable_reranker else None,
        "reranker_fallback": settings.reranker_fallback_model if settings.enable_reranker else None,
    }


@app.get("/config")
def config() -> dict[str, Any]:
    return {
        "profile": settings.model_profile,
        "primary_model": settings.llm_model,
        "fallback_model": settings.llm_fallback_model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model if settings.embedding_backend == "sentence_transformers" else settings.ollama_embedding_model,
        "reranker_model": settings.reranker_model if settings.enable_reranker else None,
        "reranker_fallback_model": settings.reranker_fallback_model if settings.enable_reranker else None,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "final_k": settings.final_k,
        "max_context_chars": settings.max_context_chars,
        "auth_enabled": settings.auth_enabled,
        "chat_retention_months": settings.chat_retention_months,
        "answer_binding_enabled": True,
        "final_quality_gate_enabled": True,
    }


@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> TokenResponse:
    try:
        user = auth.register(request.email, request.display_name, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc)) from exc
    token, expires = auth.create_access_token(user)
    return TokenResponse(access_token=token, expires_in_seconds=expires, user=UserView(**user))


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    user = auth.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires = auth.create_access_token(user)
    return TokenResponse(access_token=token, expires_in_seconds=expires, user=UserView(**user))


@app.get("/auth/me", response_model=UserView)
def me(current_user: Annotated[dict[str, Any], Depends(get_current_user)]) -> UserView:
    return UserView(**current_user)


@app.get("/conversations", response_model=list[ConversationView])
def list_conversations(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> list[ConversationView]:
    return [ConversationView(**item) for item in chat_store.list_conversations(current_user["id"])]


@app.post("/conversations", response_model=ConversationView, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreate,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> ConversationView:
    return ConversationView(**chat_store.create_conversation(current_user["id"], request.title))


@app.patch("/conversations/{conversation_id}", response_model=ConversationView)
def rename_conversation(
    conversation_id: str,
    request: ConversationRename,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> ConversationView:
    try:
        conversation = chat_store.rename_conversation(current_user["id"], conversation_id, request.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationView(**conversation)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, str]:
    _require_conversation(current_user["id"], conversation_id)
    vector_store.clear(conversation_id)
    bm25_store.invalidate(conversation_id)
    store.clear(conversation_id)
    chat_store.delete_conversation(current_user["id"], conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageView])
def list_messages(
    conversation_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> list[MessageView]:
    try:
        messages = chat_store.list_messages(current_user["id"], conversation_id, limit)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return [MessageView(**item) for item in messages]


@app.delete("/conversations/{conversation_id}/messages")
def clear_messages(
    conversation_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, str]:
    if not chat_store.clear_messages(current_user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "cleared", "conversation_id": conversation_id}


@app.post("/documents/upload", response_model=JobCreated, status_code=202)
async def upload_documents(
    session_id: Annotated[str, Form(min_length=4, max_length=120)],
    files: Annotated[list[UploadFile], File()],
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> JobCreated:
    _require_conversation(current_user["id"], session_id)
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files were supplied")
    if len(files) > settings.max_files_per_session:
        raise HTTPException(
            status_code=413,
            detail=f"A conversation supports at most {settings.max_files_per_session} PDFs",
        )
    payloads: list[tuple[str, bytes]] = []
    maximum_bytes = settings.max_upload_mb * 1024 * 1024
    consumed = 0
    for file in files:
        remaining = maximum_bytes - consumed
        if remaining <= 0:
            raise HTTPException(status_code=413, detail=f"Combined upload exceeds {settings.max_upload_mb} MB")
        data = await file.read(remaining + 1)
        await file.close()
        if len(data) > remaining:
            raise HTTPException(status_code=413, detail=f"Combined upload exceeds {settings.max_upload_mb} MB")
        consumed += len(data)
        payloads.append((file.filename or "document.pdf", data))
    job_id = jobs.create("upload", UPLOAD_STEPS, owner_id=current_user["id"])
    jobs.submit(job_id, ingest_job, session_id, payloads)
    return JobCreated(job_id=job_id)


@app.get("/documents/{session_id}")
def list_documents(
    session_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    _require_conversation(current_user["id"], session_id)
    documents = store.list_documents(session_id)
    chunks = store.load_chunks(session_id)
    return {"session_id": session_id, "documents": documents, "chunks": len(chunks)}


@app.delete("/documents/{session_id}")
def clear_documents(
    session_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, str]:
    _require_conversation(current_user["id"], session_id)
    vector_store.clear(session_id)
    bm25_store.invalidate(session_id)
    store.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/query", response_model=JobCreated, status_code=202)
def query(
    request: QueryRequest,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> JobCreated:
    _require_conversation(current_user["id"], request.session_id)
    documents = store.list_documents(request.session_id)
    if not documents:
        raise HTTPException(status_code=400, detail="Upload and process at least one PDF first")

    normalised_hash = _question_hash(request.question)
    document_set_hash = _document_set_hash(documents)
    with _query_lock:
        existing = jobs.find_by_request(current_user["id"], request.request_id)
        if existing:
            if (
                existing.get("conversation_id") != request.session_id
                or existing.get("question_hash") != normalised_hash
                or existing.get("document_set_hash") != document_set_hash
            ):
                raise HTTPException(status_code=409, detail="request_id was already used for a different question")
            return JobCreated(
                job_id=existing["job_id"],
                status=existing["status"],
                request_id=existing.get("request_id"),
                conversation_id=existing.get("conversation_id"),
                user_message_id=existing.get("user_message_id"),
                question_hash=existing.get("question_hash"),
                document_set_hash=existing.get("document_set_hash"),
            )

        followup_question = (
            chat_store.previous_user_question(current_user["id"], request.session_id)
            if _needs_history(request.question)
            else None
        )
        previous_answers = chat_store.recent_assistant_answers(current_user["id"], request.session_id, limit=6)
        user_message = chat_store.add_message(
            current_user["id"],
            request.session_id,
            "user",
            request.question,
            {
                "request_id": request.request_id,
                "question_hash": normalised_hash,
                "document_set_hash": document_set_hash,
                "message_kind": "question",
            },
        )
        binding = {
            "request_id": request.request_id,
            "conversation_id": request.session_id,
            "user_message_id": user_message["id"],
            "question_hash": normalised_hash,
            "document_set_hash": document_set_hash,
        }
        job_id = jobs.create("query", QUERY_STEPS, owner_id=current_user["id"], binding=binding)
        jobs.submit(
            job_id,
            _answer_and_save,
            current_user["id"],
            request.session_id,
            request.request_id,
            user_message["id"],
            normalised_hash,
            document_set_hash,
            request.question,
            request.mode,
            followup_question,
            previous_answers,
        )

    return JobCreated(job_id=job_id, **binding)


@app.get("/jobs/{job_id}", response_model=JobView)
def get_job(
    job_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> JobView:
    job = jobs.get(job_id, owner_id=current_user["id"])
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobView(**job)