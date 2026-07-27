from __future__ import annotations

import hashlib
import html
import time
from uuid import uuid4

import streamlit as st

from api_client import api
from ui_styles import apply_styles


st.set_page_config(
    page_title="Healthcare Guidelines RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("ui_theme", "blue")
apply_styles()

_BINDING_KEYS = ("request_id", "conversation_id", "user_message_id", "question_hash", "document_set_hash")
_MODE_LABEL_TO_VALUE = {"Smart Auto": "auto", "Deep Research": "deep", "Fast": "fast"}
_MODE_VALUE_TO_LABEL = {value: key for key, value in _MODE_LABEL_TO_VALUE.items()}
_THEME_LABEL_TO_VALUE = {"Blue": "blue", "Black": "black", "White": "white"}
_THEME_VALUE_TO_LABEL = {value: key for key, value in _THEME_LABEL_TO_VALUE.items()}


def init_state() -> None:
    defaults = {
        "token": None,
        "user": None,
        "conversation_id": None,
        "loaded_conversation_id": None,
        "messages": [],
        "thread_warnings": [],
        "last_job": None,
        "pending_request": None,
        "mode": "auto",
        "ui_theme": "blue",
        "rename_title": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def question_hash(question: str) -> str:
    normalised = " ".join(question.split()).casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def fmt_bytes(value: int) -> str:
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def short_title(title: str, limit: int = 38) -> str:
    cleaned = " ".join(str(title or "Research chat").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def validate_binding(payload: dict, expected: dict, label: str) -> None:
    for key in _BINDING_KEYS:
        expected_value = str(expected.get(key) or "")
        actual_value = str(payload.get(key) or "")
        if expected_value and actual_value != expected_value:
            raise RuntimeError(f"{label} did not match the current question ({key})")


def poll_job(job_id: str, expected: dict | None = None, status_text: str = "Working") -> dict:
    progress = st.progress(0.0)
    status_line = st.empty()
    while True:
        if expected:
            if str(st.session_state.conversation_id or "") != str(expected.get("conversation_id") or ""):
                raise RuntimeError("The active conversation changed while the answer was running")
            if st.session_state.pending_request != expected:
                raise RuntimeError("The pending request was replaced before completion")
        job = api.job(job_id)
        if expected:
            validate_binding(job, expected, "Background job")
        st.session_state.last_job = job
        progress.progress(float(job.get("progress", 0.0)))
        phase = str(job.get("phase") or status_text)
        detail = str(job.get("detail") or "")
        status_line.caption(f"{phase}: {detail}" if detail else phase)
        if job["status"] == "complete":
            result = job.get("result") or {}
            if expected:
                validate_binding(result, expected, "Completed answer")
            progress.empty()
            status_line.empty()
            return result
        if job["status"] == "failed":
            progress.empty()
            status_line.empty()
            raise RuntimeError(job.get("error") or "The background job failed")
        time.sleep(0.55)


def normalise_message(item: dict) -> dict:
    message = {
        "id": item.get("id"),
        "role": item["role"],
        "content": item["content"],
        "created_at": item.get("created_at"),
    }
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        message.update(metadata)
    return message


def normalise_thread(items: list[dict]) -> tuple[list[dict], list[str]]:
    messages: list[dict] = []
    warnings: list[str] = []
    user_messages: dict[str, dict] = {}
    for raw_item in items:
        message = normalise_message(raw_item)
        if message["role"] == "user":
            if message.get("id"):
                user_messages[str(message["id"])] = message
            messages.append(message)
            continue
        if message["role"] != "assistant":
            messages.append(message)
            continue

        parent_id = str(message.get("in_reply_to_message_id") or "")
        request_id = str(message.get("request_id") or "")
        stored_hash = str(message.get("question_hash") or "")
        if not parent_id or not request_id or not stored_hash:
            warnings.append("An unbound legacy assistant response was hidden for safety.")
            continue

        parent = user_messages.get(parent_id)
        parent_request = str((parent or {}).get("request_id") or "")
        parent_hash = str((parent or {}).get("question_hash") or "")
        parent_document_hash = str((parent or {}).get("document_set_hash") or "")
        answer_document_hash = str(message.get("document_set_hash") or "")
        quality = message.get("quality") or {}
        if (
            not parent
            or not request_id
            or not stored_hash
            or not parent_document_hash
            or not answer_document_hash
            or parent_request != request_id
            or parent_hash != stored_hash
            or parent_document_hash != answer_document_hash
            or not bool(quality.get("passed", False))
            or not bool(quality.get("previous_answer_contamination_passed", False))
        ):
            warnings.append("A saved response with an invalid question binding was hidden.")
            continue
        messages.append(message)
    return messages, list(dict.fromkeys(warnings))


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    confidence = message.get("confidence") or {}
    if confidence:
        st.caption(f"Grounding confidence: {confidence.get('label', 'unknown')} ({confidence.get('score', 0):.2f})")
    quality = message.get("quality") or {}
    if quality.get("passed"):
        retry_note = " · regenerated once" if quality.get("retry_used") else ""
        st.caption(f"Question alignment checks passed{retry_note}.")
    citations = message.get("citations") or []
    if citations:
        with st.expander(f"Evidence citations ({len(citations)})", expanded=False):
            for citation in citations:
                st.markdown(
                    f"""
                    <div class="source-card">
                      <div class="source-title">[{html.escape(str(citation['id']))}] {html.escape(str(citation['source']))}</div>
                      <div class="source-meta">Page {int(citation['page'])} · {html.escape(str(citation['section']))} · score {float(citation['score']):.3f}</div>
                      <div class="small-muted">{html.escape(str(citation['excerpt']))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def clear_login_state() -> None:
    api.set_token(None)
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.conversation_id = None
    st.session_state.loaded_conversation_id = None
    st.session_state.messages = []
    st.session_state.thread_warnings = []
    st.session_state.last_job = None
    st.session_state.pending_request = None


def render_auth_screen(health: dict) -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Authenticated · evidence-grounded · private</div>
          <h1>Healthcare Guidelines RAG Assistant</h1>
          <p>Log in to use the medical-document assistant and keep your own chat history for up to five months.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if health.get("status") != "ok":
        st.error("The FastAPI backend is unavailable. Start it before logging in.")
        st.stop()

    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
        if submitted:
            try:
                payload = api.login(email, password)
                st.session_state.token = payload["access_token"]
                st.session_state.user = payload["user"]
                api.set_token(st.session_state.token)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with register_tab:
        with st.form("register_form"):
            display_name = st.text_input("Display name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password (minimum 8 characters)", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match")
            else:
                try:
                    payload = api.register(email, display_name, password)
                    st.session_state.token = payload["access_token"]
                    st.session_state.user = payload["user"]
                    api.set_token(st.session_state.token)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    st.caption("Each account can access only its own conversations. Messages older than five months are automatically removed.")
    st.stop()


def load_current_messages() -> None:
    conversation_id = st.session_state.conversation_id
    if not conversation_id:
        st.session_state.messages = []
        st.session_state.thread_warnings = []
        st.session_state.loaded_conversation_id = None
        return
    messages, warnings = normalise_thread(api.messages(conversation_id))
    st.session_state.messages = messages
    st.session_state.thread_warnings = warnings
    st.session_state.loaded_conversation_id = conversation_id


def switch_conversation(conversation_id: str) -> None:
    if conversation_id == st.session_state.conversation_id:
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.loaded_conversation_id = None
    st.session_state.last_job = None
    st.session_state.pending_request = None
    st.rerun()


def render_sidebar(conversations: list[dict], health: dict, config: dict) -> None:
    with st.sidebar:
        st.markdown("## 🩺 RAG Control Centre")
        st.write(f"Signed in as **{st.session_state.user['display_name']}**")
        st.caption(st.session_state.user["email"])

        theme_labels = list(_THEME_LABEL_TO_VALUE.keys())
        current_theme_label = _THEME_VALUE_TO_LABEL.get(st.session_state.ui_theme, "Blue")
        selected_theme = st.selectbox(
            "UI theme",
            theme_labels,
            index=theme_labels.index(current_theme_label),
            help="Choose Blue, Black, or White. Text contrast adjusts automatically.",
        )
        st.session_state.ui_theme = _THEME_LABEL_TO_VALUE[selected_theme]

        if st.button("Log out", use_container_width=True):
            clear_login_state()
            st.rerun()

        st.divider()
        if health.get("status") == "ok":
            st.markdown('<div class="status-ok">● Backend online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-bad">● Backend unavailable</div>', unsafe_allow_html=True)
            st.caption(health.get("error", "Start the backend."))

        ollama_state = health.get("ollama") or {}
        if ollama_state.get("reachable"):
            st.markdown('<div class="status-ok">● Ollama online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-bad">● Ollama offline</div>', unsafe_allow_html=True)

        with st.expander("Active medical stack", expanded=False):
            st.caption(f"Profile: {config.get('profile', 'unknown')}")
            st.write(f"**Generator:** `{config.get('primary_model', 'unknown')}`")
            st.write(f"**Fallback:** `{config.get('fallback_model', 'unknown')}`")
            st.write(f"**Embeddings:** `{config.get('embedding_model', 'unknown')}`")
            st.write(f"**Medical reranker:** `{config.get('reranker_model') or 'disabled'}`")
            if config.get("reranker_fallback_model"):
                st.write(f"**Reranker fallback:** `{config.get('reranker_fallback_model')}`")
            st.write("**Retrieval:** Chroma cosine + BM25 + reciprocal-rank fusion")
            st.write("**Answer integrity:** request binding + relevance and contamination gates")

        st.divider()
        top_left, top_right = st.columns(2)
        if top_left.button("New chat", use_container_width=True):
            created = api.create_conversation()
            st.session_state.conversation_id = created["id"]
            st.session_state.loaded_conversation_id = None
            st.session_state.last_job = None
            st.session_state.pending_request = None
            st.rerun()
        if top_right.button("Delete chat", use_container_width=True):
            api.delete_conversation(st.session_state.conversation_id)
            st.session_state.conversation_id = None
            st.session_state.loaded_conversation_id = None
            st.session_state.messages = []
            st.session_state.pending_request = None
            st.rerun()

        if st.button("Clear saved messages", use_container_width=True):
            api.clear_messages(st.session_state.conversation_id)
            st.session_state.messages = []
            st.session_state.thread_warnings = []
            st.session_state.loaded_conversation_id = st.session_state.conversation_id
            st.session_state.last_job = None
            st.session_state.pending_request = None
            st.rerun()

        st.caption(f"History retention: {config.get('chat_retention_months', 5)} months")

        current = next((item for item in conversations if item["id"] == st.session_state.conversation_id), None)
        current_title = str((current or {}).get("title") or "Research chat")
        with st.expander("Rename selected chat", expanded=False):
            with st.form("rename_chat_form"):
                new_title = st.text_input("Chat title", value=current_title, max_chars=90)
                rename_submitted = st.form_submit_button("Save name", use_container_width=True)
            if rename_submitted:
                clean_title = " ".join(new_title.split()) or "Research chat"
                try:
                    api.rename_conversation(st.session_state.conversation_id, clean_title)
                    st.success("Chat renamed.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        st.markdown('<div class="chat-history-heading">Past chats</div>', unsafe_allow_html=True)
        st.caption("Click any chat below to reopen it.")
        with st.container(height=355, border=True):
            for index, item in enumerate(conversations):
                cid = str(item["id"])
                title = short_title(str(item.get("title") or "Research chat"), 42)
                prefix = "● " if cid == st.session_state.conversation_id else ""
                button_type = "primary" if cid == st.session_state.conversation_id else "secondary"
                if st.button(prefix + title, key=f"chat_select_{cid}_{index}", use_container_width=True, type=button_type):
                    switch_conversation(cid)


def render_knowledge_base(documents: list[dict], doc_state: dict, health: dict) -> None:
    st.markdown('<div class="top-panel">', unsafe_allow_html=True)
    st.subheader("Knowledge base for this chat")
    st.caption("Upload and process public healthcare guideline or research PDFs for the selected conversation.")
    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs and their indexes belong only to the selected conversation.",
    )
    process_disabled = not uploads or health.get("status") != "ok"
    if st.button("Process PDFs", use_container_width=True, type="primary", disabled=process_disabled):
        try:
            job_id = api.start_upload(st.session_state.conversation_id, uploads)
            with st.status("Processing PDFs", expanded=True) as status_box_parent:
                result = poll_job(job_id, status_text="Processing")
                status_box_parent.update(label="Knowledge base ready", state="complete", expanded=False)
            st.success(f"Processed {result.get('pages', 0)} pages into {result.get('chunks', 0)} evidence chunks.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    if documents:
        st.markdown("**Processed PDFs**")
    for document in documents:
        st.markdown(
            f"""
            <div class="doc-card">
              <div class="doc-title">{html.escape(str(document['file_name']))}</div>
              <div class="doc-meta">{int(document['pages'])} pages · {int(document['chunks'])} chunks · {fmt_bytes(int(document['size_bytes']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if document.get("extraction_warning"):
            st.warning(str(document["extraction_warning"]), icon="⚠️")
    if documents and st.button("Clear knowledge base", use_container_width=True):
        api.clear_documents(st.session_state.conversation_id)
        st.session_state.last_job = None
        st.session_state.pending_request = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    metric_columns = st.columns(3)
    metric_columns[0].metric("PDFs indexed", len(documents))
    metric_columns[1].metric("Pages available", sum(int(item.get("pages", 0)) for item in documents))
    metric_columns[2].metric("Evidence chunks", int(doc_state.get("chunks", 0)))


init_state()
api.set_token(st.session_state.token)

try:
    health = api.health()
    config = api.config()
except Exception as exc:  # noqa: BLE001
    health = {"status": "offline", "error": str(exc), "ollama": {"reachable": False}}
    config = {}

if st.session_state.token and not st.session_state.user:
    try:
        st.session_state.user = api.me()
    except Exception:
        clear_login_state()

if not st.session_state.user:
    render_auth_screen(health)

try:
    conversations = api.conversations()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to load your conversations: {exc}")
    st.stop()

if not conversations:
    conversations = [api.create_conversation()]

conversation_ids = [item["id"] for item in conversations]
if st.session_state.conversation_id not in conversation_ids:
    st.session_state.conversation_id = conversation_ids[0]

if st.session_state.loaded_conversation_id != st.session_state.conversation_id:
    load_current_messages()

try:
    doc_state = api.documents(st.session_state.conversation_id)
except Exception:
    doc_state = {"documents": [], "chunks": 0}

documents = doc_state.get("documents", [])

render_sidebar(conversations, health, config)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Authenticated · evidence-grounded · private</div>
      <h1>Healthcare Guidelines RAG Assistant</h1>
      <p>Research one or many uploaded medical PDFs using biomedical retrieval, hybrid search, reranking, adaptive document-wide investigation, and page-level citations.</p>
    </div>
    <div class="safety-card"><b>Educational use only.</b> This assistant does not diagnose or prescribe. Verify important decisions in the original document and with qualified clinical professionals.</div>
    """,
    unsafe_allow_html=True,
)

render_knowledge_base(documents, doc_state, health)

if not documents:
    st.info("Upload and process at least one text-based PDF above. Chat input unlocks after processing.")

for warning in st.session_state.thread_warnings:
    st.warning(warning)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

st.markdown('<div class="bottom-mode-panel">', unsafe_allow_html=True)
st.markdown('<div class="bottom-mode-title">Answer mode for next question</div>', unsafe_allow_html=True)
st.markdown('<div class="bottom-mode-help">Choose speed versus depth before sending your question.</div>', unsafe_allow_html=True)
mode_options = ["Smart Auto", "Deep Research", "Fast"]
current_mode_label = _MODE_VALUE_TO_LABEL.get(st.session_state.mode, "Smart Auto")
mode_label = st.radio(
    "Answer mode",
    mode_options,
    index=mode_options.index(current_mode_label),
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.mode = _MODE_LABEL_TO_VALUE[mode_label]
st.markdown('</div>', unsafe_allow_html=True)

question = st.chat_input(
    "Ask a question about the processed PDFs…",
    disabled=not documents or health.get("status") != "ok" or bool(st.session_state.pending_request),
)
if question:
    current_conversation = str(st.session_state.conversation_id)
    current_request_id = str(uuid4())
    expected_hash = question_hash(question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            created = api.start_query(current_conversation, question, st.session_state.mode, current_request_id)
            expected = {
                "request_id": current_request_id,
                "conversation_id": current_conversation,
                "user_message_id": str(created["user_message_id"]),
                "question_hash": expected_hash,
                "document_set_hash": str(created["document_set_hash"]),
            }
            validate_binding(created, expected, "Created query")
            st.session_state.pending_request = expected
            with st.status("Generating grounded answer", expanded=True) as status_box_parent:
                result = poll_job(str(created["job_id"]), expected, status_text="Researching")
                quality = result.get("quality") or {}
                if not bool(quality.get("passed", False)):
                    raise RuntimeError("The completed response did not pass the final quality gate")
                status_box_parent.update(label="Answer ready", state="complete", expanded=False)
            st.session_state.pending_request = None
            st.session_state.loaded_conversation_id = None
            load_current_messages()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state.pending_request = None
            st.session_state.loaded_conversation_id = None
            try:
                load_current_messages()
            except Exception:
                pass
            st.error(f"The assistant could not complete this request: {exc}")
