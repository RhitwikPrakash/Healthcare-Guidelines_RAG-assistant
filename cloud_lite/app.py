from __future__ import annotations

import hashlib
from datetime import datetime
from uuid import uuid4

import streamlit as st

from auth_utils import authenticate, auth_is_configured
from llm_client import (
    RATE_LIMIT_MESSAGE,
    ProviderRateLimitError,
    ProviderUnavailableError,
    generate_answer,
    get_gemini_api_key,
    get_secret,
)
from pdf_utils import extract_pdf_pages
from rag_core import (
    build_answer_prompt,
    build_chunks,
    build_evidence_text,
    build_index,
    build_reference_lines,
    format_evidence_cards,
    looks_incomplete,
    retrieve_chunks,
    strip_inline_citations,
)
from request_gate import wait_for_request_slot


st.set_page_config(
    page_title="Healthcare RAG Lite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.25);
}

.rag-card, .auth-card {
    border: 1px solid rgba(56, 189, 248, 0.35);
    background: rgba(15, 23, 42, 0.7);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
}

.auth-title {
    text-align: center;
    margin-bottom: 0.2rem;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.9rem;
}

.success-dot, .fail-dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.success-dot { background-color: #22c55e; }
.fail-dot { background-color: #ef4444; }

.reference-list {
    margin-top: 0.7rem;
    padding: 0.8rem 1rem;
    border-left: 3px solid rgba(56, 189, 248, 0.65);
    background: rgba(30, 41, 59, 0.35);
    border-radius: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def process_uploaded_files(file_payloads):
    all_pages = []
    for item in file_payloads:
        all_pages.extend(extract_pdf_pages(item["bytes"], item["name"]))

    chunks = build_chunks(all_pages)
    if not chunks:
        return {
            "pages": all_pages,
            "chunks": [],
            "vectorizer": None,
            "matrix": None,
        }

    vectorizer, matrix = build_index(chunks)
    return {
        "pages": all_pages,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def _make_chat(title: str = "New chat") -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": str(uuid4()),
        "title": title,
        "messages": [],
        "notice": None,
        "created_at": now,
        "updated_at": now,
    }


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("processed_key", None)
    st.session_state.setdefault("rag_data", None)

    if "chats" not in st.session_state or not st.session_state.chats:
        chat = _make_chat()
        st.session_state.chats = {chat["id"]: chat}
        st.session_state.active_chat_id = chat["id"]
    elif "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))


def active_chat() -> dict:
    chat_id = st.session_state.active_chat_id
    chat = st.session_state.chats.get(chat_id)
    if chat is None:
        chat = _make_chat()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.active_chat_id = chat["id"]
    return chat


def touch_chat(chat: dict) -> None:
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")


def create_new_chat() -> None:
    chat = _make_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.active_chat_id = chat["id"]


def clear_login_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def render_auth_screen() -> None:
    st.markdown("<h1 class='auth-title'>Healthcare Guidelines RAG Lite</h1>", unsafe_allow_html=True)
    st.caption("Sign in before accessing uploaded healthcare documents and chat tools.")

    if not auth_is_configured():
        st.error(
            "Authentication is not configured. Add AUTH_USERNAME, AUTH_EMAIL and "
            "AUTH_PASSWORD (or AUTH_PASSWORD_HASH) to Streamlit Secrets."
        )
        st.stop()

    left, centre, right = st.columns([1, 1.25, 1])
    with centre:
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Secure sign in")
            username = st.text_input("Username")
            email = st.text_input("Email ID")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Log in",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            user = authenticate(username, email, password)
            if user is None:
                st.error("Incorrect username, email ID or password.")
            else:
                st.session_state.authenticated = True
                st.session_state.auth_user = {
                    "username": user.username,
                    "email": user.email,
                }
                st.rerun()

    st.caption("Educational use only. This application does not diagnose or prescribe.")


def render_sidebar() -> None:
    user = st.session_state.auth_user or {}
    st.sidebar.title("Healthcare RAG Lite")
    st.sidebar.caption(f"Signed in as **{user.get('username', 'User')}**")
    st.sidebar.caption(str(user.get("email", "")))

    if st.sidebar.button("Log out", use_container_width=True):
        clear_login_state()
        st.rerun()

    st.sidebar.divider()

    gemini_ready = bool(get_gemini_api_key())
    groq_ready = bool(get_secret("GROQ_API_KEY"))
    if gemini_ready:
        st.sidebar.markdown(
            '<span class="success-dot"></span>Gemini API ready',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="fail-dot"></span>Gemini API missing',
            unsafe_allow_html=True,
        )

    if groq_ready:
        st.sidebar.markdown(
            '<span class="success-dot"></span>Groq fallback ready',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="fail-dot"></span>Groq fallback missing',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    if st.sidebar.button("New chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    if st.sidebar.button("Clear processed PDFs", use_container_width=True):
        st.session_state.processed_key = None
        st.session_state.rag_data = None
        active_chat()["messages"] = []
        active_chat()["notice"] = None
        touch_chat(active_chat())
        st.rerun()

    chat = active_chat()
    with st.sidebar.expander("Rename selected chat", expanded=False):
        with st.form(f"rename_chat_{chat['id']}"):
            new_title = st.text_input(
                "Chat name",
                value=chat["title"],
                max_chars=60,
            )
            rename_submitted = st.form_submit_button(
                "Rename",
                use_container_width=True,
            )
        if rename_submitted:
            cleaned_title = " ".join(new_title.split()).strip()
            if cleaned_title:
                chat["title"] = cleaned_title
                touch_chat(chat)
                st.rerun()
            else:
                st.warning("Enter a chat name.")

    st.sidebar.markdown("### Past chats")
    ordered_chats = sorted(
        st.session_state.chats.values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )
    for item in ordered_chats:
        selected = item["id"] == st.session_state.active_chat_id
        label = f"{'● ' if selected else ''}{item['title']}"
        if st.sidebar.button(
            label,
            key=f"open_chat_{item['id']}",
            use_container_width=True,
            disabled=selected,
        ):
            st.session_state.active_chat_id = item["id"]
            st.rerun()

    st.sidebar.caption("Past chats are retained only in the current Streamlit session.")
    st.sidebar.divider()
    st.sidebar.caption(
        "PDFs are processed in memory for the current session. "
        "No permanent local database is used."
    )
    st.sidebar.caption("Educational use only. Not a diagnostic or prescribing system.")


def render_reference_block(reference_lines: list[str]) -> None:
    if not reference_lines:
        return
    st.markdown("**References**")
    for line in reference_lines:
        st.markdown(f"- {line}")


def render_messages() -> None:
    chat = active_chat()
    for message in chat["messages"]:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])
            continue

        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(message["content"])
            render_reference_block(message.get("references", []))

            provider = message.get("provider")
            model = message.get("model")
            if provider and model:
                st.caption(f"Generated by {provider} · {model}")

            evidence_cards = message.get("evidence_cards", [])
            if evidence_cards:
                with st.expander(f"Evidence citations ({len(evidence_cards)})"):
                    for card in evidence_cards:
                        st.markdown(
                            f"**Evidence {card['label']}: {card['source']} — page {card['page']}**"
                        )
                        st.write(card["text"])
                        st.divider()

            if message.get("completion_warning"):
                st.warning(
                    "The provider returned a possibly incomplete answer. "
                    "Review the listed evidence before relying on it."
                )

    if chat.get("notice"):
        st.warning(chat["notice"])


def _countdown_callback(progress_box, text_box):
    def update(fraction: float, remaining: int, total: int) -> None:
        progress_box.progress(fraction)
        if remaining > 0:
            text_box.markdown(
                f"**Researching retrieved evidence and pacing the API request... "
                f"about {remaining} second(s) remaining.**"
            )
        else:
            text_box.markdown("**Evidence prepared. Requesting the grounded answer...**")

    return update


def answer_question(question: str, answer_mode: str) -> bool:
    rag_data = st.session_state.rag_data
    chat = active_chat()
    chat["notice"] = None

    if not rag_data or not rag_data["chunks"]:
        chat["notice"] = "Please upload and process at least one PDF first."
        return False

    retrieved = retrieve_chunks(
        question=question,
        chunks=rag_data["chunks"],
        vectorizer=rag_data["vectorizer"],
        matrix=rag_data["matrix"],
        top_k=8,
    )
    evidence_text = build_evidence_text(retrieved)
    prompt = build_answer_prompt(
        question=question,
        evidence_text=evidence_text,
        answer_mode=answer_mode,
    )
    evidence_cards = format_evidence_cards(retrieved)
    references = build_reference_lines(evidence_cards)

    try:
        with st.status("Researching and synthesizing the answer", expanded=True) as status:
            status.write("Relevant PDF evidence has been retrieved and ranked.")
            countdown_text = st.empty()
            countdown_progress = st.progress(0.0)
            wait_for_request_slot(
                _countdown_callback(countdown_progress, countdown_text),
                minimum_delay=20.0,
                maximum_delay=30.0,
                minimum_spacing=25.0,
            )
            answer, provider, model = generate_answer(prompt)
            status.update(
                label="Grounded answer prepared",
                state="complete",
                expanded=False,
            )
    except ProviderRateLimitError:
        chat["notice"] = RATE_LIMIT_MESSAGE
        touch_chat(chat)
        return False
    except ProviderUnavailableError as error:
        chat["notice"] = str(error)
        touch_chat(chat)
        return False
    except Exception:
        chat["notice"] = "The answer could not be generated. Please try again shortly."
        touch_chat(chat)
        return False

    clean_answer = strip_inline_citations(answer)
    chat["messages"].append(
        {
            "role": "assistant",
            "content": clean_answer,
            "provider": provider,
            "model": model,
            "references": references,
            "evidence_cards": evidence_cards,
            "completion_warning": looks_incomplete(clean_answer),
        }
    )
    touch_chat(chat)
    return True


def build_markdown_download() -> str:
    chat = active_chat()
    lines = [
        f"# {chat['title']}",
        "",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for message in chat["messages"]:
        if message["role"] == "user":
            lines.extend(["## User", message["content"], ""])
            continue

        lines.extend(["## Assistant", message["content"], ""])
        references = message.get("references", [])
        if references:
            lines.append("### References")
            lines.extend(f"- {item}" for item in references)
            lines.append("")

        evidence_cards = message.get("evidence_cards", [])
        if evidence_cards:
            lines.append("### Evidence")
            for card in evidence_cards:
                lines.append(
                    f"- Evidence {card['label']}: {card['source']}, page {card['page']}"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    init_state()

    if not st.session_state.authenticated:
        render_auth_screen()
        st.stop()

    render_sidebar()

    st.title("Healthcare Guidelines RAG Lite")
    st.caption(
        "Upload healthcare guideline PDFs, ask questions, and receive "
        "evidence-grounded answers with references shown after each answer."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF guideline documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        process_clicked = st.button(
            "Process PDFs",
            type="primary",
            use_container_width=True,
        )
    with col2:
        answer_mode = st.radio(
            "Answer mode",
            ["Smart Auto", "Deep Research", "Fast"],
            horizontal=True,
        )

    if uploaded_files and process_clicked:
        file_payloads = []
        for uploaded_file in uploaded_files:
            data = uploaded_file.getvalue()
            file_payloads.append(
                {
                    "name": uploaded_file.name,
                    "bytes": data,
                    "hash": file_hash(data),
                }
            )

        combined_key = "|".join(item["hash"] for item in file_payloads)
        with st.spinner("Processing PDFs and building retrieval index..."):
            rag_data = process_uploaded_files(file_payloads)

        st.session_state.processed_key = combined_key
        st.session_state.rag_data = rag_data
        active_chat()["messages"] = []
        active_chat()["notice"] = None
        touch_chat(active_chat())

        page_count = len(rag_data["pages"])
        chunk_count = len(rag_data["chunks"])
        st.success(
            f"Processed {len(uploaded_files)} PDF(s), "
            f"{page_count} page(s), {chunk_count} chunks."
        )

    if st.session_state.rag_data:
        pages = len(st.session_state.rag_data["pages"])
        chunks = len(st.session_state.rag_data["chunks"])
        st.info(f"Active knowledge base: {pages} page(s), {chunks} evidence chunk(s).")

    st.divider()
    render_messages()

    question = st.chat_input(
        "Ask a question about the uploaded PDFs...",
        disabled=not bool(st.session_state.rag_data),
    )
    if question:
        chat = active_chat()
        chat["notice"] = None
        chat["messages"].append({"role": "user", "content": question})
        if chat["title"] == "New chat":
            compact_question = " ".join(question.split())
            chat["title"] = compact_question[:52] + ("…" if len(compact_question) > 52 else "")
        touch_chat(chat)

        answer_question(question, answer_mode)
        st.rerun()

    if active_chat()["messages"]:
        st.download_button(
            "Download this chat as Markdown",
            data=build_markdown_download(),
            file_name="healthcare_rag_lite_chat.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()