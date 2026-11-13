from __future__ import annotations

import hashlib

from history_store import HistoryStore, history_owner_from_google_claims
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st

from llm_client import (
    RATE_LIMIT_MESSAGE,
    ProviderRateLimitError,
    ProviderUnavailableError,
    generate_answer,
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
    parse_answer_evidence,
    retrieve_chunks,
    strip_inline_citations,
)
from request_gate import wait_for_request_slot


st.set_page_config(
    page_title="Healthcare Guidelines RAG Assistant",
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


# UI preferences are session-specific; no changes to Google OAuth or RAG logic.
# Streamlit's config.toml provides the initial base theme. These custom
# user-selected palettes are applied as CSS on every script rerun.
UI_THEMES = {
    "Black": {
        "bg": "#0a0c0f",
        "sidebar": "#11151b",
        "panel": "#1b2028",
        "border": "#38414d",
        "text": "#f4f6f8",
        "muted": "#b5becb",
        "accent": "#fb7185",
        "accent_text": "#171016",
    },
    "Dark Blue": {
        "bg": "#071827",
        "sidebar": "#0a2137",
        "panel": "#102c47",
        "border": "#315a7d",
        "text": "#f1f7fc",
        "muted": "#b5c9dc",
        "accent": "#38bdf8",
        "accent_text": "#061827",
    },
    "Medium–Dark Brown": {
        "bg": "#211711",
        "sidebar": "#2b1f19",
        "panel": "#3c2a21",
        "border": "#70513e",
        "text": "#fff4e8",
        "muted": "#dfc8b6",
        "accent": "#e8ac70",
        "accent_text": "#2b190e",
    },
}


def apply_ui_theme() -> None:
    st.session_state.setdefault("ui_theme", "Dark Blue")
    palette = UI_THEMES.get(st.session_state.ui_theme, UI_THEMES["Dark Blue"])
    values = "\n".join(
        f"--rag-{name.replace('_', '-')}: {color};"
        for name, color in palette.items()
    )
    st.markdown(
        "<style>\n"
        ":root {\n" + values + "\n}\n"
        """
        /* Page surface, header and sidebar */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: var(--rag-bg) !important;
            color: var(--rag-text) !important;
        }
        [data-testid="stHeader"] {
            background-color: var(--rag-bg) !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background-color: var(--rag-sidebar) !important;
            color: var(--rag-text) !important;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--rag-border) !important;
        }
        /* Content containers and evidence panels */
        .rag-card, .auth-card, .reference-list,
        [data-testid="stChatMessage"],
        [data-testid="stExpander"] {
            background-color: var(--rag-panel) !important;
            border-color: var(--rag-border) !important;
        }
        .reference-list { border-left-color: var(--rag-accent) !important; }
        .small-muted, .stCaption { color: var(--rag-muted) !important; }
        .stApp [data-testid="stMarkdownContainer"],
        .stApp h1, .stApp h2, .stApp h3,
        .stApp label { color: var(--rag-text) !important; }
        .stApp a { color: var(--rag-accent) !important; }
        /* Input widgets, including the theme selector and chat input */
        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div,
        .stApp [data-testid="stTextInput"] input,
        .stApp [data-testid="stChatInput"] textarea,
        .stApp [data-testid="stFileUploaderDropzone"] {
            background-color: var(--rag-panel) !important;
            border-color: var(--rag-border) !important;
            color: var(--rag-text) !important;
        }
        .stApp input, .stApp textarea { color: var(--rag-text) !important; }
        .stApp [data-testid="stButton"] button,
        .stApp [data-testid="stDownloadButton"] button,
        .stApp [data-testid="stFormSubmitButton"] button {
            background-color: var(--rag-panel) !important;
            color: var(--rag-text) !important;
            border-color: var(--rag-border) !important;
        }
        .stApp button[kind="primary"] {
            background-color: var(--rag-accent) !important;
            color: var(--rag-accent-text) !important;
            border-color: var(--rag-accent) !important;
        }
        """
        "</style>",
        unsafe_allow_html=True,
    )


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_chat(title: str = "New chat") -> dict:
    now = _utc_now()
    return {
        "id": str(uuid4()), "title": title, "messages": [], "notice": None,
        "created_at": now, "updated_at": now,
    }


def history_context():
    """Only trust Google ID token claims supplied by Streamlit OIDC."""
    claims = st.user.to_dict() if st.user.is_logged_in else {}
    owner = history_owner_from_google_claims(claims)
    return owner, HistoryStore.from_config()


def _save_chat(chat: dict) -> None:
    owner, store = history_context()
    if not owner or not store.is_configured:
        return
    try:
        store.upsert_chat(owner, chat)
        st.session_state["_history_error"] = None
    except Exception:
        st.session_state["_history_error"] = "Chat history could not be saved."


def _remove_saved_chat(chat_id: str) -> None:
    owner, store = history_context()
    if not owner or not store.is_configured:
        return
    try:
        store.delete_chat(owner, chat_id)
        st.session_state["_history_error"] = None
    except Exception:
        st.session_state["_history_error"] = "Chat history could not be deleted from storage."


def init_state() -> None:
    st.session_state.setdefault("processed_key", None)
    st.session_state.setdefault("rag_data", None)
    owner, store = history_context()
    # Do not reuse one logged-in user's session state for a different Google user.
    identity = owner or f"session-only:{st.user.to_dict().get('email', '')}"
    if st.session_state.get("_loaded_for") != identity:
        st.session_state.chats = {}
        st.session_state.active_chat_id = None
        st.session_state.processed_key = None
        st.session_state.rag_data = None
        st.session_state["_history_error"] = None
        if owner and store.is_configured:
            try:
                saved = store.load_chats(owner)
                st.session_state.chats = {item["id"]: item for item in saved}
            except Exception:
                st.session_state["_history_error"] = "Saved chat history is temporarily unavailable."
        if st.session_state.chats:
            ordered = sorted(
                st.session_state.chats.values(),
                key=lambda c: c.get("updated_at", ""), reverse=True,
            )
            st.session_state.active_chat_id = ordered[0]["id"]
        else:
            chat = _make_chat()
            st.session_state.chats = {chat["id"]: chat}
            st.session_state.active_chat_id = chat["id"]
        st.session_state["_loaded_for"] = identity


def active_chat() -> dict:
    chat_id = st.session_state.active_chat_id
    chat = st.session_state.chats.get(chat_id)
    if chat is None:
        chat = _make_chat()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.active_chat_id = chat["id"]
    return chat


def touch_chat(chat: dict) -> None:
    chat["updated_at"] = _utc_now()


def create_new_chat() -> None:
    chat = _make_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.active_chat_id = chat["id"]
    st.session_state.processed_key = None
    st.session_state.rag_data = None


def clear_login_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def render_auth_screen() -> None:
    st.markdown(
        "<h1 class='auth-title'>Healthcare Guidelines RAG Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Sign in with your Google account to access PDF research and chat tools.")

    left, centre, right = st.columns([1, 1.25, 1])
    with centre:
        st.selectbox("Interface color", list(UI_THEMES), key="ui_theme")
        st.subheader("Secure Google sign-in")
        st.button(
            "Continue with Google",
            type="primary",
            use_container_width=True,
            on_click=st.login,
        )

    st.caption("Educational use only. This application does not diagnose or prescribe.")


def render_sidebar() -> None:
    user = st.user.to_dict()
    st.sidebar.title("Healthcare Guidelines RAG Assistant")
    st.sidebar.selectbox("Interface color", list(UI_THEMES), key="ui_theme")
    st.sidebar.caption(f"Signed in as **{user.get('name') or user.get('email') or 'Google user'}**")
    st.sidebar.caption(str(user.get("email", "")))

    if st.sidebar.button("Log out", use_container_width=True):
        clear_login_state()
        st.logout()

    st.sidebar.divider()

    st.sidebar.markdown("**About this assistant**")
    st.sidebar.markdown(
        "<div style='text-align:left;line-height:1.6;font-size:0.90rem;'>"
        "Explore uploaded healthcare guidelines, ask research questions, "
        "and receive concise, evidence-grounded answers with PDF references."
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    if st.sidebar.button("New chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    if st.sidebar.button("Clear processed PDFs", use_container_width=True):
        st.session_state.processed_key = None
        st.session_state.rag_data = None
        # Clearing the active PDF index does not erase stored conversation history.
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
                _save_chat(chat)
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
            # PDFs are not persisted; prevent using another conversation's index.
            st.session_state.rag_data = None
            st.session_state.processed_key = None
            st.rerun()

    owner, store = history_context()
    if owner and store.is_configured:
        st.sidebar.caption(
            "Chat messages and selected evidence excerpts are stored in Supabase for "
            "72 hours after their last update. PDFs are not saved; re-upload to continue."
        )
        if st.session_state.get("_history_error"):
            st.sidebar.warning(st.session_state["_history_error"])
    else:
        st.sidebar.caption(
            "History storage is not configured. Chats will disappear when "
            "this session ends."
        )
    if st.sidebar.button("Delete selected chat", use_container_width=True):
        deleted_id = st.session_state.active_chat_id
        _remove_saved_chat(deleted_id)
        if not st.session_state.get("_history_error"):
            st.session_state.chats.pop(deleted_id, None)
            if st.session_state.chats:
                st.session_state.active_chat_id = next(iter(st.session_state.chats))
            else:
                create_new_chat()
            st.session_state.rag_data = None
            st.session_state.processed_key = None
            st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("Only chats/evidence snippets are retained, not PDF files.")
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
            abstained = (
                "the uploaded document does not provide enough evidence to answer this"
                in message.get("content", "").casefold()
            )
            render_reference_block([] if abstained else message.get("references", []))
            # Provider/model remain internal diagnostics; never show them to readers.
            evidence_cards = [] if abstained else message.get("evidence_cards", [])
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

    try:
        with st.status("Researching and synthesizing the answer", expanded=True) as status:
            status.write("Relevant PDF evidence has been retrieved and ranked.")
            countdown_text = st.empty()
            countdown_progress = st.progress(0.0)
            wait_for_request_slot(
                _countdown_callback(countdown_progress, countdown_text),
                minimum_delay=0.0,
                maximum_delay=0.0,
                minimum_spacing=3.0,
            )
            token_limits = {"Fast": 1100, "Smart Auto": 1600, "Deep Research": 2600}
            answer, provider, model = generate_answer(
                prompt, max_tokens=token_limits.get(answer_mode, 1600)
            )
            status.update(
                label="Grounded answer prepared",
                state="complete",
                expanded=False,
            )
    except ProviderRateLimitError:
        chat["notice"] = RATE_LIMIT_MESSAGE
        touch_chat(chat)
        return False
    except ProviderUnavailableError:
        chat["notice"] = "Answer generation is temporarily unavailable. Please try again shortly."
        touch_chat(chat)
        return False
    except Exception:
        chat["notice"] = "The answer could not be generated. Please try again shortly."
        touch_chat(chat)
        return False

    clean_answer, supported_cards = parse_answer_evidence(answer, evidence_cards)
    references = build_reference_lines(supported_cards)
    chat["messages"].append(
        {
            "role": "assistant",
            "content": clean_answer,
            "provider": provider,
            "model": model,
            "references": references,
            "evidence_cards": supported_cards,
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
        abstained = "the uploaded document does not provide enough evidence to answer this" in message.get("content", "").casefold()
        references = [] if abstained else message.get("references", [])
        if references:
            lines.append("### References")
            lines.extend(f"- {item}" for item in references)
            lines.append("")

        evidence_cards = [] if abstained else message.get("evidence_cards", [])
        if evidence_cards:
            lines.append("### Evidence")
            for card in evidence_cards:
                lines.append(
                    f"- Evidence {card['label']}: {card['source']}, page {card['page']}"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    apply_ui_theme()
    if not st.user.is_logged_in:
        render_auth_screen()
        st.stop()

    init_state()
    render_sidebar()

    st.title("Healthcare Guidelines RAG Assistant")
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

        if active_chat()["messages"]:
            create_new_chat()  # Preserve earlier chat before changing PDF corpus.
        st.session_state.processed_key = combined_key
        st.session_state.rag_data = rag_data
        active_chat()["notice"] = None

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
        _save_chat(active_chat())
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