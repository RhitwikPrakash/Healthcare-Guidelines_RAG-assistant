import streamlit as st


_THEMES = {
    "blue": {
        "bg": "#07111f",
        "bg_soft": "#0a1728",
        "panel": "rgba(14, 27, 47, 0.88)",
        "panel_strong": "#10233b",
        "border": "rgba(151, 181, 214, 0.20)",
        "text": "#edf6ff",
        "muted": "#9fb5cb",
        "cyan": "#59d8e6",
        "blue": "#6ea8fe",
        "green": "#5ee3a3",
        "amber": "#f8c76d",
        "danger": "#ff8e9d",
        "sidebar": "linear-gradient(180deg, #0a1728 0%, #081321 100%)",
        "app_bg": "radial-gradient(circle at 15% 0%, rgba(51, 121, 179, .18), transparent 34%), radial-gradient(circle at 90% 15%, rgba(41, 178, 168, .12), transparent 30%), #07111f",
        "input_bg": "#081728",
    },
    "black": {
        "bg": "#050505",
        "bg_soft": "#0b0b0b",
        "panel": "rgba(20, 20, 20, 0.92)",
        "panel_strong": "#171717",
        "border": "rgba(255, 255, 255, 0.16)",
        "text": "#f7f7f7",
        "muted": "#b7b7b7",
        "cyan": "#c8c8c8",
        "blue": "#e5e5e5",
        "green": "#8ff0b6",
        "amber": "#ffd27a",
        "danger": "#ff9aaa",
        "sidebar": "linear-gradient(180deg, #090909 0%, #030303 100%)",
        "app_bg": "#050505",
        "input_bg": "#111111",
    },
    "white": {
        "bg": "#f6f8fb",
        "bg_soft": "#ffffff",
        "panel": "rgba(255, 255, 255, 0.96)",
        "panel_strong": "#ffffff",
        "border": "rgba(20, 35, 55, 0.16)",
        "text": "#102033",
        "muted": "#5c6b7c",
        "cyan": "#006d77",
        "blue": "#1f5fbf",
        "green": "#067647",
        "amber": "#8a5a00",
        "danger": "#b42318",
        "sidebar": "linear-gradient(180deg, #ffffff 0%, #f2f5f9 100%)",
        "app_bg": "linear-gradient(180deg, #ffffff 0%, #eef4fb 100%)",
        "input_bg": "#ffffff",
    },
}


def apply_styles(theme: str = "blue") -> None:
    values = _THEMES.get(theme, _THEMES["blue"])
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {values['bg']};
            --bg-soft: {values['bg_soft']};
            --panel: {values['panel']};
            --panel-strong: {values['panel_strong']};
            --border: {values['border']};
            --text: {values['text']};
            --muted: {values['muted']};
            --cyan: {values['cyan']};
            --blue: {values['blue']};
            --green: {values['green']};
            --amber: {values['amber']};
            --danger: {values['danger']};
            --input-bg: {values['input_bg']};
        }}

        .stApp {{
            background: {values['app_bg']};
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: {values['sidebar']};
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{ max-width: 1180px; padding-top: 1.4rem; padding-bottom: 7.5rem; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; color: var(--text); }}
        p, li, label, span, div {{ color: inherit; }}

        .hero {{
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, var(--panel), var(--panel-strong));
            border-radius: 22px;
            box-shadow: 0 20px 55px rgba(0,0,0,.18);
            margin-bottom: 1rem;
        }}
        .hero-kicker {{ color: var(--cyan); font-size: .78rem; text-transform: uppercase; letter-spacing: .18em; font-weight: 800; }}
        .hero h1 {{ margin: .35rem 0 .4rem; font-size: clamp(2rem, 4vw, 3.15rem); color: var(--text); }}
        .hero p {{ color: var(--muted); margin: 0; max-width: 820px; }}

        .safety-card {{
            border-left: 3px solid var(--amber);
            background: color-mix(in srgb, var(--amber) 12%, transparent);
            padding: .8rem 1rem;
            border-radius: 10px;
            color: var(--text);
            margin: .8rem 0 1rem;
        }}

        .top-panel, .doc-card, .source-card, .pipeline-card, .chat-list-card {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 14px;
            padding: .8rem .9rem;
            margin-bottom: .55rem;
        }}
        .top-panel {{ padding: 1rem; margin: 1rem 0; }}
        .doc-title, .source-title {{ font-weight: 750; color: var(--text); }}
        .doc-meta, .source-meta, .small-muted {{ color: var(--muted); font-size: .82rem; }}
        .status-ok {{ color: var(--green); font-weight: 800; }}
        .status-bad {{ color: var(--danger); font-weight: 800; }}
        .step-complete {{ color: var(--green); }}
        .step-running {{ color: var(--cyan); font-weight: 800; }}
        .step-pending {{ color: var(--muted); }}

        .chat-history-heading {{
            margin-top: .9rem;
            margin-bottom: .35rem;
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
        }}
        .selected-chat-note {{
            color: var(--cyan);
            font-size: .78rem;
            margin: -.25rem 0 .45rem 0;
        }}

        [data-testid="stChatMessage"] {{
            border: 1px solid var(--border);
            background: color-mix(in srgb, var(--panel-strong) 82%, transparent);
            border-radius: 17px;
            padding: .25rem .55rem;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 11px;
            border: 1px solid color-mix(in srgb, var(--blue) 40%, transparent);
            background: color-mix(in srgb, var(--panel-strong) 86%, transparent);
            color: var(--text);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--cyan);
            color: var(--text);
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid var(--border);
            background: color-mix(in srgb, var(--panel) 72%, transparent);
            border-radius: 12px;
        }}

        div[data-testid="stMetric"] {{
            background: color-mix(in srgb, var(--panel) 60%, transparent);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: .75rem .9rem;
        }}

        div[data-testid="stRadio"] label, div[data-testid="stSelectbox"] label {{ color: var(--text); }}
        div[data-baseweb="radio"] {{ gap: .3rem; }}

        .bottom-mode-panel {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 14px;
            padding: .75rem .9rem .35rem;
            margin-top: 1.1rem;
            margin-bottom: .35rem;
        }}
        .bottom-mode-title {{
            font-weight: 800;
            color: var(--text);
            margin-bottom: .15rem;
        }}
        .bottom-mode-help {{
            color: var(--muted);
            font-size: .82rem;
            margin-bottom: .25rem;
        }}

        textarea, input {{
            background-color: var(--input-bg) !important;
            color: var(--text) !important;
        }}

        .stAlert {{ color: var(--text); }}
        a {{ color: var(--cyan); }}
        hr {{ border-color: var(--border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- FINAL THEME OVERRIDE START ---
def apply_styles(theme: str | None = None) -> None:
    import streamlit as st

    theme = (theme or st.session_state.get("ui_theme") or "Blue").strip().lower()
    if theme not in {"blue", "black", "white"}:
        theme = "blue"

    palettes = {
        "blue": {
            "bg": "#061425",
            "sidebar": "#07182d",
            "card": "#0b1f38",
            "card2": "#102844",
            "text": "#f4f8ff",
            "muted": "#a8bad2",
            "border": "#29476b",
            "accent": "#52d5e8",
            "input": "#091b31",
        },
        "black": {
            "bg": "#050505",
            "sidebar": "#0b0b0b",
            "card": "#121212",
            "card2": "#1a1a1a",
            "text": "#f7f7f7",
            "muted": "#bdbdbd",
            "border": "#333333",
            "accent": "#7dd3fc",
            "input": "#111111",
        },
        "white": {
            "bg": "#f6f8fb",
            "sidebar": "#ffffff",
            "card": "#ffffff",
            "card2": "#edf3fb",
            "text": "#111827",
            "muted": "#4b5563",
            "border": "#cbd5e1",
            "accent": "#2563eb",
            "input": "#ffffff",
        },
    }

    p = palettes[theme]

    st.markdown(
        f"""
<style>
:root {{
    --rag-bg: {p["bg"]};
    --rag-sidebar: {p["sidebar"]};
    --rag-card: {p["card"]};
    --rag-card2: {p["card2"]};
    --rag-text: {p["text"]};
    --rag-muted: {p["muted"]};
    --rag-border: {p["border"]};
    --rag-accent: {p["accent"]};
    --rag-input: {p["input"]};
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
    background: var(--rag-bg) !important;
    color: var(--rag-text) !important;
}}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    background: var(--rag-sidebar) !important;
    color: var(--rag-text) !important;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background: transparent !important;
}}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
p, span, label, div, h1, h2, h3, h4, h5, h6, li {{
    color: var(--rag-text) !important;
}}

small, .small-text, .muted-text {{
    color: var(--rag-muted) !important;
}}

section[data-testid="stFileUploader"] {{
    background: var(--rag-card) !important;
    border: 1px solid var(--rag-border) !important;
    border-radius: 12px !important;
}}

input, textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div,
[data-testid="stChatInput"] textarea {{
    background: var(--rag-input) !important;
    color: var(--rag-text) !important;
    border-color: var(--rag-border) !important;
}}

button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {{
    background: var(--rag-card2) !important;
    color: var(--rag-text) !important;
    border: 1px solid var(--rag-border) !important;
}}

button:hover {{
    border-color: var(--rag-accent) !important;
    color: var(--rag-accent) !important;
}}

[data-testid="stChatMessage"],
.rag-card,
.rag-hero,
.rag-panel,
.rag-chat-card {{
    background: var(--rag-card) !important;
    color: var(--rag-text) !important;
    border-color: var(--rag-border) !important;
}}

hr {{
    border-color: var(--rag-border) !important;
}}

a {{
    color: var(--rag-accent) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )
# --- FINAL THEME OVERRIDE END ---

