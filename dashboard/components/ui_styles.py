import streamlit as st


def apply_custom_glassmorphism():
    """Injects the dashboard's visual design system.

    Deliberately restrained: a muted slate/navy palette, one accent color,
    subtle 1px borders instead of heavy blur/glow, and system-native
    typography -- aiming for "professional internal analytics tool" rather
    than a marketing landing page. Function name kept for backward
    compatibility with existing call sites (dashboard/app.py).
    """
    st.markdown("""
        <style>
        :root {
            --bg: #0b0e14;
            --panel: #10141d;
            --panel-border: rgba(255,255,255,0.08);
            --text-primary: #e6e9ef;
            --text-secondary: #9aa4b2;
            --accent: #4f8cff;
            --accent-soft: rgba(79,140,255,0.12);
            --good: #2fb673;
            --good-soft: rgba(47,182,115,0.12);
            --bad: #e5566d;
            --bad-soft: rgba(229,86,109,0.12);
            --warn: #d9a441;
            --warn-soft: rgba(217,164,65,0.12);
            --radius: 10px;
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp { background-color: var(--bg); color: var(--text-primary); }

        h1, h2, h3 { letter-spacing: -0.01em; font-weight: 650; }
        h1 { font-size: 1.65rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.05rem !important; }

        p, li, label, .stMarkdown { color: var(--text-primary); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }

        /* ---- Cards: flat panel, thin border, no blur/glow ---- */
        .app-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .kicker {
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 4px;
        }

        /* ---- Verdict badges ---- */
        .metric-badge-real, .metric-badge-fake, .metric-badge-neutral {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 650;
            font-size: 1.0rem;
            border: 1px solid transparent;
        }
        .metric-badge-real  { background: var(--good-soft); color: var(--good); border-color: var(--good); }
        .metric-badge-fake  { background: var(--bad-soft);  color: var(--bad);  border-color: var(--bad); }
        .metric-badge-neutral { background: var(--warn-soft); color: var(--warn); border-color: var(--warn); }

        /* ---- Small "which model served this score" tag ---- */
        .model-tag {
            display: inline-block;
            font-size: 0.72rem;
            color: var(--text-secondary);
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--panel-border);
            border-radius: 5px;
            padding: 2px 8px;
            margin-top: 4px;
        }

        /* ---- Empty state (no analysis / no image yet) ---- */
        .empty-state {
            border: 1px dashed var(--panel-border);
            border-radius: var(--radius);
            padding: 28px 20px;
            text-align: center;
            color: var(--text-secondary);
            background: rgba(255,255,255,0.015);
        }
        .empty-state .icon { font-size: 1.6rem; margin-bottom: 6px; }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background-color: var(--panel);
            border-right: 1px solid var(--panel-border);
        }
        section[data-testid="stSidebar"] * { color: var(--text-primary); }

        /* ---- Streamlit component polish ---- */
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            padding: 12px 16px;
        }
        div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

        .stButton>button, .stDownloadButton>button {
            border-radius: 6px;
            border: 1px solid var(--panel-border);
            font-weight: 600;
        }
        .stButton>button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px 6px 0 0;
            padding: 8px 14px;
        }

        div[data-testid="stDataFrame"] { border: 1px solid var(--panel-border); border-radius: var(--radius); }

        hr { border-color: var(--panel-border) !important; }
        </style>
    """, unsafe_allow_html=True)


def render_verdict_badge(label: str) -> str:
    """Returns HTML for a REAL/FAKE verdict badge, honoring the flat design system."""
    if label == "FAKE":
        return "<span class='metric-badge-fake'>&#9888; FAKE / MANIPULATED</span>"
    if label == "REAL":
        return "<span class='metric-badge-real'>&#10003; LIKELY AUTHENTIC</span>"
    return f"<span class='metric-badge-neutral'>{label}</span>"


def render_model_tag(model_name: str) -> str:
    """Small caption-style tag showing exactly which model/fallback served a score."""
    return f"<div class='model-tag'>Model: {model_name}</div>"


def render_empty_state(icon: str, message: str, sub: str = "") -> None:
    """Consistent 'nothing to show yet' panel, used instead of silently
    falling back to demo/sample data anywhere in the app."""
    sub_html = f"<div style='margin-top:6px;font-size:0.85rem;'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='empty-state'><div class='icon'>{icon}</div>{message}{sub_html}</div>",
        unsafe_allow_html=True,
    )
