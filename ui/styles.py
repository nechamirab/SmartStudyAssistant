from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #F4F7FB;
            --surface: #FFFFFF;
            --surface-soft: #EEF5F9;
            --text-main: #102033;
            --text-muted: #526173;
            --border: #D5DEE8;
            --primary: #0B3D91;
            --primary-hover: #082F6F;
            --primary-soft: #E7F0FF;
            --accent: #007C72;
            --accent-soft: #DDF7F3;
            --warning-bg: #FFF4D6;
            --warning-text: #825300;
            --danger-bg: #FFE3E3;
            --danger-text: #A20F18;
            --shadow: 0 10px 24px rgba(16, 32, 51, .08);
        }
        .stApp { background: var(--app-bg); color: var(--text-main); }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: var(--text-main); letter-spacing: 0; }
        [data-testid="stSidebar"] { display: none; }
        .top-nav {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .75rem .9rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
            position: sticky;
            top: 1rem;
            z-index: 20;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: .45rem;
            min-height: 2.4rem;
            color: var(--primary);
            font-weight: 800;
            font-size: 1.05rem;
            white-space: nowrap;
        }
        .nav-active {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.45rem;
            border-radius: 8px;
            background: var(--primary-soft);
            color: var(--primary);
            font-weight: 800;
            border: 1px solid #AFC8F2;
        }
        .status-bar {
            display: flex;
            flex-wrap: wrap;
            gap: .75rem;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .7rem .95rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(16, 32, 51, .05);
            color: var(--text-muted);
            font-size: .9rem;
        }
        .status-bar strong { color: var(--text-main); }
        .hero-card {
            background: linear-gradient(135deg, #0B3D91 0%, #007C72 100%);
            color: #FFFFFF;
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(11, 61, 145, .18);
        }
        .hero-card h1 { color: #FFFFFF; margin: 0; font-size: 1.55rem; }
        .hero-card p { color: #EAF6FF; margin: .35rem 0 0; }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            margin-bottom: .85rem;
            box-shadow: var(--shadow);
        }
        .card-title { color: var(--primary); font-weight: 700; font-size: 1.04rem; margin-bottom: .35rem; }
        .muted { color: var(--text-muted); font-size: .9rem; }
        .roadmap-card {
            border-left: 5px solid var(--accent);
            position: relative;
        }
        .roadmap-index {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--primary);
            color: white;
            font-weight: 800;
            margin-right: .45rem;
        }
        .objective-list { margin: .45rem 0 .2rem 1.15rem; color: var(--text-main); }
        .badge {
            display: inline-block;
            border-radius: 8px;
            padding: .18rem .55rem;
            font-weight: 700;
            font-size: .75rem;
            margin-right: .25rem;
        }
        .badge-primary { background: var(--primary-soft); color: var(--primary); }
        .badge-secondary { background: #E9EEF5; color: #38485A; }
        .badge-accent { background: var(--accent-soft); color: #005F57; }
        .badge-current { background: var(--primary-soft); color: var(--primary); }
        .badge-next { background: var(--accent-soft); color: #005F57; }
        .badge-success { background: #DFF5E6; color: #146534; }
        .badge-warning { background: var(--warning-bg); color: var(--warning-text); }
        .badge-error { background: var(--danger-bg); color: var(--danger-text); }
        .tag {
            display: inline-block;
            background: var(--accent-soft);
            color: #005F57;
            border-radius: 8px;
            padding: .16rem .48rem;
            margin: .16rem .18rem .16rem 0;
            font-size: .75rem;
        }
        .prompt-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .8rem .9rem;
            min-height: 4.2rem;
            box-shadow: 0 8px 18px rgba(16, 32, 51, .05);
            color: var(--text-main);
        }
        .source-label { color: var(--text-muted); font-size: .85rem; }

        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            min-height: 2.45rem;
            border-radius: 8px;
            border: 1px solid #B9C7D6;
            background: var(--surface);
            color: var(--text-main);
            font-weight: 700;
            box-shadow: 0 1px 2px rgba(16, 32, 51, .06);
            transition: background-color .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease;
        }
        div.stButton > button p,
        div.stDownloadButton > button p,
        div[data-testid="stFormSubmitButton"] > button p {
            color: inherit;
            font-weight: inherit;
        }
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: var(--primary);
            border-color: var(--primary);
            color: #FFFFFF;
            box-shadow: 0 8px 18px rgba(11, 61, 145, .22);
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: var(--accent);
            background: #F5FBFA;
            color: var(--text-main);
            box-shadow: 0 4px 12px rgba(0, 124, 114, .12);
        }
        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background: var(--primary-hover);
            border-color: var(--primary-hover);
            color: #FFFFFF;
        }
        div.stButton > button:focus,
        div.stDownloadButton > button:focus,
        div[data-testid="stFormSubmitButton"] > button:focus {
            outline: 3px solid rgba(0, 124, 114, .28);
            outline-offset: 2px;
            box-shadow: none;
        }
        div.stButton > button:disabled,
        div.stDownloadButton > button:disabled,
        div[data-testid="stFormSubmitButton"] > button:disabled {
            background: #E6ECF2;
            border-color: #D4DDE7;
            color: #6A7788;
            box-shadow: none;
        }
        div[data-testid="stProgress"] > div > div > div { background-color: var(--accent); }
        div[data-testid="stAlert"] {
            border-radius: 8px;
            border-color: var(--border);
        }
        div[data-testid="stFileUploader"] section {
            border-color: var(--border);
            background: var(--surface);
            border-radius: 8px;
        }
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            color: var(--text-main);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
