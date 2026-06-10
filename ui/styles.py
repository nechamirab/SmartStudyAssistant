from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F7F6F2; color: #172033; }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #172033; letter-spacing: 0; }
        [data-testid="stSidebar"] { display: none; }
        .top-nav {
            background: #FFFEFB;
            border: 1px solid #E5E0D6;
            border-radius: 12px;
            padding: .75rem .9rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(23, 32, 51, .07);
            position: sticky;
            top: 1rem;
            z-index: 20;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: .45rem;
            min-height: 2.4rem;
            color: #3730A3;
            font-weight: 800;
            font-size: 1.05rem;
            white-space: nowrap;
        }
        .nav-active {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.45rem;
            border-radius: 999px;
            background: #EDE9FE;
            color: #3730A3;
            font-weight: 800;
            border: 1px solid #C4B5FD;
        }
        .status-bar {
            display: flex;
            flex-wrap: wrap;
            gap: .75rem;
            align-items: center;
            background: #FFFEFB;
            border: 1px solid #E5E0D6;
            border-radius: 12px;
            padding: .7rem .95rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(23, 32, 51, .045);
            color: #626B7F;
            font-size: .9rem;
        }
        .status-bar strong { color: #172033; }
        .hero-card {
            background: linear-gradient(135deg, #3730A3 0%, #047857 100%);
            color: #FFFFFF;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 34px rgba(55, 48, 163, .16);
        }
        .hero-card h1 { color: #FFFFFF; margin: 0; font-size: 1.55rem; }
        .hero-card p { color: #E0F2FE; margin: .35rem 0 0; }
        .card {
            background: #FFFEFB;
            border: 1px solid #E5E0D6;
            border-radius: 12px;
            padding: 1.05rem 1.1rem;
            margin-bottom: .85rem;
            box-shadow: 0 12px 28px rgba(23, 32, 51, .07);
        }
        .card-title { color: #3730A3; font-weight: 700; font-size: 1.04rem; margin-bottom: .35rem; }
        .muted { color: #626B7F; font-size: .9rem; }
        .roadmap-card {
            border-left: 5px solid #047857;
            position: relative;
        }
        .roadmap-index {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #3730A3;
            color: white;
            font-weight: 800;
            margin-right: .45rem;
        }
        .objective-list { margin: .45rem 0 .2rem 1.15rem; color: #172033; }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: .18rem .55rem;
            font-weight: 700;
            font-size: .75rem;
            margin-right: .25rem;
        }
        .badge-primary { background: #EDE9FE; color: #3730A3; }
        .badge-secondary { background: #F1F5F9; color: #475569; }
        .badge-accent { background: #CCFBF1; color: #0F766E; }
        .badge-current { background: #EDE9FE; color: #3730A3; }
        .badge-next { background: #CCFBF1; color: #0F766E; }
        .badge-success { background: #DCFCE7; color: #15803D; }
        .badge-warning { background: #FEF3C7; color: #B45309; }
        .badge-error { background: #FEE2E2; color: #B91C1C; }
        .tag {
            display: inline-block;
            background: #D1FAE5;
            color: #047857;
            border-radius: 999px;
            padding: .16rem .48rem;
            margin: .16rem .18rem .16rem 0;
            font-size: .75rem;
        }
        .prompt-card {
            background: #FFFEFB;
            border: 1px solid #E5E0D6;
            border-radius: 12px;
            padding: .8rem .9rem;
            min-height: 4.2rem;
            box-shadow: 0 8px 18px rgba(23, 32, 51, .05);
        }
        .source-label { color: #626B7F; font-size: .85rem; }
        div.stButton > button {
            border-radius: 999px;
            border-color: #E5E0D6;
            color: #172033;
        }
        div.stButton > button[kind="primary"] { background: #3730A3; border-color: #3730A3; }
        div.stButton > button:hover { border-color: #047857; color: #172033; }
        div[data-testid="stProgress"] > div > div > div { background-color: #047857; }
        div[data-testid="stAlert"] {
            border-radius: 10px;
            border-color: #E5E0D6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
