"""Streamlit login gate rendering for public demo mode."""
from __future__ import annotations

import random

import streamlit as st

from app.ui.demo_content import ASCII_POOL, HAIKU_POOL


def render_login_gate(demo_password: str) -> None:
    """Render a password gate. Stops execution until authenticated."""
    if "demo_ascii" not in st.session_state:
        st.session_state["demo_ascii"] = random.choice(ASCII_POOL)
    if "demo_haiku" not in st.session_state:
        st.session_state["demo_haiku"] = random.choice(HAIKU_POOL)

    st.markdown(
        """
        <style>
        .mac-bar {
            background: linear-gradient(#f7f7f7, #ececec);
            border: 1px solid #d8d8d8;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 14px;
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .mac-dot {width: 11px; height: 11px; border-radius: 50%; display: inline-block;}
        .mac-red {background: #ff5f57;}
        .mac-yellow {background: #febc2e;}
        .mac-green {background: #28c840;}
        .gate-title {font-weight: 600; margin-bottom: 4px;}
        .ascii-art {font-size: 12px; line-height: 1.25; color: #444; margin: 6px 0 10px 0;}
        .haiku {font-style: italic; color: #555; white-space: pre-line; margin: 8px 0 14px 0;}
        div[data-testid="stForm"] {border: 0 !important; padding: 0 !important;}
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #d8d8d8 !important;
            background: #fff !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, .12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1.0, 1.9, 1.0])
    with center:
        with st.container(border=True):
            st.markdown(
                """
                <div class="mac-bar">
                  <span class="mac-dot mac-red"></span>
                  <span class="mac-dot mac-yellow"></span>
                  <span class="mac-dot mac-green"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<p class="gate-title">Public demo access</p>', unsafe_allow_html=True)
            st.markdown(f"<pre class='ascii-art'>{st.session_state['demo_ascii']}</pre>", unsafe_allow_html=True)
            st.markdown(f"<p class='haiku'>{st.session_state['demo_haiku']}</p>", unsafe_allow_html=True)

            with st.form("demo_login_form", clear_on_submit=False):
                password = st.text_input("Password", type="password", key="demo_password_input")
                submit = st.form_submit_button("Enter demo", type="primary", use_container_width=True)
            if submit:
                if password == demo_password:
                    st.session_state["demo_authenticated"] = True
                    st.session_state.pop("demo_auth_error", None)
                    st.rerun()
                else:
                    st.session_state["demo_auth_error"] = "Incorrect password"
            if st.session_state.get("demo_auth_error"):
                st.error(st.session_state["demo_auth_error"])

