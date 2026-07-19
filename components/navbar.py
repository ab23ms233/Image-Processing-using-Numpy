import streamlit as st
from utils.paths import LOGO

# st.set_page_config(
#     page_title="Image Processing Studio",
#     page_icon="../icons/logo.png",
#     initial_sidebar_state="collapsed",
# )

def render_navbar():
    st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inria+Sans:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Intel+One+Mono:ital,wght@0,300..700;1,300..700&display=swap');

                .nav-title {
                font-size: 18px;
                font-family: "Intel One Mono", monospace;
                color: white;
                }

                .nav-options {
                font-size: 18px;
                font-family: "Inria Sans", sans-serif;
                color: 'white';
                }
                </style>
                """, unsafe_allow_html=True
                )
    # Top padding
    # st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    left, spacer, right = st.columns([3, 4, 2], gap="medium")

    with left:
        logo, text = st.columns([1, 5], gap="small")

        with logo:
            st.image(str(LOGO), use_container_width=True)
        with text:
            st.markdown("<div class='nav-title'>Image Processing Studio</div>", unsafe_allow_html=True)

    with right:
        tech, github = st.columns(2, gap="small")

        with tech:
            st.markdown("<div class='nav-options'>Tech Stack</div>", unsafe_allow_html=True)
        with github:
            st.markdown("<div class='nav-options'>GitHub</div>", unsafe_allow_html=True)

    