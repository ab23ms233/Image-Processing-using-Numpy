import streamlit as st
from utils.paths import LOGO_REMOVEBG
from Class_Actions import Actions

def render_navbar():
    st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inria+Sans:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Intel+One+Mono:ital,wght@0,300..700;1,300..700&display=swap');

                .nav-title {
                font-size: 18px;
                font-family: "Intel One Mono", monospace;
                }

                .nav-option {
                margin-top: 5px;
                font-size: 18px;
                font-family: "Inria Sans", sans-serif;
                transition: all 0.3s ease;
                }

                .nav-option:hover {
                color: #04A7FF;
                }

                .nav-link {
                text-decoration: none !important;
                color: white !important;
                }

                .st-key-nav_home button {
                background: transparent;
                border: none;
                }

                .st-key-nav_home button p {
                font-size: 18px;
                font-family: "Inria Sans", sans-serif;
                transition: all 0.3s ease;
                }

                .st-key-nav_home button p:hover {
                color: #04A7FF;
                }
                </style>
                """, unsafe_allow_html=True
                )
    # Top padding
    # st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    left, spacer, right = st.columns([3, 4, 1.5], gap="medium")

    with left:
        logo, text = st.columns([1, 5], gap="small")

        with logo:
            img = Actions.load_cached_image(str(LOGO_REMOVEBG))
            st.image(img, use_container_width=True)
        with text:
            st.markdown("<div class='nav-title'>Image Processing Studio</div>", unsafe_allow_html=True)

    with right:
        home, github = st.columns(2, gap="small")

        with home:
            if st.button("Home", key="nav_home"):
                st.session_state.clear()
                st.session_state["page"] = "home"
                st.rerun()
        with github:
            st.markdown("""<a href='https://github.com/ab23ms233/Image-Processing-using-Numpy' target='_blank' class="nav-link">
                                <div class='nav-option'>GitHub</div>
                        </a>""", unsafe_allow_html=True)

    