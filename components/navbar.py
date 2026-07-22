import streamlit as st
from utils.paths import LOGO_REMOVEBG
from Class_Actions import Actions

def render_navbar():
    st.markdown("""
                <style>

                .nav-title {
                font-size: 1rem;
                font-family: "Intel One Mono", monospace;
                }

                .nav-option {
                margin-top: 5px;
                font-size: 1.1rem;
                font-family: "Inria Sans", sans-serif;
                transition: all 0.3s ease;
                }

                .nav-option:hover {
                color: #FF4B4B;
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
                font-size: 1.1rem;
                font-family: "Inria Sans", sans-serif;
                transition: all 0.3s ease;
                }
                </style>
                """, unsafe_allow_html=True)
    # Top padding
    # st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    left, spacer, right = st.columns([3, 4, 2], gap="medium")

    with left:
        logo, text = st.columns([1, 5], gap="small")

        with logo:
            img = Actions.load_cached_image(str(LOGO_REMOVEBG))
            st.image(img, use_container_width=True)
        with text:
            st.markdown("<div class='nav-title'>Image Processing Studio</div>", unsafe_allow_html=True)

    with right:
        home, github = st.columns([1, 1.5], gap="medium")

        with home:
            if st.button("Home", key="nav_home"):
                st.session_state.clear()
                st.session_state["page"] = "home"
                st.rerun()
        with github:
            st.markdown("""<a href='https://github.com/ab23ms233/Image-Processing-using-Numpy' target='_blank' class="nav-link">
                                <div class='nav-option'>View on GitHub</div>
                        </a>""", unsafe_allow_html=True)

    