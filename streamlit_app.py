import streamlit as st
import time
# from PIL import Image

from app_pages.home import home_page
from app_pages.editor import editor_page
from utils.paths import LOGO

# Page configuration
st.set_page_config(
    page_title="Image Processing Studio",
    page_icon=str(LOGO),
    layout="wide",
    initial_sidebar_state="collapsed")

# Maximum image dimension
MAX_SIZE = 1200

# Hide streamlit default header, menu and footer
st.markdown("""
<style>
    header {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
            
    .block-container {
            padding-top: 0.5rem;
            padding-right: 5rem;
            padding-left: 5rem;
            }
</style>
""", unsafe_allow_html=True)

# Page styles
st.markdown(
    """
    <style>
    .stApp {
        background: #020617;
        color: #e2e8f0;
    }""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "editor":
    editor_page(MAX_SIZE)
elif st.session_state["page"] == "home":
    start = time.perf_counter()
    home_page(MAX_SIZE)
    print(f"Time for rendering homepage: {time.perf_counter()-start}")