import streamlit as st
from Class_Actions import Actions

from components.navbar import render_navbar
from components.img_ops import render_img_ops
from components.export import render_img_info, render_export_settings

def editor_page(max_img_size):
    # Navigation menu
    render_navbar()
    # Padding
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

    tabs = ["Transform", "Filters", "Colors"]
    # Initializing states and variables
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "redo_stack" not in st.session_state:
        st.session_state["redo_stack"] = []
    if "selected_tool" not in st.session_state:
        st.session_state["selected_tool"] = None
    if "export_mode" not in st.session_state:
        st.session_state["export_mode"] = False

    # start = time.perf_counter()
    # Image preview and operations section
    render_img_ops(tabs)
    st.divider()

    # Display export settings if user exports image
    if st.session_state["export_mode"]:
        st.session_state["image_arr"] = Actions.export_image()
        render_img_info(st.session_state["image_arr"])
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        render_export_settings()
    
    # Uploading new image
    new_image = st.file_uploader("Change Picture", type=["jpeg", "png", "jpg", "bmp"], key="new_image")
    if new_image:
        st.session_state["image"] = Actions.image_uploaded(new_image, max_img_size)
        st.rerun()