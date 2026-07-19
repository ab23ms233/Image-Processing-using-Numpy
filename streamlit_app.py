import streamlit as st
from PIL import Image
import numpy as np
from Class_Actions import Actions
import time

from components.navbar import render_navbar
from components.hero import render_hero
from components.features_tech import render_features_tech

# Page configuration
st.set_page_config(
    page_title="Image Processing Studio",
    page_icon="icons/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed")

# Maximum image dimension
MAX_SIZE = 1200

tabs = ["Transform", "Filters", "Colors"]

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

# Custom dark theme styling for the hero and cards
st.markdown(
    """
    <style>
    .stApp {
        background: #020617;
        color: #e2e8f0;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 32px;
        padding: 36px;
    }""", unsafe_allow_html=True)

# LANDING PAGE
def landing_page():
    print("Landing page")

    render_navbar()
    render_hero()
    st.markdown("<div style='padding-top: 1rem'></div>", unsafe_allow_html=True)
    render_features_tech()
    # HERO SECTION
    # hero_left, hero_right = st.columns([1.5, 1], gap="medium")

    # # Left side
    # with hero_left:
    #     start = time.perf_counter()
    #     # Heading
    #     st.markdown("<h1 style='color:#f8fafc; margin-bottom: 0.25rem;'>Image Processing Studio</h1>", unsafe_allow_html=True)
    #     # Description
    #     st.markdown("<p style='color:#cbd5e1; font-size:1.2rem; margin-top:0; max-width:650px;'>Upload, edit and download images with fast one-click controls in a clean dark interface.</p>",
    #                 unsafe_allow_html=True)
    #     # st.markdown("<p style='color:#94a3b8; margin-top:1rem; max-width:650px;'>Rotate, flip, blur, sharpen and download your final image quickly with instant preview and responsive controls.</p>", unsafe_allow_html=True)
        
    #     # File upload button
    #     file_uploader_col, right = st.columns([3, 1])
    #     with file_uploader_col:
    #         uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "bmp"])
    #     print(f"Time for rendering hero_left: {time.perf_counter() - start}")
    
    # # Right side
    # with hero_right:
    #     # start = time.perf_counter()

    #     # Image
    #     @st.cache_data
    #     def load_image():
    #         image = Image.open("images/deep-mind-compressed.webp").convert("RGB")
    #         return image
        
    #     start = time.perf_counter()
    #     hero_img = load_image()
    #     print(f"Time for loading image: {time.perf_counter()-start}")
    #     start = time.perf_counter()
    #     st.image(hero_img, use_container_width=True)
    #     print(f"Time for rendering image: {time.perf_counter()-start}")
    #     # Description
    #     st.markdown("<p style='text-align: center;'>Photo by <a href='https://unsplash.com/@googledeepmind' target='_blank'>Google DeepMind</a></p>",
    # unsafe_allow_html=True)
        # print(f"Time for rendering hero_right: {time.perf_counter() - start}")
    
    start = time.perf_counter()
    # Feature cards
    # st.markdown("---")
    # card_1, card_2, card_3, card_4 = st.columns(4, gap="small")
    # with card_1:
    #     st.markdown("<div class='feature-card feature-card-blue'><h3>⚡Fast Preview</h3><p>Apply filters instantly.</p></div>", unsafe_allow_html=True)
    # with card_2:
    #     st.markdown("<div class='feature-card feature-card-teal'><h3>🔄 Rotate & Flip</h3><p>One-click orientation controls.</p></div>", unsafe_allow_html=True)
    # with card_3:
    #     st.markdown("<div class='feature-card feature-card-purple'><h3>🎨 Filters</h3><p>Grayscale, binarize, sharpen, etc.</p></div>", unsafe_allow_html=True)
    # with card_4:
    #     st.markdown("<div class='feature-card feature-card-cyan'><h3>⬇ Download</h3><p>Save your edited image as a PNG.</p></div>", unsafe_allow_html=True)

    # st.markdown("---")
    # print(f"Time for rendering feature_cards: {time.perf_counter() - start}")

    # If file is uploaded
    # if uploaded_file is not None:
    #     st.session_state["image"] = Actions.image_uploaded(uploaded_file, MAX_SIZE)
    #     print(st.session_state["image_metadata"]["format"])
    #     st.rerun()


# EDITOR PAGE
def editor_page():
    # print("Editor page")
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "redo_stack" not in st.session_state:
        st.session_state["redo_stack"] = []
    if "selected_tool" not in st.session_state:
        st.session_state["selected_tool"] = None
    if "export_clicked" not in st.session_state:
        st.session_state["export_clicked"] = False

    original, processed, editor_pane = st.columns([1, 1, 1.5], gap="medium")
    # start = time.perf_counter()

    # Original image
    with original:
        st.markdown("<h3 style='text-align: center'>Original</h3>", unsafe_allow_html=True)
        st.image(Image.fromarray(st.session_state["ori_img"].img_arr), width=360)
    # Processed image
    with processed:
        st.markdown("<h3 style='text-align: center'>Processed</h3>", unsafe_allow_html=True)
        st.image(Image.fromarray(st.session_state["curr_img"].img_arr), width=360)
    st.markdown("---")

    # print(f"Time for displaying images: {time.perf_counter() - start}")
    
    # Editor pane
    with editor_pane:
        # Different tabs for image processing tools
        selected_tab = st.segmented_control("Category", tabs, label_visibility="collapsed", default="Transform")
        # start = time.perf_counter()

        # TRANSFORM TAB
        if selected_tab == "Transform":
            st.subheader("Transform")

            # Rotate
            rot_c_col, rot_ac_col = st.columns(2, gap="medium")
            Actions.button_renderer(rot_c_col, "⟳ Rotate Clockwise", "rotate_clockwise", lambda p: p.rotate_img(num=1), "Rotate Clockwise")
            Actions.button_renderer(rot_ac_col, "⟲ Rotate Anti-Clockwise", "rotate_anti_clockwise", lambda p: p.rotate_img(num=-1), "Rotate Anti-clockwise")
            # Flip
            fliph_col, flipv_col = st.columns(2, gap="medium")
            Actions.button_renderer(fliph_col, "↔️ Flip Horizontal", "flip_horizontal", lambda p: p.flip_img(plane='h'), "Flip Horizontal")
            Actions.button_renderer(flipv_col, "↕️ Flip Vertical", "flip_vertical", lambda p: p.flip_img(plane='v'), "Flip Vertical")

        # FILTERS TAB
        if selected_tab == "Filters":
            st.subheader("Filters")
            sharpen_col, blur_col, edge_col = st.columns(3, gap="medium")

            # Sharpen
            with sharpen_col:
                if st.button("Sharpen", use_container_width=True, key="sharpen"):
                    st.session_state["selected_tool"] = "Sharpen"
            # Blur
            with blur_col:
                if st.button("Blur", use_container_width=True, key="blur"):
                    st.session_state["selected_tool"] = "Blur"
            # Edge detection
            Actions.button_renderer(edge_col, "Edge Detection", "edge_detection", lambda p: p.edge_detection(), "Edge Detection")

        # COLORS
        if selected_tab == "Colors":
            st.subheader("Colors")
            negative_col, grayscale_col, binarize_col = st.columns(3, gap="medium")

            # Negative
            Actions.button_renderer(negative_col, "Negative", "negative", lambda p: p.negative(), "Negative")
            # Grayscale
            Actions.button_renderer(grayscale_col, "Grayscale", "grayscale", lambda p: p.grayscale(), "Grayscale")
            # Binarize
            with binarize_col:
                if st.button("Binarize", use_container_width=True, key="binarize"):
                    st.session_state["selected_tool"] = "Binarize"
        
        # Render sliders if required by operation
        Actions.tool_renderer(selected_tab)
        st.markdown("-----")
        # print(f"Time for rendering tabs: {time.perf_counter()-start}")

        # Displaying HISTORY
        Actions.display_history()

        undo_col, redo_col, reset_col, export_col = st.columns([1, 1, 1.5, 1], gap="small")

        # Undo
        with undo_col:
            if st.button("Undo", use_container_width=True, key="undo"):
                Actions.undo()
        # Redo
        with redo_col:
            if st.button("Redo", use_container_width=True, key="redo"):
                Actions.redo()

        # Reset to original
        with reset_col:
            if st.button("Reset to Original", use_container_width=True, key="reset"):
                st.session_state["curr_img"] = st.session_state["ori_img"]

                st.session_state["history"] = []
                st.session_state["redo_stack"] = []
                st.session_state["operation_history"] = []
                st.rerun()

        # Export image
        with export_col:
            if st.button("Export", key="export", use_container_width=True):
                st.session_state["export_clicked"] = True

    
    # Actions.display_ori_img_info()

    if st.session_state["export_clicked"]:        
        Actions.on_export_clicked()

            # buf = io.BytesIO()
            # final_img = Image.fromarray(st.session_state["curr_img"].img_arr.astype("uint8"))
            # final_img.save(buf, format="PNG")
            # buf.seek(0)
            # st.download_button(
            #     label="Download",
            #     data=buf,
            #     file_name="edited_image.png",
            #     mime="image/png", use_container_width=True, key="download")
    
        # Uploading new image
    
    # Uploading new image
    new_image = st.file_uploader("Change Picture", type=["jpeg", "png", "jpg", "bmp"], key="new_image")
    if new_image:
        st.session_state["image"] = Actions.image_uploaded(new_image, MAX_SIZE)
        st.rerun()

    # Back to HOMEPAGE
    # if st.button("Back to Homepage", key="back_to_homepage"):
    #     st.session_state.clear()
    #     st.rerun()
    


image_loaded = "ori_img" in st.session_state

if image_loaded:
    editor_page()
else:
    start = time.perf_counter()
    landing_page()
    print(f"Time for rendering homepage: {time.perf_counter()-start}")