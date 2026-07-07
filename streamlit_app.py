import streamlit as st
from PIL import Image
import numpy as np
from Class_Actions import ImageState, Actions
import io
import time

# Page configuration
st.set_page_config(
    page_title="Image Processing Application",
    layout="wide",
    initial_sidebar_state="collapsed")

# Maximum image dimension
MAX_SIZE = 1200

operations = ["Sharpen", "Blur", "Rotate Clockwise", "Rotate Anti-clockwise", ""]
tabs = ["Transform", "Filters", "Colors", "History"]

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
    }

    .feature-card {
    position: relative;
    background: #0f1720;
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 22px 22px 22px 30px;
    overflow: hidden;

    transition:
        border-color 0.25s ease,
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

/* Left accent strip */
.feature-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 6px;
    height: 100%;
    border-radius: 18px 0 0 18px;
}

/* Hover animation */
.feature-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}

/* Individual accent colours */
.feature-card-blue {
    --accent: #60a5fa;
}

.feature-card-blue::before {
    background: #60a5fa;
}

.feature-card-teal {
    --accent: #2dd4bf;
}

.feature-card-teal::before {
    background: #2dd4bf;
}

.feature-card-purple {
    --accent: #a78bfa;
}

.feature-card-purple::before {
    background: #a78bfa;
}

.feature-card-cyan {
    --accent: #22d3ee;
}

.feature-card-cyan::before {
    background: #22d3ee;
}

/* Headings */
.feature-card h3 {
    color: white;
    margin-bottom: 8px;
    font-size: 1.15rem;
}

/* Paragraph */
.feature-card p {
    color: #94a3b8;
    margin-bottom: 0;
    line-height: 1.5;
}
""", unsafe_allow_html=True)

# LANDING PAGE
def landing_page():
    print("Landing page")
    # HERO SECTION
    hero_left, hero_right = st.columns([1.5, 1], gap="medium")

    # Left side
    with hero_left:
        start = time.perf_counter()
        # Heading
        st.markdown("<h1 style='color:#f8fafc; margin-bottom: 0.25rem;'>Image Processing Studio</h1>", unsafe_allow_html=True)
        # Description
        st.markdown("<p style='color:#cbd5e1; font-size:1.2rem; margin-top:0; max-width:650px;'>Upload, edit and download images with fast one-click controls in a clean dark interface.</p>",
                    unsafe_allow_html=True)
        # st.markdown("<p style='color:#94a3b8; margin-top:1rem; max-width:650px;'>Rotate, flip, blur, sharpen and download your final image quickly with instant preview and responsive controls.</p>", unsafe_allow_html=True)
        
        # File upload button
        file_uploader_col, right = st.columns([3, 1])
        with file_uploader_col:
            uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "bmp"])
        print(f"Time for rendering hero_left: {time.perf_counter() - start}")
    
    # Right side
    with hero_right:
        # start = time.perf_counter()

        # Image
        @st.cache_data
        def load_image():
            image = Image.open("images/deep-mind-compressed.webp").convert("RGB")
            return image
        
        start = time.perf_counter()
        hero_img = load_image()
        print(f"Time for loading image: {time.perf_counter()-start}")
        start = time.perf_counter()
        st.image(hero_img, use_container_width=True)
        print(f"Time for rendering image: {time.perf_counter()-start}")
        # Description
        st.markdown("<p style='text-align: center;'>Photo by <a href='https://unsplash.com/@googledeepmind' target='_blank'>Google DeepMind</a></p>",
    unsafe_allow_html=True)
        # print(f"Time for rendering hero_right: {time.perf_counter() - start}")
    
    start = time.perf_counter()
    # Feature cards
    st.markdown("---")
    card_1, card_2, card_3, card_4 = st.columns(4, gap="small")
    with card_1:
        st.markdown("<div class='feature-card feature-card-blue'><h3>⚡Fast Preview</h3><p>Apply filters instantly.</p></div>", unsafe_allow_html=True)
    with card_2:
        st.markdown("<div class='feature-card feature-card-teal'><h3>🔄 Rotate & Flip</h3><p>One-click orientation controls.</p></div>", unsafe_allow_html=True)
    with card_3:
        st.markdown("<div class='feature-card feature-card-purple'><h3>🎨 Filters</h3><p>Grayscale, binarize, sharpen, etc.</p></div>", unsafe_allow_html=True)
    with card_4:
        st.markdown("<div class='feature-card feature-card-cyan'><h3>⬇ Download</h3><p>Save your edited image as a PNG.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    print(f"Time for rendering feature_cards: {time.perf_counter() - start}")

    # If file is uploaded
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        # Reduced height and width of images (in case it is larger than MAX_SIZE)
        h, w = img.height, img.width
        scale = min(MAX_SIZE / h, MAX_SIZE / w, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h))

        image_arr = np.array(img)
        st.session_state["ori_img"] = ImageState("Original", image_arr)
        st.session_state["curr_img"] = ImageState("Original", image_arr)
        st.rerun()


# EDITOR PAGE
def editor_page():
    # print("Editor page")
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "redo_stack" not in st.session_state:
        st.session_state["redo_stack"] = []
    if "selected_tool" not in st.session_state:
        st.session_state["selected_tool"] = {"tab": None, "tool": None}
    
    original, processed, editor_pane = st.columns([1, 1, 1.5], gap="medium")
    # start = time.perf_counter()

    # Original image
    with original:
        st.markdown("<h3 style='text-align: center;'>Original</h3>", unsafe_allow_html=True)
        st.image(Image.fromarray(st.session_state["ori_img"].img_arr), width=360)
    # Processed image
    with processed:
        st.markdown("<h3 style='text-align: center;'>Processed</h3>", unsafe_allow_html=True)
        st.image(Image.fromarray(st.session_state["curr_img"].img_arr), width=360)
    st.markdown("---")

    # print(f"Time for displaying images: {time.perf_counter() - start}")
    
    # Editor pane
    with editor_pane:
        # Different tabs for image processing tools
        transform_tab, filter_tab, color_tab, history_tab = st.tabs(tabs)

        # start = time.perf_counter()
        # TRANSFORM TAB
        with transform_tab:
            st.subheader("Transform")
            st.session_state["selected_tool"]["tab"] = "Transform"
            # Rotate
            rot_c_col, rot_ac_col = st.columns(2, gap="medium")
            Actions.button_renderer(rot_c_col, "⟳ Rotate Clockwise", "rotate_clockwise", lambda p: p.rotate_img(num=1), "Rotate Clockwise")
            Actions.button_renderer(rot_ac_col, "⟲ Rotate Anti-Clockwise", "rotate_anti_clockwise", lambda p: p.rotate_img(num=-1), "Rotate Anti-clockwise")
            # Flip
            fliph_col, flipv_col = st.columns(2, gap="medium")
            Actions.button_renderer(fliph_col, "↔️ Flip Horizontal", "flip_horizontal", lambda p: p.flip_img(plane='h'), "Flip Horizontal")
            Actions.button_renderer(flipv_col, "↕️ Flip Vertical", "flip_vertical", lambda p: p.flip_img(plane='v'), "Flip Vertical")

        # COLORS
        with color_tab:
            st.subheader("Colors")
            st.session_state["selected_tool"]["tab"] = "Colors"
            negative_col, grayscale_col, binarize_col = st.columns(3, gap="medium")

            # Negative
            Actions.button_renderer(negative_col, "Negative", "negative", lambda p: p.negative(), "Negative")
            # Grayscale
            Actions.button_renderer(grayscale_col, "Grayscale", "grayscale", lambda p: p.grayscale(), "Grayscale")
            # Binarize
            with binarize_col:
                if st.button("Binarize", use_container_width=True, key="binarize"):
                    st.session_state["selected_tool"]["tab"] = "Binarize"

            # st.markdown("-----")

        # FILTERS
        with filter_tab:            
            st.subheader("Filters")
            st.session_state["selected_tool"]["tab"] = "Filters"
            sharpen_col, blur_col, edge_col = st.columns(3, gap="medium")

            # Sharpen
            with sharpen_col:
                if st.button("Sharpen", use_container_width=True, key="sharpen"):
                    st.session_state["selected_tool"]["tool"] = "Sharpen"
            # Blur
            with blur_col:
                if st.button("Blur", use_container_width=True, key="blur"):
                    st.session_state["selected_tool"]["tool"] = "Blur"
            # Edge detection
            Actions.button_renderer(edge_col, "Edge Detection", "edge_detection", lambda p: p.edge_detection(), "Edge Detection")

            # st.markdown("-----")
        
        # Render sliders if required by operation
        Actions.tool_renderer()
        

        # HISTORY
        # with history_tab:

        st.markdown("-----")
        # print(f"Time for rendering tabs: {time.perf_counter()-start}")

        undo_col, redo_col, reset_col, download_col = st.columns([1, 1, 1.5, 1], gap="small")

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

        # Download image
        with download_col:
            buf = io.BytesIO()
            final_img = Image.fromarray(st.session_state["curr_img"].img_arr.astype("uint8"))
            final_img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="Download",
                data=buf,
                file_name="edited_image.png",
                mime="image/png", use_container_width=True, key="download")
    
    if st.button("Back to Homepage", key="back_to_homepage"):
        st.session_state.clear()
        st.rerun()
    # Displaying history
    # Actions.draw_history()


image_loaded = "ori_img" in st.session_state

if image_loaded:
    editor_page()
else:
    start = time.perf_counter()
    landing_page()
    print(f"Time for rendering homepage: {time.perf_counter()-start}")