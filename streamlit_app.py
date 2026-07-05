import streamlit as st
from PIL import Image
import numpy as np
from Class_ImgProcessing import ImageProcessing
import io

# Page configuration
st.set_page_config(
    page_title="Image Processing Application",
    layout="wide",
    initial_sidebar_state="collapsed")

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
    # HERO SECTION
    hero_left, hero_right = st.columns([1.5, 1], gap="medium")

    # Left side
    with hero_left:
        # Heading
        st.markdown("<h1 style='color:#f8fafc; margin-bottom: 0.25rem;'>Image Processing Application</h1>", unsafe_allow_html=True)
        # Description
        st.markdown("<p style='color:#cbd5e1; font-size:1.2rem; margin-top:0; max-width:650px;'>Upload, edit and download images with fast one-click controls in a clean dark interface.</p>",
                    unsafe_allow_html=True)
        # st.markdown("<p style='color:#94a3b8; margin-top:1rem; max-width:650px;'>Rotate, flip, blur, sharpen and download your final image quickly with instant preview and responsive controls.</p>", unsafe_allow_html=True)
        
        # File upload button
        file_uploader_col, right = st.columns([3, 1])
        with file_uploader_col:
            uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "bmp"])
        
    # st.write("")
    
    # Right side
    with hero_right:
        placeholder = Image.open("images/deep-mind-compressed.jpg").convert("RGB")
        # Image
        st.image(placeholder, use_container_width=True)
        # Description
        st.markdown(
    "<p style='text-align: center;'>Photo by <a href='https://unsplash.com/@googledeepmind' target='_blank'>Google DeepMind</a></p>",
    unsafe_allow_html=True)
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

    # If file is uploaded
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state["ori_img"] = np.array(img)
        st.session_state["curr_img"] = np.array(img)
        st.rerun()


# EDITOR PAGE
def editor_page():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "redo_stack" not in st.session_state:
        st.session_state["redo_stack"] = []
    if "operation_history" not in st.session_state:
        st.session_state["operation_history"] = []

    original, processed = st.columns(2, gap="small")
    # Original image
    with original:
        st.subheader("Original")
        st.image(Image.fromarray(st.session_state["ori_img"]), width=360)
    # Processed image
    with processed:
        st.subheader("Processed (preview)")
        st.image(Image.fromarray(st.session_state["curr_img"]), width=360)
    st.markdown("---")

    # Function to apply image operations
    def apply_operation(operation, name):
        # Including last image state in history
        st.session_state["history"].append(st.session_state["curr_img"].copy())
        st.session_state["operation_history"].append(name)
        # Clearing redo stack
        st.session_state["redo_stack"].clear()

        # Applying current operation
        proc = ImageProcessing(st.session_state["curr_img"].copy())
        st.session_state["curr_img"] = operation(proc)
        # Rerun to see changes
        st.rerun()

    # Different tabs for image processing tools
    transform_tab, filter_tab, color_tab, generate_tab = st.tabs(
    ["Transform", "Filters", "Colors", "Generate"])

    # TRANSFORM TAB
    with transform_tab:
        st.subheader("Transform")
        rotate_col, flip_col = st.columns(2, gap="large")

        # Rotate
        with rotate_col:
            st.markdown("### Rotate")
            clock_col, anti_clock_col = st.columns(2, gap="medium")
            
            with clock_col:     # Clockwise
                if st.button("⟳ Clockwise", use_container_width=True, key="rotate_clockwise"):
                    apply_operation(lambda p: p.rotate_img(num=1), "Rotate clockwise")
            with anti_clock_col:        # Anti-clockwise
                if st.button("⟲ Anti-Clockwise", use_container_width=True, key="rotate_anticlockwise"):
                    apply_operation(lambda p: p.rotate_img(num=-1), "Rotate Anti-Clockwise")
        
        # Flip
        with flip_col:      
            st.markdown("### Flip")
            fliph_col, flipv_col = st.columns(2, gap="medium")

            with fliph_col:     # Horizontal
                if st.button("↔️ Horizontal", use_container_width=True, key="flip_horizontal"):
                    apply_operation(lambda p: p.flip_img(plane='h'), "Flip Horizontal")
            with flipv_col:     # Vertical
                if st.button("↕️ Vertical", use_container_width=True, key="flip_vertical"):
                    apply_operation(lambda p: p.flip_img(plane='v'), "Flip Vertical")
    
    # FILTER
    with filter_tab:
        st.subheader("Filters")
        negative_col, grayscale_col, binarize_col = st.columns(3, gap="medium")

        with negative_col:      # Negative 
            if st.button("Negative", use_container_width=True, key="negative"):
                apply_operation(lambda p: p.negative(), "Negative")
        with grayscale_col:        # Grayscale
            if st.button("Grayscale", use_container_width=True, key="grayscale"):
                apply_operation(lambda p: p.grayscale(), "Grayscale")
        with binarize_col:      # Binarize
            if st.button("Binarize", use_container_width=True, key="binarize"):
                threshold = st.slider("Threshold", 0, 255, 128)
                apply_operation(lambda p: p.binarise(int(threshold)), "Binarize")
    
    
    action_col, reset_col, download_col = st.columns(3, gap="small")

    # Image operations
    with action_col:
        if st.button("Preview"):
            proc = ImageProcessing(st.session_state["curr_img"].copy())
            # Binarize
            if operation == "Binarize":
                st.session_state["curr_img"] = proc.binarise(int(threshold))
            # Negative
            elif operation == "Negative":
                st.session_state["curr_img"] = proc.negative()
            # Grayscale
            elif operation == "Grayscale":
                st.session_state["curr_img"] = proc.grayscale()
            # Sharpen
            elif operation == "Sharpen":
                st.session_state["curr_img"] = proc.sharpen_img()
            # Blur
            elif operation == "Blur":
                st.session_state["curr_img"] = proc.blur_img()
            # Edge detection
            elif operation == "Edge Detection":
                st.session_state["curr_img"] = proc.edge_detection()
    
    # Reset to original
    with reset_col:
        if st.button("Reset to Original"):
            st.session_state["curr_img"] = st.session_state["ori_img"].copy()
    
    # Download image
    with download_col:
        buf = io.BytesIO()
        final_img = Image.fromarray(st.session_state["curr_img"].astype("uint8"))
        final_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="Download",
            data=buf,
            file_name="edited_image.png",
            mime="image/png",
        )
    
image_loaded = "ori_img" in st.session_state

if image_loaded:
    editor_page()
else:
    landing_page()