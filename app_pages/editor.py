import streamlit as st
from components.navbar import render_navbar
from Class_Actions import Actions
from PIL import Image

def editor_page(max_img_size):
    # print("Editor page")
    render_navbar()
    tabs = ["Transform", "Filters", "Colors"]
    
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
        st.session_state["image"] = Actions.image_uploaded(new_image, max_img_size)
        st.rerun()

    # Back to HOMEPAGE
    # if st.button("Back to Homepage", key="back_to_homepage"):
    #     st.session_state.clear()
    #     st.rerun()