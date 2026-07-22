import time
import streamlit as st
from Class_Actions import Actions

def render_img_ops(tabs):
    st.markdown("""
            <style>
                .st-key-transform_container button,
                .st-key-colors_container button,
                .st-key-filters_container button {
                padding: 0.8rem;
                border-radius: 16px;
                }

                .st-key-transform_container button p,
                .st-key-colors_container button p,
                .st-key-filters_container button p {
                font-family: "Fira Code", monospace;
                font-size: 0.9rem;
                }

                .st-key-other_ops_container button {
                padding: 0.8rem;
                border-radius: 16px;
                }

                .st-key-other_ops_container button p {
                font-family: "Inria Sans", sans-serif;
                }

                div[data-testid="stExpander"] p {
                font-family: "Inria Sans", sans-serif;
                }

                .st-key-reset button {
                border-color: rgb(255, 75, 75);
                background: transparent;
                color: rgb(255, 75, 75);
                transition: all 0.3s ease;
                }

                .st-key-reset button:hover {
                background: rgba(255, 75, 75, 0.2);
                }

                .st-key-export button {
                background: #FF2727;
                color: white;
                transition: all 0.3s ease;
                }

                .st-key-export button:hover {
                background: #FF5656;
                color: white;
                }

                .st-key-export button:focus {
                background: #FF5656 !important;
                color: white !important;
                border: 2px solid white !important;
                }
            </style>
                """, unsafe_allow_html=True)
    
    original, processed, editor_pane = st.columns([1, 1, 1.5], gap="small")
    
    # Editor pane
    with editor_pane:
        start = time.perf_counter()
        # Different tabs for image processing tools
        selected_tab = st.segmented_control("Category", tabs, label_visibility="collapsed", default="Transform")

        # TRANSFORM TAB
        if selected_tab == "Transform":
            st.subheader("Transform")
            transform_container = st.container(key="transform_container")

            with transform_container:
                # Rotate
                rot_c_col, rot_ac_col = st.columns(2, gap="medium")
                Actions.button_renderer(rot_c_col, "⟳ Rotate Clockwise", "rotate_clockwise", lambda p: p.rotate_img(num=1), "Rotate Clockwise")
                Actions.button_renderer(rot_ac_col, "⟲ Rotate Anti-Clockwise", "rotate_anti_clockwise", lambda p: p.rotate_img(num=-1), "Rotate Anti-clockwise")
                # Flip
                fliph_col, flipv_col = st.columns(2, gap="medium")
                Actions.button_renderer(fliph_col, "↔ Flip Horizontal", "flip_horizontal", lambda p: p.flip_img(plane='h'), "Flip Horizontal")
                Actions.button_renderer(flipv_col, "↕ Flip Vertical", "flip_vertical", lambda p: p.flip_img(plane='v'), "Flip Vertical")

        # FILTERS TAB
        if selected_tab == "Filters":
            st.subheader("Filters")
            filters_container = st.container(key="filters_container")

            with filters_container:
                sharpen_col, blur_col, edge_col = st.columns(3, gap="medium")

                # Sharpen
                with sharpen_col:
                    if st.button("✨ Sharpen", use_container_width=True, key="sharpen"):
                        st.session_state["selected_tool"] = "Sharpen"
                # Blur
                with blur_col:
                    if st.button("🌫️ Blur", use_container_width=True, key="blur"):
                        st.session_state["selected_tool"] = "Blur"
                # Edge detection
                Actions.button_renderer(edge_col, "📐 Edges", "edge_detection", lambda p: p.edge_detection(), "Edge Detection")

        # COLORS
        if selected_tab == "Colors":
            st.subheader("Colors")
            colors_container = st.container(key="colors_container")

            with colors_container:
                negative_col, grayscale_col, binarize_col = st.columns(3, gap="medium")

                # Negative
                Actions.button_renderer(negative_col, "🌓 Negative", "negative", lambda p: p.negative(), "Negative")
                # Grayscale
                Actions.button_renderer(grayscale_col, "⚫⚪ Grayscale", "grayscale", lambda p: p.grayscale(), "Grayscale")
                # Binarize
                with binarize_col:
                    if st.button("◼️◻️ Binarize", use_container_width=True, key="binarize"):
                        st.session_state["selected_tool"] = "Binarize"
        
        # Render sliders if required by operation
        Actions.tool_renderer(selected_tab)
        st.markdown("-----")

        # Displaying HISTORY
        Actions.display_history()
        other_ops_container = st.container(key="other_ops_container")
        with other_ops_container:
            undo_col, redo_col, reset_col, export_col = st.columns([1, 1, 1.5, 1], gap="small")

            # Undo
            with undo_col:
                if st.button("↩ Undo", use_container_width=True, key="undo"):
                    Actions.undo()
            # Redo
            with redo_col:
                if st.button("↪ Redo", use_container_width=True, key="redo"):
                    Actions.redo()

            # Reset to original
            with reset_col:
                if st.button("Reset to Original", use_container_width=True, key="reset"):
                    Actions.reset()

            # Export image
            with export_col:
                if st.button("Export", key="export", use_container_width=True):
                    st.session_state["export_mode"] = True
        
        print(f"Time for rendering tabs: {time.perf_counter()-start}")

    start = time.perf_counter()
    # Original image
    with original:
        st.markdown("<h3 style='text-align: center'>Original</h3>", unsafe_allow_html=True)
        st.image(st.session_state["ori_preview"], width=360)
        # st.image(Image.fromarray(st.session_state["ori_img"].img_arr), width=360)
    # Processed image
    with processed:
        st.markdown("<h3 style='text-align: center'>Processed</h3>", unsafe_allow_html=True)
        st.image(st.session_state["curr_preview"], width=360)
        # st.image(Image.fromarray(st.session_state["curr_img"].img_arr), width=360)

    print(f"Time for displaying editor images: {time.perf_counter() - start}")

        
