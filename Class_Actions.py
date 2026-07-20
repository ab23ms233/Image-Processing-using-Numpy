from dataclasses import dataclass
import streamlit as st
import numpy as np
from Class_ImgProcessing import ImageProcessing
from typing import Literal, Optional
from PIL import Image
from io import BytesIO

# Data class for storing image operation history
@dataclass
class ImageState:
    operation_name: str
    img_arr: np.ndarray
    parameters: Optional[dict] = None

# Image actions class
class Actions:
    # Undo operation
    @staticmethod
    def undo():
        if st.session_state["history"]:
            curr_state: ImageState = st.session_state["curr_img"]
            st.session_state["redo_stack"].append(curr_state)
            st.session_state["curr_img"] = st.session_state["history"].pop()
            st.rerun()
    
    # Redo operation
    @staticmethod
    def redo():
        if st.session_state["redo_stack"]:
            redo_op: ImageState = st.session_state["redo_stack"].pop()
            st.session_state["history"].append(st.session_state["curr_img"])
            st.session_state["curr_img"] = redo_op
            st.rerun()
    
    # Function to apply image operations
    @staticmethod
    def apply_operation(operation, name, parameters=None):
        # start = time.perf_counter()
        # Including last image state in history
        if len(st.session_state["history"]) == 0:
            st.session_state["history"].append(st.session_state["ori_img"])
        else:
            st.session_state["history"].append(st.session_state["curr_img"])
        # Clearing redo stack
        st.session_state["redo_stack"].clear()

        # Applying current operation
        proc = ImageProcessing(st.session_state["curr_img"].img_arr.copy())
        st.session_state["curr_img"] = ImageState(name, operation(proc), parameters)
        # print(f"Time taken for {name}: {time.perf_counter() - start}")
        # Rerun to see changes
        st.rerun()
    
    # Function to render a button in a column
    @staticmethod
    def button_renderer(button_col, button_name, button_key, operation, operation_name: Literal["Rotate Clockwise", "Rotate Anti-clockwise", "Flip Horizontal", "Flip Vertical", "Grayscale", "Negative", "Binarize", "Edge Detection"], icon: Optional[str] = None):
        with button_col:
            if st.button(button_name, use_container_width=True, key=button_key, icon=icon):
                st.session_state["selected_tool"] = operation_name
                Actions.apply_operation(operation, operation_name)
    
    @staticmethod
    def tool_renderer(selected_tab: str):
        tool = st.session_state["selected_tool"]

        # Sharpen
        if selected_tab == "Filters" and tool == "Sharpen":
            strength = st.slider("Sharpen strength", 0.6, 3.0, 1.0, step=0.2)
            if st.button("Apply sharpen", key="apply_sharpen"):
                Actions.apply_operation(lambda p: p.sharpen_img(strength=strength), "Sharpen", parameters={"strength": strength})
        # Blur
        elif selected_tab == "Filters" and tool == "Blur":
            strength = st.slider("Blur strength", 0.6, 3.0, 1.0, step=0.2)
            if st.button("Apply blur", key="apply_blur"):
                Actions.apply_operation(lambda p: p.blur_img(strength=strength), "Blur", parameters={"strength": strength})
        # Binarize
        elif selected_tab == "Colors" and tool == "Binarize":
            threshold = st.slider("Threshold", 0, 255, 128)
            if st.button("Apply binarize", key="apply_binarize"):
                Actions.apply_operation(lambda p: p.binarise(int(threshold)), "Binarize", parameters={"threshold": threshold})
    
    # Function for creating history
    @staticmethod
    def display_history():
        with st.expander("History", expanded=False):
            # If history is empty
            if len(st.session_state["history"]) == 0:
                st.markdown("No operations applied yet.")
            # If history is present
            else:
                for num, history in enumerate(st.session_state["history"]):
                    operation = f"{num+1}. {history.operation_name}"
                    # Displaying parameters if present
                    parameters = history.parameters
                    if parameters:
                        params = [f"{parameter + ": "+ str(parameters[parameter])}" for parameter in parameters]
                        operation += f" ({",".join(params)})"
                    
                    st.markdown(operation)

    # Function for displaying image once uploaded
    @staticmethod
    def image_uploaded(uploaded_file, MAX_SIZE):
        img = Image.open(uploaded_file)
        img_format = img.format
        # print(img_format)
        img_size = uploaded_file.size
        img = img.convert("RGB")

        # Reduced height and width of images (in case it is larger than MAX_SIZE)
        h, w = img.height, img.width
        scale = min(MAX_SIZE/h, MAX_SIZE/w, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h))

        st.session_state["image_metadata"] = {
            "name": uploaded_file.name,
            "original_w": w,
            "original_h": h,
            "working_w": new_w,
            "working_h": new_h,
            "format": img_format,
            "size": img_size
        }

        st.session_state["page"] = "editor"
        image_arr = np.array(img)
        st.session_state["ori_img"] = ImageState("Original", image_arr)
        st.session_state["curr_img"] = ImageState("Original", image_arr)
        return img

    @staticmethod
    def format_file_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"
    
    @staticmethod
    def on_export_clicked():
        # Original image metadata
        original_h, original_w = st.session_state["image_metadata"]["working_h"], st.session_state["image_metadata"]["working_w"]
        img_format = st.session_state["image_metadata"]["format"]
        original_size = st.session_state["image_metadata"]["size"]

        # Edited image metadata
        shape = st.session_state["curr_img"].img_arr.shape
        edited_h, edited_w = shape[0], shape[1]
        # size
        buffer = BytesIO()
        final_img = Image.fromarray(st.session_state["curr_img"].img_arr.astype("uint8"))
        final_img.save(buffer, format=img_format)
        size_bytes = len(buffer.getvalue())

        # Rendering image info table
        st.markdown("### Image information")
        st.write("")
        st.markdown(f"""
    <table style="width:100%; border-collapse: collapse;">
    <tr>
        <th style="border:1px solid gray; padding:8px; text-align: center;">Parameter</th>
        <th style="border:1px solid gray; padding:8px; text-align: center;">Original</th>
        <th style="border:1px solid gray; padding:8px; text-align: center;">Edited</th>
    </tr>
    <tr>
        <td style="border:1px solid gray; padding:8px; text-align: center;">Dimensions</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{original_w} × {original_h}</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{edited_w} × {edited_h}</td>
    </tr>
    <tr>
        <td style="border:1px solid gray; padding:8px; text-align: center;">Format</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{img_format}</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{img_format}</td>
    </tr>
    <tr>
        <td style="border:1px solid gray; padding:8px; text-align: center;">Size</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{Actions.format_file_size(original_size)}</td>
        <td style="border:1px solid gray; padding:8px; text-align: center;">{Actions.format_file_size(size_bytes)}</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
        st.write("")
        st.markdown("### Export settings")
        file_name_col, format_col, size_col = st.columns(3, gap="large")

        with file_name_col:
            file_name = st.text_input(
                "File Name",
                # value="edited_image",
                placeholder="Enter output file name") 
            
        with format_col:
            format_option = st.selectbox(
                "File Format",
                ["Original", "JPEG/JPG", "PNG", "WebP"],
                index=0,
            )

            if format_option == "Original":
                format_option = img_format
            elif format_option == "JPEG/JPG":
                format_option = "JPEG"

            if format_option in ["JPEG", "JPG", "WebP"]:
                quality = st.slider("Quality", 1, 100, 90)
            else:
                quality = None
        
        buffer = BytesIO()
        final_img.save(buffer, format=format_option, quality=quality)
        compressed_size_bytes = len(buffer.getvalue())
        size_col.metric("Export Size", Actions.format_file_size(compressed_size_bytes))

        extension = format_option.lower() 
        st.download_button(
                label="Download Image",
                data=buffer.getvalue(),
                file_name=f"{file_name}.{extension}",
                key="download")
    
    @staticmethod
    @st.cache_data
    def load_cached_image(img_path):
        img = Image.open(img_path)
        return img
    
