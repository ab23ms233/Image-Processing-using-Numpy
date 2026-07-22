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
            st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
    
    # Redo operation
    @staticmethod
    def redo():
        if st.session_state["redo_stack"]:
            redo_op: ImageState = st.session_state["redo_stack"].pop()
            st.session_state["history"].append(st.session_state["curr_img"])
            st.session_state["curr_img"] = redo_op
            st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
    
    @staticmethod
    def reset():
        st.session_state["curr_img"] = st.session_state["ori_img"]
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
        st.session_state["export_mode"] = False

        st.session_state["history"] = []
        st.session_state["redo_stack"] = []
        st.session_state["operation_history"] = []

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

        # Storing preview images
        if "ori_peview" not in st.session_state:
            st.session_state["ori_preview"] = Image.fromarray(st.session_state["ori_img"].img_arr)
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)

        # Exiting export mode
        st.session_state["export_mode"] = False
        # print(f"Time taken for {name}: {time.perf_counter() - start}")
    
    # Function to render a button in a column
    @staticmethod
    def button_renderer(button_col, button_name, button_key, operation, operation_name: Literal["same", "Rotate Clockwise", "Rotate Anti-clockwise", "Flip Horizontal", "Flip Vertical", "Grayscale", "Negative", "Binarize", "Edge Detection"] = "same", icon: Optional[str] = None):
        if operation_name == "same":
            operation_name = button_name

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

        st.session_state["ori_preview"] = Image.fromarray(st.session_state["ori_img"].img_arr)
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
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
    @st.cache_data
    def load_cached_image(img_path):
        img = Image.open(img_path)
        return img
    
