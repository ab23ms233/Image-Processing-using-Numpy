from dataclasses import dataclass
import streamlit as st
import numpy as np
from Class_ImgProcessing import ImageProcessing
from typing import Literal, Optional

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
    def button_renderer(button_col, button_name, button_key, operation, operation_name: Literal["Rotate Clockwise", "Rotate Anti-clockwise", "Flip Horizontal", "Flip Vertical", "Grayscale", "Negative", "Binarize", "Edge Detection"]):
        with button_col:
            if st.button(button_name, use_container_width=True, key=button_key):
                st.session_state["selected_tool"]["tool"] = operation_name
                Actions.apply_operation(operation, operation_name)
    
    @staticmethod
    def tool_renderer():
        tool = st.session_state["selected_tool"].get("tool")
        tab = st.session_state["selected_tool"].get("tab")

        # Sharpen
        if tab == "Filters" and tool == "Sharpen":
            strength = st.slider("Sharpen strength", 0.6, 3.0, 1.0, step=0.2)
            if st.button("Apply sharpen", key="apply_sharpen"):
                Actions.apply_operation(lambda p: p.sharpen_img(strength=strength), "Sharpen", parameters={"strength": strength})
        # Blur
        elif tab == "Filters" and tool == "Blur":
            strength = st.slider("Blur strength", 0.6, 3.0, 1.0, step=0.2)
            if st.button("Apply blur", key="apply_blur"):
                Actions.apply_operation(lambda p: p.blur_img(strength=strength), "Blur", parameters={"strength": strength})
        # Binarize
        elif tab == "Transform" and tool == "Binarize":
            threshold = st.slider("Threshold", 0, 255, 128)
            if st.button("Apply binarize", key="apply_binarize"):
                Actions.apply_operation(lambda p: p.binarise(int(threshold)), "Binarize", {"threshold": threshold})
    
    # Function for creating history
    @staticmethod
    def display_history():
        st.markdown("### History")
        op_names = []

        for num, history in enumerate(st.session_state["history"]):
            op_name = f"{num+1}. {history.operation_name}"
    
        
        
    
