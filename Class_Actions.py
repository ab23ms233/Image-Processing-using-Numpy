"""
## Purpose:
    Manages image editing actions, state transitions, and editor interactions.

## Responsibilities:
- Applies image processing operations to the current image.
- Maintains edit history with undo, redo, and reset functionality.
- Renders editor controls, tool-specific widgets, and operation history.
- Display preview of original and edited images.

## Classes used:
- ImageState: Stores an image snapshot along with the operation and parameters that produced it.
- Actions: Provides static methods for image editing, history management, UI rendering, and image handling.

## Dependencies:
- streamlit: For rendering web interface.
- numpy: For storing and manipulating image arrays.
- Pillow (PIL): For image loading, resizing, preview generation, and caching.
- Class_ImgProcessing: For executing image processing algorithms.
- dataclasses: For defining the ImageState data class.
- typing: For type annotations such as Callable, Literal, and Optional.
"""

from dataclasses import dataclass
import streamlit as st
import numpy as np
from Class_ImgProcessing import ImageProcessing
from typing import Callable, Literal, Optional
from PIL import Image

# Data class for storing image operation history
@dataclass
class ImageState:
    """
    Stores the state of an image at a given point in the editing workflow.

    Attributes:
        operation_name (str): The name of the operation that produced this image state (E.g., *"Sharpen"*").
        operation (Callable): A callable function of the operation that produced this state.
        img_arr (np.ndarray): The image data as a NumPy array.
        parameters (Optional[dict[str, float]]): Optional parameters used by the operation, such as threshold
            or strength values. Defaults to None if no parameters were used.
    """
    operation_name: str
    operation: Callable | None
    img_arr: np.ndarray
    parameters: Optional[dict[str, float]] = None

# Image actions class
class Actions:
    """
    Provides methods for managing image editing operations. Handles image transformations,
    history management, UI rendering, image uploads, and helper functions used
    throughout the image editor.

    ### Methods
    - **apply_operation:** Applies an image operation and updates the current state, history, previews, and export status.
    - **undo:** Restores the previous image state from the edit history.
    - **redo:** Reapplies the most recently undone operation.
    - **reset:** Restores the original image and clears the editing history.
    - **button_renderer:** Renders an operation button and executes the corresponding action when clicked.
    - **tool_renderer:** Displays parameter controls for tools that require user input before execution.
    - **display_history:** Displays the list of applied image operations and their parameters.
    - **image_uploaded:** Loads an uploaded image, extracts metadata, initializes editor state, and prepares preview images.
    - **format_file_size:** Converts a file size in bytes into a human-readable format.
    - **load_cached_image:** Loads and caches an image for faster rendering.
    - **export_image:** Apply all operations in "history" on original, unresized image array before exporting
    """
    # Function to apply image operations
    @staticmethod
    def apply_operation(operation: Callable, name: str, parameters: Optional[dict[str, float]] = None):
        """
        Apply an image operation to the current image state.

        Parameters:
            operation (Callable): A callable that takes an ImageProcessing instance and returns a processed image array.
            name (str): A human-readable name for the operation, used in history.
            parameters (Optional[dict]): Optional parameters associated with the operation,
                such as threshold or strength values.
        """
        # start = time.perf_counter()

        # history is a list of ImageState objects - each object containing the state
        # of the image during some point in the editing workflow

        # If history is empty, append original img state, else append current img state
        # before applying operation
        if len(st.session_state["history"]) == 0:
            st.session_state["history"].append(st.session_state["ori_img"])
        else:
            st.session_state["history"].append(st.session_state["curr_img"])

        # Clearing redo stack, since applying operation creates a new branch and
        # older branch is not accessible anymore
        st.session_state["redo_stack"].clear()

        # Applying current operation
        proc = ImageProcessing(st.session_state["curr_img"].img_arr.copy())
        st.session_state["curr_img"] = ImageState(name, operation, operation(proc), parameters)

        # Storing preview images
        if "ori_peview" not in st.session_state:
            st.session_state["ori_preview"] = Image.fromarray(st.session_state["ori_img"].img_arr)
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)

        # If export button was clicked earlier, export_mode = True
        # Afterwards, if some operation is applied, export_mode = False
        # User has to export after applying operation
        st.session_state["export_mode"] = False
        # print(f"Time taken for {name}: {time.perf_counter() - start}")
    
    # Undo operation
    @staticmethod
    def undo():
        """
        Revert the current image state to the previous state in the edit history.
        """
        # If any operation has been applied
        if st.session_state["history"]:
            curr_state: ImageState = st.session_state["curr_img"]
            # Append current state into redo_stack for implementing redo operations on demand
            st.session_state["redo_stack"].append(curr_state)
            # The current state is the last state in history
            st.session_state["curr_img"] = st.session_state["history"].pop()
            # Updating preview image
            st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
            # Reset export mode
            st.session_state["export_mode"] = False
    
    # Redo operation
    @staticmethod
    def redo():
        """
        Restore the most recently undone image state.
        """
        # If there are operations in redo stack (undo has been done)
        if st.session_state["redo_stack"]:
            # Extract the last operation from redo stack
            redo_op: ImageState = st.session_state["redo_stack"].pop()
            # Append the current img state to history
            st.session_state["history"].append(st.session_state["curr_img"])
            # The current state is redo_op
            st.session_state["curr_img"] = redo_op
            # Update preview image
            st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
            # Reset export mode
            st.session_state["export_mode"] = False
    
    # Reset operation
    @staticmethod
    def reset():
        """
        Reset the editor to its original state.
        """
        # Reset current img state to original img state
        st.session_state["curr_img"] = st.session_state["ori_img"]
        # Update preview
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)
        # Reset export mode
        st.session_state["export_mode"] = False
        # Clear all histories
        st.session_state["history"] = []
        st.session_state["redo_stack"] = []

    # Function to render a button in a column
    @staticmethod
    def button_renderer(button_col,
                        button_name: str,
                        button_key: str,
                        operation: Callable,
                        operation_name: Literal["same", "Rotate Clockwise", "Rotate Anti-clockwise", "Flip Horizontal", "Flip Vertical", "Grayscale", "Negative", "Binarize", "Edge Detection"] = "same",
                        icon: Optional[str] = None):
        """
        Render a button and its fucntioning with given attributes.

        Parameters:
            button_col: The column or container where the button will be rendered.
            button_name (str): The label displayed on the button.
            button_key (str): The unique key assigned to the button widget.
            operation (Callable): The operation to execute when the button is clicked.
            operation_name (Literal["same", "Rotate Clockwise", "Rotate Anti-clockwise", "Flip Horizontal", "Flip Vertical", "Grayscale", "Negative", "Binarize", "Edge Detection"]):
                The name used for the operation in session state and history. If *"same"*, `button_name` is used. Defaults to *"same"*.
            icon (Optional[str]): An optional icon to display with the button label.
        """
        # If operation name is same as button name
        if operation_name == "same":
            operation_name = button_name
        # Render button with given attributes
        with button_col:
            if st.button(button_name, use_container_width=True, key=button_key, icon=icon):
                st.session_state["selected_tool"] = operation_name
                # Apply operation
                Actions.apply_operation(operation, operation_name)
    
    @staticmethod
    def tool_renderer(selected_tab: str):
        """
        Renders interactive controls for the currently selected editing tool.

        Parameters:
            selected_tab (str): The currently selected tab, such as *"Filters"* or *"Colors"*.
        """
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
        """
        Display the list of applied image operations in an expandable section.
        """
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
    def image_uploaded(uploaded_file, MAX_SIZE: int) -> Image.Image:
        """
        Stores image information, does resizing and changes to *"editor"* page once an image is uploaded.

        Parameters:
            uploaded_file: The image file which is uploaded.
            MAX_SIZE (int): Allowed maximum size for previewing images in the editor page. If image size
                is greater than this, then resize images for faster preview.
            
        Returns:
            Image.Image: The resized image
        """
        img = Image.open(uploaded_file)
        # Format of image
        img_format = img.format
        # Size of uploaded image
        img_size = uploaded_file.size
        # Storing the original, unresized image array for applying operations before exporting
        st.session_state["image_arr"] = np.array(img)

        # Reduced height and width of images (in case it is larger than MAX_SIZE)
        h, w = img.height, img.width
        scale = min(MAX_SIZE/h, MAX_SIZE/w, 1)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h))

        # Storing image information
        st.session_state["image_metadata"] = {
            "name": uploaded_file.name,
            "original_w": w,
            "original_h": h,
            "working_w": new_w,
            "working_h": new_h,
            "format": img_format,
            "size": img_size
        }

        # Change page to editor if image is uploaded
        st.session_state["page"] = "editor"
        # Convert image to numpy array
        image_arr = np.array(img)
        # Store image states
        st.session_state["ori_img"] = ImageState("Original", None, image_arr)
        st.session_state["curr_img"] = ImageState("Original", None, image_arr)
        # Store image previews
        st.session_state["ori_preview"] = Image.fromarray(st.session_state["ori_img"].img_arr)
        st.session_state["curr_preview"] = Image.fromarray(st.session_state["curr_img"].img_arr)

        return img

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Display file size in KB, MB or GB based on size in bytes

        Parameters:
            size_bytes (int): File size in bytes
        
        Returns:
            str: File size in KB, MB or GB depending on `size_bytes`
        """
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
    def load_cached_image(img_path: str) -> Image.Image:
        """
        Load an image from disk and cache it to avoid repeated file I/O.

        Parameters:
            img_path (str): Path to the image file.

        Returns:
            A cached PIL Image object.
        """
        img = Image.open(img_path)
        img.load()
        return img
    
    # Applying operations before exporting image
    @staticmethod
    def export_image() -> np.ndarray:
        """
        Applies all operations in "history "to the original, unresized image array before exporting

        Returns:
            np.ndarray: The image array after applying all operations in "history"
        """
        if "edited_img_rendered" not in st.session_state:
            st.session_state["edited_img_rendered"] = False
        # If export_mode is False, edited image has not been rendered yet
        if st.session_state["export_mode"] == False:
            st.session_state["edited_img_rendered"] = False
        # If all operations have already been applied to original image
        if st.session_state["edited_img_rendered"]:
            return st.session_state["image_arr"]
        
        # Original unresized image array
        img_arr = st.session_state["image_arr"]
        proc = ImageProcessing(img_arr)

        # Applying all operation on original array. 1st ImageState is the original image, hence it is skipped
        for state in st.session_state["history"][1:]:
            op = state.operation
            name = state.operation_name
            print(name)
            op(proc)

        # Applying last operation not appended in history
        last_op = st.session_state["curr_img"].operation
        if last_op is not None:
            last_op(proc)

        st.session_state["edited_img_rendered"] = True
        return proc.arr

    @staticmethod
    def get_color_mode(img_arr: np.ndarray) -> str:
        if img_arr.ndim == 2:
            return "Grayscale"

        channels = img_arr.shape[2]
        if channels == 1:
            return "Grayscale"
        elif channels == 3:
            return "RGB"
        elif channels == 4:
            return "RGBA"
        else:
            return "Unknown"
        
