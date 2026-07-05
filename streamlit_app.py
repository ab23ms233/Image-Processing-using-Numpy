from requests import session
import streamlit as st
from PIL import Image
import numpy as np
from Class_ImgProcessing import ImageProcessing
import io

# Page configuration
st.set_page_config(page_title="Image Processor", layout="wide")

# Header section
st.title("Image Processing App")
st.write("Welcome to the all-in-one image processor. Whatever you need - all in one place")
st.subheader("Rotate. Flip. Blur. Sharpen")
st.info("Info box")

# Image upload
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# Reading file
if uploaded_file is not None:
    ori_img = Image.open(uploaded_file).convert("RGB")
    ori_arr = np.array(ori_img)

    if "ori_img" not in st.session_state or st.session_state.get("uploaded_name") != uploaded_file.name:
        st.session_state["ori_img"] = ori_arr
        st.session_state["current_img"] = ori_arr.copy()
        st.session_state["uploaded_name"] = uploaded_file.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(Image.fromarray(st.session_state["ori_img"]), use_container_width=True)
    with col2:
        st.subheader("Processed (preview)")
        st.image(Image.fromarray(st.session_state["current_img"]), use_container_width=True)
    
    st.markdown("---")

    with st.form("controls"):
        st.write("## Operations")
        operation = st.selectbox("Operation", ("Binarize", ))
        threshold = st.slider("Threshold", 0, 255, 128)
        submitted = st.form_submit_button("Preview")
    
    if submitted:
        proc = ImageProcessing(st.session_state["ori_img"])
        out = proc.binarise(int(threshold))
        st.session_state["current_img"] = out

    if st.button("Reset to Original"):
        st.session_state["current_img"] = st.session_state["ori_img"].copy()
else:
    st.info("Please upload an image to get started")
