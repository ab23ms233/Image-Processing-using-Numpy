import time

import streamlit as st
from io import BytesIO
from PIL import Image
from Class_Actions import Actions

def render_img_info():
    st.markdown("""
            <style>
                .table {
                width: 100%;
                }

                .table td, .table th {
                border: 1px solid gray;
                padding: 8px;
                text-align: center;
                }

                .table th {
                text-decoration: bold;
                }

                .table td {
                font-family: "Inria Sans", sans-serif;
                }
            </style>
                """, unsafe_allow_html=True)
    
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
    # start = time.perf_counter()
    st.markdown("### Image information")
    st.write("")
    st.markdown(f"""
    <table class='table'>
    <tr>
        <th>Parameter</th>
        <th>Original</th>
        <th>Edited</th>
    </tr>
    <tr>
        <td>Dimensions</td>
        <td>{original_w} × {original_h}</td>
        <td>{edited_w} × {edited_h}</td>
    </tr>
    <tr>
        <td>Format</td>
        <td>{img_format}</td>
        <td>{img_format}</td>
    </tr>
    <tr>
        <td>Size</td>
        <td>{Actions.format_file_size(original_size)}</td>
        <td>{Actions.format_file_size(size_bytes)}</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
    # print(f"Time for rendering table: {time.perf_counter() - start}")

    return final_img

def render_export_settings(final_img):
    st.markdown("""
            <style>
                .st-key-download button {
                padding: 0.8rem;
                background: #FF2727;
                border-radius: 16px;
                transition: all 0.3s ease;
                }

                .st-key-download button:hover {
                background: transparent !important;
                }
                
                .st-key-download button p {
                font-family: "Inria Sans", sans-serif;
                }""", unsafe_allow_html=True)
    
    st.markdown("### Export settings")
    img_format = st.session_state["image_metadata"]["format"]
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
            index=0)
        
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