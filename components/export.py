import time
import streamlit as st
from io import BytesIO
from PIL import Image
from Class_Actions import Actions
from numpy import ndarray

def render_img_info(edited_img_arr: ndarray):
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

                .edited {
                color: #00ff00;
                }
                
                .original {
                color: cyan;
                }
            </style>
                """, unsafe_allow_html=True)
    
    # Original image metadata
    original_h, original_w = st.session_state["image_metadata"]["original_h"], st.session_state["image_metadata"]["original_w"]
    original_cmode = Actions.get_color_mode(st.session_state["ori_img"].img_arr)

    # Edited image metadata
    shape = edited_img_arr.shape
    edited_h, edited_w = shape[0], shape[1]
    edited_cmode = Actions.get_color_mode(edited_img_arr)

    # Rendering image info table
    # start = time.perf_counter()
    st.markdown("### Image information")
    st.write("")
    st.markdown(f"""
    <table class='table'>
    <tr>
        <th>Parameter</th>
        <th class="original">Original</th>
        <th class="edited">Edited</th>
    </tr>
    <tr>
        <td>Dimensions</td>
        <td class="original">{original_w} × {original_h}</td>
        <td class="edited">{edited_w} × {edited_h}</td>
    </tr>
    <tr>
        <td>Color mode</td>
        <td class="original">{original_cmode}</td>
        <td class="edited">{edited_cmode}</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
    # print(f"Time for rendering table: {time.perf_counter() - start}")

def render_export_settings():
    st.markdown("""
            <style>
                .st-key-download button {
                padding: 0.8rem;
                background: #FF2727;
                border-radius: 16px;
                }

                .st-key-download button:hover {
                background: transparent !important;
                }

                .st-key-download button:active {
                background: transparent !important;
                color: #FF4B4B !important;
                transform: scale(0.95);
                }

                .st-key-download button:focus {
                background: transparent !important;
                }
                
                .st-key-download button p {
                font-family: "Inria Sans", sans-serif;
                }""", unsafe_allow_html=True)

    # Initialising variables to check if these stay same across sessions
    if "export_quality" not in st.session_state:
        st.session_state["export_quality"] = None
    if "export_format" not in st.session_state:
        st.session_state["export_format"] = None
    if "comp_sizeb" not in st.session_state:
        st.session_state["comp_sizeb"] = 100
    if "buffer" not in st.session_state:
        st.session_state["buffer"] = None

    st.markdown("### Export settings")
    img_format = st.session_state["image_metadata"]["format"]
    file_name_col, format_col, size_col = st.columns(3, gap="large")

    with file_name_col:
        file_name = st.text_input(
            "File Name",
            # value="edited_image",
            placeholder="Enter output file name") 
        
        if file_name == "":
            file_name = "edited_image"

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
            quality = st.slider("Quality", 1, 100, 85)
        else:
            quality = None

    # print(quality, format_option)
    if st.session_state["export_format"] != format_option or st.session_state["export_quality"] != quality:
        print(f"Edited: {quality}, {format_option}")
        st.session_state["export_quality"] = quality
        st.session_state["export_format"] = format_option

        buffer = BytesIO()
        final_img = Image.fromarray(st.session_state["curr_img"].img_arr.astype("uint8"))
        final_img.save(buffer, format=format_option, quality=quality)
        compressed_size_bytes = len(buffer.getvalue())
        st.session_state["comp_sizeb"] = compressed_size_bytes
        st.session_state["buffer"] = buffer

    extension = format_option.lower()
    size_col.metric("Export Size", Actions.format_file_size(st.session_state["comp_sizeb"]))
    st.download_button(
            label="Download Image",
            data=st.session_state["buffer"].getvalue(),
            file_name=f"{file_name}.{extension}",
            key="download")