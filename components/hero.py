import streamlit as st
from typing import Any
from streamlit_image_comparison import image_comparison
from utils.paths import COVER_EDITED, COVER_ORIGINAL

def render_hero() -> Any | None:
    st.markdown("""
            <style>

                .headline {
                font-size: 4rem;
                color: white;
                font-family: "Inria Serif", serif;
                line-height: 5rem;
                }
                
                .sub-header {
                font-size: 1.2rem;
                font-family: "Inria Sans", sans-serif;
                color: white;
                }

                .st-key-upload_image button {
                width: 10rem;
                height: 3.3rem;
                background: #FF2727; 
                border-radius: 16px;
                padding: 20px;
                transition: all 0.3s ease;
                }

                .st-key-upload_image button:hover {
                background: transparent;
                }

                .st-key-upload_image button:focus {
                background: transparent !important;
                }

                .st-key-upload_image button p {
                font-weight: 600;
                font-size: 1rem;
                }
            </style>""", unsafe_allow_html=True)
    
    # Initially, no file is uploaded, hence uploaded_file is set to None. 
    uploaded_file = None
    left, right = st.columns(2, gap="small")

    # Headline, subheader and upload button
    with left:
        st.markdown("<div style='padding-top: 5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=headline> Image Processing<br>made easy.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=sub-header>Enhance, transform and export images in real time.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)

        # Initialise state for detecting if upload button is clicked
        if "upload_clicked" not in st.session_state:
            st.session_state["upload_clicked"] = False
        # Upload button. "upload_clicked" checks if upload button is clicked 
        if st.button("Upload Image", key="upload_image"):
            st.session_state["upload_clicked"] = True
        # Render file uploader if upload button is clicked
        if st.session_state["upload_clicked"]:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

    # with right:
    #     image_comparison(
    #         img1=str(COVER_ORIGINAL),
    #         img2=str(COVER_EDITED),
    #         label1="Original",
    #         label2="Edited",
    #         width=900
    #     )
        # Return the uploaded file if it exists, else return None
        return uploaded_file
        

