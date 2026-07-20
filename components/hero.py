import streamlit as st
from Class_Actions import Actions

def render_hero(max_img_size):
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inria+Sans:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Inria+Serif:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
                
                .headline {
                font-size: 60px;
                color: white;
                font-family: "Inria Serif", serif;
                line-height: 5rem;
                }
                
                .sub-header {
                font-size: 20px;
                font-family: "Inria Sans", sans-serif;
                color: white;
                }

                .st-key-upload_image button {
                width: 10rem;
                height: 3.3rem;
                background: #04A7FF;
                border-radius: 16px;
                padding: 20px;
                transition: all 0.3s ease;
                }

                .st-key-upload_image button:hover {
                background: #020617;
                color: #04A7FF;
                border: 2px solid #04A7FF;
                }

                .st-key-upload_image button:focus {
                background: #020617 !important;
                color: #04A7FF !important;
                border: 2px solid #04A7FF !important;
                }

                .st-key-upload_image button p {
                font-weight: 600;
                font-family: "Inter", sans-serif;
                font-size: 1rem;
                }
            </style>""", unsafe_allow_html=True)
    
    uploaded_file = None
    left, right = st.columns([1.5, 1], gap="small")

    with left:
        st.markdown("<div style='padding-top: 5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=headline> Image Processing<br>made easy.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=sub-header>Enhance, transform and export images in real time.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)

        if "upload_clicked" not in st.session_state:
            st.session_state["upload_clicked"] = False

        if st.button("Upload Image", key="upload_image"):
            st.session_state["upload_clicked"] = True

        if st.session_state["upload_clicked"]:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

        # If file is uploaded
        if uploaded_file is not None:
            st.session_state["image"] = Actions.image_uploaded(uploaded_file, max_img_size)
            st.rerun()
        

