import streamlit as st

def render_hero():
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

        return uploaded_file
        

