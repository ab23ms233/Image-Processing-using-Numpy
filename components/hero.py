import streamlit as st

def render_hero():
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

                .cta-text {
                font-size: 18px;
                font-family: "Inter", sans-serif;
                color: white;
                }
                
                [data-testid="stFileUploader"] {
                border: none;
                }       

                /* Upload button */
                [data-testid="stFileUploader"] button {
                    background-color: #04A7FF;
                    color: white;
                    border-radius: 20px;
                    border: none;
                    padding: 0.7rem 1.5rem;
                    font-size: 18px;
                    font-weight: 600;
                }

                [data-testid="stFileUploader"] button:hover {
                background: #0c8ce9;
                }
                </style>""", unsafe_allow_html=True)
    
    st.markdown("""
<style>
                .feature-card {
    position: relative;
    background: #0f1720;
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 22px 22px 22px 30px;
    overflow: hidden;

    transition:
        border-color 0.25s ease,
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

/* Left accent strip */
.feature-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 6px;
    height: 100%;
    border-radius: 18px 0 0 18px;
}

/* Hover animation */
.feature-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}
                </style>""", unsafe_allow_html=True)
    
    left, right = st.columns([1.5, 1], gap="small")

    with left:
        st.markdown("<div style='padding-top: 5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=headline> Image Processing<br>made easy.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class=sub-header>Enhance, transform and export images in real time.</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding-top: 3rem'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
    
    st.divider()

