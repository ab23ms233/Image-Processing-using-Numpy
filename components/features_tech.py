import streamlit as st
from utils.paths import PYTHON, NUMPY, PILLOW, SCIPY

def feature_card(heading, text, accent):
    st.markdown(f"""
                <div class='feature-cards' style='border-left: 4px solid {accent};'>
                    <div class='options'>{heading}</div>
                    <div class='divider'></div>
                    <div class='text'>{text}</div>
                </div>""", unsafe_allow_html=True) 

def tech_cards(name, icon_path):
    st.markdown(f"""
                <div class='tech-cards'>
                    <div class='icons'><img src={icon_path}></div>
                    <div class='tech-names'>{name}</div>
                </div>
                """, unsafe_allow_html=True)
    
def render_features_tech():
    st.markdown("""
                <style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300..700&family=Inria+Sans:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&display=swap');

                .headers {
                font-size: 1.5rem;
                font-family: "Inria Sans", sans-serif;
                color: white;
                }
                
                .options {
                font-size: 1.2rem;
                font-family: "Fira Code", monospace;
                color: white;
                text-align: center;
                }
                
                .text {
                font-family: "Inria Sans", sans-serif;
                font-size: 1rem;
                color: white
                }
                
                .feature-cards {
                width: 18rem;
                height: 9rem;
                background-color: #0F1720;
                border-radius: 16px;
                border: 2px solid #26374A;
                padding: 20px;
                transition: all 0.3s ease;
                }

                .divider {
                margin: 12px 0;
                border-top: 1px solid rgba(255,255,255,0.4);
                }

                .feature-cards:hover {
                transform: translateY(-4px);
                box-shadow: 5px 5px #26374A;
                }
                
                .tech-cards {
                width: 15rem;
                height: 4rem;
                display: flex;
                flex-direction: row;
                gap: 0.5rem;
                justify-content: center;
                align-items: center;
                background-color: #D9D9D9;
                border-radius: 16px;
                color: black;
                }
                
                .tech-names {
                text-align: center;
                font-size: 1rem;
                font-family: "Fira Code", monospace;
                }

                .icons {
                width: 1rem;
                }""", unsafe_allow_html=True)
    
    st.markdown("<div class='headers'>Features</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding-top: 2rem'></div>", unsafe_allow_html=True)

    card1, card2, card3, card4 = st.columns(4, gap="small")
    with card1:
        feature_card("Transform", "Rotate and flip.", "#9F8A6F")
    with card2:
        feature_card("Color", "Negative, grayscale and binarize.", "#5EE9F0")
    with card3:
        feature_card("Filters", "Sharpen, blur and detect edges.", "#C0F654")
    with card4:
        feature_card("Export", "Compress and convert to your desired format.", "#D67FD8")

    st.markdown("<div style='margin-top: 5rem;'></div", unsafe_allow_html=True)
    st.markdown("<div class='headers'>Tech Stack</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding-top: 2rem'></div>", unsafe_allow_html=True)

    tech1, tech2, tech3, tech4, tech5 = st.columns(5, gap="small")
    with tech1:
        print(str(PYTHON))
        tech_cards("Python", str(PYTHON))
    with tech2:
        tech_cards("NumPy", str(NUMPY))
    with tech3:
        tech_cards("Pillow", str(PILLOW))
    with tech4:
        tech_cards("SciPy", str(SCIPY))



 