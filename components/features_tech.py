"""
## Purpose:
    Renders the features and tech stack section of the HOMEPAGE.

## Responsibilities:
- Displays feature cards describing the app capabilities.
- Shows technology cards for the libraries used in the project.

## Functions used:
- encode_icon: Encodes an icon image to base64 for embedding in HTML.
- feature_card: Renders a feature card with heading, text, and accent color.
- tech_cards: Renders a tech card with name, icon, and link.

## Dependencies:
- streamlit: For rendering the web interface.
- utils.paths: For accessing paths to tech stack icons.
- base64: For converting local images/icons to encoded files for embedding in HTML
"""

import streamlit as st
from utils.paths import PYTHON, NUMPY, PILLOW, SCIPY, STREAMLIT
from pathlib import Path
import base64
from typing import Union

# Function for encoding the icon image of tech stack to base64 for embedding in HTML
def encode_icon(icon_path: Union[str, Path]) -> str:
    icon_path = Path(icon_path)
    suffix = icon_path.suffix.lower()
    mime_type = "image/svg+xml" if suffix == ".svg" else f"image/{suffix[1:]}"
    encoded_bytes = base64.b64encode(icon_path.read_bytes())
    return f"data:{mime_type};base64,{encoded_bytes.decode('ascii')}"

# Function to render a feature card with a heading, text, and color
def feature_card(heading: str, text: str, accent: str):
    st.markdown(f"""
                <div class='feature-cards' style='border-left: 5px solid {accent};'>
                    <div class='options'>{heading}</div>
                    <div class='divider'></div>
                    <div class='text'>{text}</div>
                </div>""", unsafe_allow_html=True)

# Function to render a tech card with a name, icon, and link
def tech_cards(name: str, icon_path: Union[str, Path], link: str):
    icon_src = encode_icon(icon_path)
    st.markdown(f"""
                <a href="{link}" target="_blank" class="tech-card-link">
                    <div class='tech-card'>
                        <div class='icon'><img src="{icon_src}" alt="{name} logo"/></div>
                        <div class='tech-name'>{name}</div>
                    </div>
                </a>
                """, unsafe_allow_html=True)
    

def render_features_tech():
    st.markdown("""
                <style>

                .headers {
                font-size: 1.5rem;
                font-family: "Inria Sans", sans-serif;
                }
                
                .options {
                font-size: 1.2rem;
                font-family: "Fira Code", monospace;
                text-align: center;
                }
                
                .text {
                font-family: "Inria Sans", sans-serif;
                font-size: 1rem;
                }
                
                .feature-cards {
                width: 18rem;
                height: 9rem;
                background-color: #0F1720;
                border-radius: 16px;
                border: 2px solid rgba(255, 75, 75, 0.2);
                padding: 20px;
                transition: all 0.3s ease;
                }

                .divider {
                margin: 12px 0;
                border-top: 1px solid rgba(255,255,255,0.4);
                }

                .feature-cards:hover {
                transform: translateY(-4px);
                box-shadow: 5px 5px rgba(255, 75, 75, 0.2);
                }
                
                .tech-card {
                height: 4rem;
                display: flex;
                flex-direction: row;
                gap: 0.8rem;
                justify-content: center;
                align-items: center;
                background-color: #D9D9D9;
                border-radius: 16px;
                color: black;
                transition: all 0.3s ease;
                }

                .tech-card:hover {
                transform: scale(1.03);
                }
                
                .tech-name {
                text-align: center;
                font-size: 1rem;
                font-family: "Fira Code", monospace;
                }

                .icon img {
                width: 2.5rem;
                height: auto;
                }

                .tech-card-link {
                text-decoration: none !important;
                }
                </style>
                """, unsafe_allow_html=True)
    
    # FEATURES header
    st.markdown("<div class='headers'>Features</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding-top: 2rem'></div>", unsafe_allow_html=True)

    # FEATURE cards
    card1, card2, card3, card4 = st.columns(4, gap="small")
    with card1:
        feature_card("Transform", "Rotate and flip.", "#9F8A6F")
    with card2:
        feature_card("Color", "Negative, grayscale and binarize.", "#5EE9F0")
    with card3:
        feature_card("Filters", "Sharpen, blur and detect edges.", "#C0F654")
    with card4:
        feature_card("Export", "Compress and convert to your desired format.", "#D67FD8")

    st.markdown("<div style='margin-top: 5rem;'></div>", unsafe_allow_html=True)

    # TECH STACK header
    st.markdown("<div class='headers'>Tech Stack</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding-top: 2rem'></div>", unsafe_allow_html=True)
    
    # TECH STACK cards
    tech1, tech2, tech3, tech4, tech5 = st.columns(5, gap="medium")
    with tech1:
        tech_cards("Python", str(PYTHON), link="https://www.python.org/")
    with tech2:
        tech_cards("NumPy", str(NUMPY), link="https://numpy.org/")
    with tech3:
        tech_cards("Pillow", str(PILLOW), link="https://pillow.readthedocs.io/en/stable/")
    with tech4:
        tech_cards("SciPy", str(SCIPY), link="https://scipy.org/")
    with tech5:
        tech_cards("Streamlit", str(STREAMLIT), link="https://streamlit.io/")



 