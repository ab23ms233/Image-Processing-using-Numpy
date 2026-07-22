"""
## Purpose
    Renders the homepage of the website.

## Responsibilities
- Displays the navigation bar, hero section, features, and tech stack.
- Handles image upload and navigation to the editor page.

## Dependencies
- streamlit: For rendering the web interface.
- Class_Actions: For handling image upload operations.
- components.navbar: For rendering the navigation bar.
- components.hero: For rendering the hero section.
- components.features_tech: For rendering the features and tech stack section.
"""

import streamlit as st
from Class_Actions import Actions

from components.navbar import render_navbar
from components.hero import render_hero
from components.features_tech import render_features_tech

# LANDING PAGE
def home_page(max_img_size):
    # Navigation menu
    render_navbar()

    # Hero section
    # Returns the uploaded file if any, else returns None
    uploaded_file = render_hero()

    # Padding and divider
    st.markdown("<div style='padding-top: 1rem'></div>", unsafe_allow_html=True)
    st.divider()

    # Features and tech section
    render_features_tech()
    
    # If file is uploaded
    if uploaded_file is not None:
        Actions.image_uploaded(uploaded_file, max_img_size)
        st.rerun()