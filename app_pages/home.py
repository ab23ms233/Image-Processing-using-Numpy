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
    uploaded_file = render_hero()

    # Padding and divider
    st.markdown("<div style='padding-top: 1rem'></div>", unsafe_allow_html=True)
    st.divider()

    # Features and tech section
    render_features_tech()
    
    # If file is uploaded
    if uploaded_file is not None:
        st.session_state["image"] = Actions.image_uploaded(uploaded_file, max_img_size)
        st.rerun()
    # # Right side
    # with hero_right:
    #     # start = time.perf_counter()

    #     # Image
    #     @st.cache_data
    #     def load_image():
    #         image = Image.open("images/deep-mind-compressed.webp").convert("RGB")
    #         return image
        
    #     start = time.perf_counter()
    #     hero_img = load_image()
    #     print(f"Time for loading image: {time.perf_counter()-start}")
    #     start = time.perf_counter()
    #     st.image(hero_img, use_container_width=True)
    #     print(f"Time for rendering image: {time.perf_counter()-start}")
    #     # Description
    #     st.markdown("<p style='text-align: center;'>Photo by <a href='https://unsplash.com/@googledeepmind' target='_blank'>Google DeepMind</a></p>",
    # unsafe_allow_html=True)
        # print(f"Time for rendering hero_right: {time.perf_counter() - start}")
    
    
