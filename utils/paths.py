"""
Contains the paths to all the images/icons used throughout the website. Anytime an image is accessed, the path from this file is used.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ICONS = PROJECT_ROOT/"icons"
IMAGES = PROJECT_ROOT/"images"

LOGO = ICONS/"logo.png"
LOGO_REMOVEBG = ICONS/"logo-removebg.png"
COVER_ORIGINAL = IMAGES/"cover.jpg"
COVER_EDITED = IMAGES/"edited.jpeg"
PYTHON = ICONS/"python.svg"
PILLOW = ICONS/"pillow.png"
SCIPY = ICONS/"scipy.svg"
NUMPY = ICONS/"numpy.svg"
STREAMLIT = ICONS/"streamlit.svg"
