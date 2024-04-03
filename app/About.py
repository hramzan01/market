

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from streamlit_player import st_player

import matplotlib.pylab as plt

# import datetime
import requests

from PIL import Image
import os


st.set_page_config(page_title="Market", initial_sidebar_state="collapsed")


# Background
st.markdown(

    """
    <style>
    [data-testid="stHeader"] {
        background-color: #D16643;
    }
    </style>
    """
    """
    <style>
    [data-testid="stApp"] {
        background: linear-gradient(180deg, rgba(255,124,82,1) 0%, rgba(0,0,0) 47%, rgba(0,0,0) 100%);
        height:auto;
    }
    </style>
    """
    """
    <style>
    [data-testid="stSlider"] {
        background-color: #EEF0F4;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """
    """
    <style>
        .stPlotlyChart {
            border-radius: 10px;
            overflow: hidden; /* This is important to ensure the border radius is applied properly */
        }
    </style>
    """
    """
    <style>
        .vp-center {
            width: 80%;
            margin: 0px 0px 0px 0px;
            background-color: #ffffff;
            opacity: .4;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():

    # HOME: MARKET logo
    col1, col2, col3 = st.columns([2, 3, 2])

    image = Image.open('app/assets/logo.png')
    col2.image(image, use_column_width=True)

    st.markdown('')  # Empty markdown line for spacing
    st.markdown('')  # Empty markdown line for spacing
    st.markdown('')  # Empty markdown line for spacing


def page_home():
    st.markdown("""<center><iframe src="https://player.vimeo.com/video/929080587?h=deb89b82de&autoplay=1" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></center>""", unsafe_allow_html=True)
    st.markdown('')
    st.write("""MARKET is a trading platform offering users investment opportunities in collective ownership of solar farms. ☀️""")

    st.write("""By leveraging weather forecasts and solar panel analytics, it delivers tailored recommendations for buying, holding, or selling energy generated from within your community""")

    st.subheader("""Empowerment: 💪""", anchor='center')
    st.write("""
    Empower users to participate in renewable energy ownership, promoting sustainability and community engagement.
    """, align='center')
    st.subheader("""Optimization: ⏱️""")
    st.write("""
    Optimize users' trading decisions by leveraging weather forecasts and solar energy output predictions to maximize profitability.
    """)
    st.subheader("""Accessibility:🤙""")
    st.write("""
            Make renewable energy trading accessible to all, regardless of technical expertise and type of accomodation.
    """)

page_home()
