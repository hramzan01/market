

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


import matplotlib.pylab as plt

# import datetime
import requests

from PIL import Image
from streamlit_option_menu import option_menu
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


def page_profile():
    Postcode = st.text_input("Postcode", "")
    House_price = st.number_input("House price", step=10000)
    Income = st.number_input("Income", step=10000)
    Battery_Size = st.number_input("Battery Size", step=1)
    Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    House_type = ["Bungalow","Terraced house", "Detached house", "Flat or maisonette", "Semi-detached house"]
    selected_date = st.date_input('Select a date', datetime.today())

    selected_option = st.selectbox("Select an option", House_type)


page_profile()
