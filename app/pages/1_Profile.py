

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from streamlit_extras.switch_page_button import switch_page

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
    col1, col2, col3 = st.columns(3)
    try:
        image = Image.open(f'app/assets/{st.session_state.house}.png').resize((100, 100))
        col1.image(image, use_column_width=True)
        col2.subheader(f"""{st.session_state.name}'s Profile""")
    except:
        #image = Image.open(f'app/assets/Bungalow.png').resize((100, 100))
        #col1.image(image, use_column_width=True)
        col2.subheader("")
    Name = st.text_input("Name", "")
    Postcode = st.text_input("Postcode", "")
    #House_price = st.number_input("House price", step=10000)
    #Income = st.number_input("Income", step=10000)
    Battery_Size = st.number_input("Battery Size", step=1)
    Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    House_type = ["Bungalow","Terrace house", "Detached house", "Flat", "Semi-detached house"]
    #selected_date = st.date_input('Select a date', datetime.today())
    selected_house = st.selectbox("Select an option", House_type, key="house")
    st.button("Submit")
    st.session_state.name = Name
    st.session_state.postcode = Postcode

def page_profile():
    col1, col2, col3 = st.columns(3)
    try:
        image = Image.open(f'app/assets/{st.session_state.house}.png').resize((100, 100))
        col1.image(image, use_column_width=True)
        col2.subheader(f"""{st.session_state.name}'s Profile""")
    except:
        #image = Image.open(f'app/assets/Bungalow.png').resize((100, 100))
        #col1.image(image, use_column_width=True)
        col2.subheader("")
    Name = st.text_input("Name", "")
    Postcode = st.text_input("Postcode", "")
    #House_price = st.number_input("House price", step=10000)
    #Income = st.number_input("Income", step=10000)
    Battery_Size = st.number_input("Battery Size", step=1)
    Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    House_type = ["Bungalow","Terrace house", "Detached house", "Flat", "Semi-detached house"]
    #selected_date = st.date_input('Select a date', datetime.today())
    selected_house = st.selectbox("Choose your house type", House_type, key="house")
    st.button("Submit")
    st.session_state.name = Name
    st.session_state.postcode = Postcode

def page_profileV2():
    col1, col2= st.columns([1,3])
    try:
        image = Image.open(f'app/assets/{st.session_state.house}.png').resize((100, 100))
        col1.image(image, use_column_width=False)
        #col2.subheader(f"""{st.session_state.name}'s Profile""")
    except:
        image = Image.open(f'app/assets/Bungalow.png').resize((100, 100))
        col1.image(image, use_column_width=False)
        #col2.subheader("")
    col2.text_input("Username", "", key="name")
    House_type = ["Bungalow","Terrace house", "Detached house", "Flat", "Semi-detached house"]
    #selected_date = st.date_input('Select a date', datetime.today())
    col2.selectbox("Choose your house type", House_type, key="house")
    col3, col4= st.columns([1,3])
    try:
        image = Image.open(f'app/assets/{st.session_state.Bat_type}.png').resize((100, 100))
        col3.image(image, use_column_width=False)
        #col2.subheader(f"""{st.session_state.name}'s Profile""")
    except:
        image = Image.open(f'app/assets/Small battery.png').resize((100, 100))
        col3.image(image, use_column_width=False)
        #col2.subheader("")
    battery_type = ["Small battery", "Large battery", "Electric vehicle"]
    col4.selectbox("Choose your battery size", battery_type, key="Bat_type")
    col4.number_input("Battery Charge (%)", min_value=0, max_value=100, step=1)
    col5, col6= st.columns([1,3])
    try:
        if st.session_state.Num_solar < 5:
            image = Image.open(f'app/assets/Single solar.png').resize((100, 100))
            col5.image(image, use_column_width=False)
        elif 5 < st.session_state.Num_solar < 10:
            image = Image.open(f'app/assets/Double solar.png').resize((100, 100))
            col5.image(image, use_column_width=False)
        elif 10 < st.session_state.Num_solar < 15:
            image = Image.open(f'app/assets/Triple solar.png').resize((100, 100))
            col5.image(image, use_column_width=False)
        else:
            image = Image.open(f'app/assets/Quad solar.png').resize((100, 100))
            col5.image(image, use_column_width=False)
        #col2.subheader(f"""{st.session_state.name}'s Profile""")
    except:
        image = Image.open(f'app/assets/Double solar.png').resize((100, 100))
        col5.image(image, use_column_width=False)
        #col2.subheader("")
    #House_price = st.number_input("House price", step=10000)
    #Income = st.number_input("Income", step=10000)
    col6.number_input("Solar Size (kW)", step=1, max_value=25, key="Num_solar")
    #Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    col6.text_input("Postcode", "", key="postcode")
    col6.button("Submit")

    submit = st.button("Submit")
    if submit:
        switch_page("Dashboard")

page_profileV2()
