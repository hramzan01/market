

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



def page_dashboard():
    from IPython.display import Image
    import requests
        # Output: User visuals (this is user dashboard)
    st.header('Your Output', divider='grey')


    day = st.slider("Select Days", 1,7,1)

    root = os.getcwd()
    data = pd.read_csv(os.path.join(root, 'app/data/final_prediction.csv'))

    # Create a Plotly figure
    fig = go.Figure()

    # Add a line trace to the figure

    fig.add_trace(go.Scatter(y=data['kwh'][0:(day*24)], mode='lines',fill='tozeroy', name='Line Chart'))


    # Set title and axes labels
    fig.update_layout(title='Line Chart', xaxis_title='X-axis', yaxis_title='Y-axis')
    fig.update_layout(
        plot_bgcolor='#EEF0F4',  # Transparent background
        paper_bgcolor='#EEF0F4'   # Transparent background
    )
    # fig.update_traces(line=dict(color='red', width=2))  # Change 'red' to your desired color and 2 to your desired thickness

    # Display the Plotly figure in Streamlit

    st.plotly_chart(fig)
    wmo_url = 'https://gist.githubusercontent.com/stellasphere/9490c195ed2b53c707087c8c2db4ec0c/raw/76b0cb0ef0bfd8a2ec988aa54e30ecd1b483495d/descriptions.json'
    wmo_description = requests.get(wmo_url).json()
    image = wmo_description['2']['day']['image']
    total = round(data['kwh'].sum())
    col1, col2, col3 = st.columns(3)
    col1.image(image)
    col2.metric("Avg Daily Power",total, 'kWh')
    col3.metric("Weather",wmo_description['2']['day']['description'])

page_dashboard()
