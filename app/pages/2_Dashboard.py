

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
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px
from PIL import Image
import time
import os
st.set_page_config(page_title="Market", initial_sidebar_state="collapsed", layout='wide')


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
    battery_size = 5
    battery_charge = 3
    postcode = 'E1 5DY'
    name = 'Haaris'
    # Return Lat & Lon from postcode
    base_url = 'https://api.postcodes.io/postcodes'
    response = requests.get(f'{base_url}/{postcode}').json()
    lat = response['result']['latitude']
    lon = response['result']['longitude']
    # Output: User visuals (this is user dashboard)
    with st.form(key='params_for_api'):
        # Test_API_predict
        params = {
            #'date': f'{selected_date} 00:00:00',
            'battery_size': 5,
            'battery_charge': 3
        }
        api_url = 'https://marketpricelight1-d2w7qz766q-ew.a.run.app/predict?battery_size=5&battery_charge=1'
        complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])
        # complete_url
        # Generate Dashboard when submit is triggered
        if st.form_submit_button('CHARGE UP :battery:', use_container_width=True):
            # Simulate progress bar
            progress_text = "Charging up... Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            # Make API call
            response = requests.get(api_url, params=params)
            data = response.json()
            saleprice = data['prediction_data']['SalePrice_p/kwh']
            buyprice = data['prediction_data']['PurchasePrice_p/kwh']
            power_gen = data['prediction_data']['Generation_kwh']
            power_cons = data['prediction_data']['Consumption_kwh']
            res_opt_batt = data['res_opt_batt']['0']
            res_opt_buyprice = data['res_opt_buyprice']['0']
            res_opt_sellprice = data['res_opt_sellprice']['0']
            res_opt_baseprice = data['res_opt_baseprice']['0']
            x_sale, y_sale = zip(*saleprice.items()) # unpack a list of pairs into two tuples
            x_buy, y_buy = zip(*buyprice.items())
            x_gen, y_gen = zip(*power_gen.items())
            x_cons, y_cons = zip(*power_cons.items())
            x_battopt, y_battopt = zip(*res_opt_batt.items())
            x_bpopt, y_bpopt = zip(*res_opt_buyprice.items())
            x_spopt, y_spopt = zip(*res_opt_sellprice.items())
            x_basep, y_basep = zip(*res_opt_baseprice.items())
            dates = pd.to_datetime(x_buy)
            # VISUALS
            # plotly map
            @st.cache_data
            def london_map(lat, lon):
                # Create a Plotly figure with Mapbox
                fig = go.Figure(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=13,
                        color='orange'
                    ),
                    text=['London']
                ))
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=5, r=5, t=0, b=40),
                    hovermode='closest',
                    mapbox=dict(
                        style='carto-positron',
                        bearing=0,
                        center=dict(
                            lat=lat,
                            lon=lon
                        ),
                        pitch=0,
                        zoom=16
                    ),
                    width=1280,
                    height=400
                )
                return fig
            st.write(london_map(lat, lon))
            # Header
            st.subheader(f"{name}'s Energy Hub")
            st.markdown('')  # Empty markdown line for spacing
            st.markdown('')  # Empty markdown line for spacing
            # Split the remaining space into three columns
            col0, col1, col2 = st.columns(3)
            # Display images
            image1 = Image.open('app/assets/money.png').resize((115, 100))
            image2 = Image.open('app/assets/energy.png').resize((100, 100))
            image3 = Image.open('app/assets/battery.png').resize((55, 100))
            with col0:
                st.image(image1, use_column_width=False)
            with col1:
                st.image(image2, use_column_width=False)
            with col2:
                st.image(image3, use_column_width=False)
            st.markdown('')  # Empty markdown line for spacing
            # Split the remaining space into three columns
            col3, col4, col5 = st.columns(3)
            # First column: Buy vs Sell Price
            with col3:
                # Buy vs Sell Price
                fig = px.line(x=dates, y=y_sale, labels={'x': 'Date', 'y': 'Price'}, title='Buy vs Sell Price')
                fig.add_scatter(x=dates, y=y_buy, mode='lines', name='Buy Price')
                st.plotly_chart(fig)
            with col4:
                # Power gen vs power con
                fig_power = px.line(x=dates, y=[y_gen, y_cons], labels={'x': 'Date', 'y': 'Power'}, title='Power Generation vs Consumption')
                st.plotly_chart(fig_power)
            with col5:
                # Battery Output
                fig_battopt = px.area(x=x_battopt, y=y_battopt, labels={'x': 'Date', 'y': 'Battery Output'}, title='Battery Output')
                fig_battopt.update_layout(width=400)
                st.plotly_chart(fig_battopt)
            # Get definitiion for weather WMO codes
            wmo_url = 'https://gist.githubusercontent.com/stellasphere/9490c195ed2b53c707087c8c2db4ec0c/raw/76b0cb0ef0bfd8a2ec988aa54e30ecd1b483495d/descriptions.json'
            wmo_description = requests.get(wmo_url).json()
            # Forecast header
            st.subheader('7 Day Energy Forecast')
            st.markdown('')  # Empty markdown line for spacing
            st.markdown('')  # Empty markdown line for spacing
            # get images for weather
            image1 = wmo_description['2']['day']['image']
            image2 = wmo_description['3']['day']['image']
            image3 = wmo_description['45']['day']['image']
            image4 = wmo_description['53']['day']['image']
            image5 = wmo_description['53']['day']['image']
            image6 = wmo_description['45']['day']['image']
            image7 = wmo_description['3']['day']['image']
            # Split the columns for 7 images for 7 days of week
            mon, tue, wed, thu, fri, sat, sun = st.columns(7)
            mon.image(image1)
            mon.text('DAY 1')
            tue.image(image2)
            tue.text('DAY 2')
            wed.image(image3)
            wed.text('DAY 3')
            thu.image(image4)
            thu.text('DAY 4')
            fri.image(image5)
            fri.text('DAY 5')
            sat.image(image6)
            sat.text('DAY 6')
            sun.image(image7)
            sun.text('DAY 7')


page_dashboard()
