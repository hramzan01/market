from streamlit.components.v1 import html
from datetime import datetime
import matplotlib.pylab as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px
from PIL import Image
import time
import os
import numpy as np
from datetime import date, timedelta



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
    postcode = 'E2 8DY'
    name = 'Le Wagon LDN'

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
        api_url = 'https://marketpricelightver4-d2w7qz766q-ew.a.run.app/predict?battery_size=5&battery_charge=3&solar_size=5'
        complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])

        # Generate Dashboard when submit is triggered
        if st.form_submit_button('CHARGE ⚡️', use_container_width=True):

            # # Simulate progress bar
            # progress_text = "Charging up... Please wait."
            # my_bar = st.progress(0, text=progress_text)
            # for percent_complete in range(100):
            #     time.sleep(0.2)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
            # time.sleep(1)
            # my_bar.empty()

            with st.spinner('charging up your dashboard...'):

                # Make API call
                response = requests.get(api_url, params=params)
                data = response.json()
                weather = data['res_weather_code']
                saleprice = data['prediction_saleprice']
                buyprice = data['prediction_purchaseprice']
                power_gen = data['prediction_gen']
                power_cons = data['prediction_cons']
                res_opt_batt = data['res_opt_batt']['0']
                res_opt_buyprice = data['res_opt_buyprice']['0']
                res_opt_sellprice = data['res_opt_sellprice']['0']
                res_opt_baseprice = data['res_opt_baseprice']['0']
                x_sale, y_sale = zip(*saleprice.items()) # unpack a list of pairs into two tuples
                x_buy, y_buy = zip(*buyprice.items())
                #x_gen, y_gen = zip(*power_gen.items())
                x_cons, y_cons = zip(*power_cons.items())
                x_battopt, y_battopt = zip(*res_opt_batt.items())
                x_bpopt, y_bpopt = zip(*res_opt_buyprice.items())
                x_spopt, y_spopt = zip(*res_opt_sellprice.items())
                x_basep, y_basep = zip(*res_opt_baseprice.items())
                dates = pd.to_datetime(x_buy)

                ### WEATHER
                # Import datetime module
                from datetime import datetime

                # Initialize an empty list to store the weather codes at midday
                midday_weather_codes = []

                # Iterate over the hourly weather codes
                for index, weather_code in enumerate(weather):
                    # Calculate the hour of the day using index (assuming index starts from 0)
                    hour_of_day = index % 24

                    # Check if the hour is midday (12:00 PM)
                    if hour_of_day == 12:
                        # Add the weather code to the list
                        midday_weather_codes.append(weather_code)

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
                            size=30,
                            color='orange',
                        ),
                        text=['London']
                    ))
                    fig.update_layout(
                        autosize=True,
                        margin=dict(l=0, r=0, t=0, b=0),
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
                        height=400,
                    )
                    return fig
                st.write(london_map(lat, lon))

                # Main optimised prohet graph
                st.divider()
                # Battery Output
                #fig_final = px.area(x=x_battopt, y=y_battopt, labels={'x': 'Date', 'y': 'Battery Output'}, title='Battery Output')
                #fig_final.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', width=600, height=400)
                #fig_final.update_layout(width=400)
                #fig_final.update_layout(width=1280)
                #st.plotly_chart(fig_final)


                # Header
                st.subheader(f"{name}'s Energy Hub")
                st.divider()
                st.markdown('')  # Empty markdown line for spacing

                # Tracker cards
                track1, track2, track3 = st.columns(3)
                track1.metric("Money Saved YTD", "£96.20", "£5.25 vs 2023")
                track2.metric("Energy Saved YTD", "⌁568kWh", "0.46% vs 2023")
                track3.metric("Energy Sold YTD", "⌁451kWh", "+4.87% vs 2023")
                st.markdown('')  # Empty markdown line for spacing

                # Split the remaining space into three columns
                col0, col1, col2 = st.columns(3)

                # Display images
                st.markdown('')  # Empty markdown line for spacing
                st.markdown('')  # Empty markdown line for spacing

                image1 = Image.open('app/assets/money.png').resize((100, 100))
                image2 = Image.open('app/assets/energy.png').resize((100, 100))
                image3 = Image.open('app/assets/battery.png').resize((100, 100))
                with col0:
                    st.image(image1, use_column_width=False)
                with col1:
                    st.image(image2, use_column_width=False)
                with col2:
                    st.image(image3, use_column_width=False)
                #attempt at new graph

                # convert model price dictionary into numpy array and cumulative sum
                result = data['res_delta_buy_sell_price']['0'].items()
                graph_data = list(result)
                model = np.asarray(np.array(graph_data)[:,1], dtype=float).cumsum()/100
                # convert res_baseline_price_no_solar:  price dictionary into numpy array and cumulative sum
                result = data['res_baseline_price_no_solar']['0'].items()
                graph_data = list(result)
                baseline_no_solar = np.asarray(np.array(graph_data)[:,1], dtype=float).cumsum()/100
                # convert res_opt_baseprice:  price dictionary into numpy array and cumulative sum
                result = data['res_opt_baseprice']['0'].items()
                graph_data = list(result)
                baseline = np.asarray(np.array(graph_data)[:,1], dtype=float).cumsum()/100
                # Set up date range - will need to be imported from the streamlit
                today = date.today()
                first_date = today
                last_date = today + timedelta(days=7)
                date_range = pd.date_range(start = first_date, end=last_date, freq = 'h')
                date_range = date_range[:168]
                import matplotlib.pyplot as plt
                #plt.plot(date_range, model, label='Optimised Price')
                #plt.plot(date_range, baseline, label = 'Unoptimised Price')
                #plt.plot(date_range, baseline_no_solar, label = 'Price No Solar')
                #plt.ylabel('Weekly Cost (£)')
                #plt.legend()

                # Battery Output
                df = pd.DataFrame({'date': date_range ,'Solar plus Market': model, 'Solar': baseline, 'No Solar': baseline_no_solar})
                #fig_final = px.line(x=date_range, y=[model, baseline, baseline_no_solar], labels={'x': 'Date', 'y': 'Cumulative Cost', 'wide_variable_0': 'Solar plus Market', 'wide_variable_1': 'Solar', 'wide_variable_2': 'Baseline'}, title='Total Savings')
                fig_final = px.line(df, x='date', y=['No Solar', 'Solar', 'Solar plus Market'], labels={'x': 'Date', 'y': 'Cumulative Cost'}, title='Forcasted Weekly Cost') #Remove date and update y axis lable
                fig_final.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', width=600, height=400)
                fig_final.update_layout(width=400)
                fig_final.update_layout(width=1280)
                st.plotly_chart(fig_final)
                # Split the remaining space into three columns
                col3, col4, col5 = st.columns(3)
                st.divider()

                # First column: Buy vs Sell Price
                # Define a common color for all lines
                color = 'orange'
                with col3:
                    # Buy vs Sell Price
                    fig = px.line(x=date_range, y=y_sale, labels={'x': 'Date', 'y': 'Price (£)'}, title='FORECASTED ENERGY PRICE')
                    fig.update_layout(
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        width=600,
                        height=400,
                        showlegend=False, # Hide legend
                        yaxis_range=[0,25]
                    )
                    fig.add_scatter(x=date_range, y=y_buy, mode='lines', name='Buy Price')
                    fig.add_scatter(x=date_range, y=y_sale, mode='lines', name='Sell Price')
                    fig.update_layout(width=400)
                    st.plotly_chart(fig)


                with col4:
                    # Power gen vs power con
                    fig_power = px.line(x=date_range, y=[power_gen, y_cons], labels={'x': 'Date', 'y': 'Energy (kWh)'}, title='FORECASTED GEN & USE')
                    fig_power.update_layout(
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        width=600,
                        height=400,
                        showlegend=False  # Hide legend
                    )
                    fig_power.add_scatter(x=date_range, y=power_gen, mode='lines', name='generated')
                    fig_power.add_scatter(x=date_range, y=y_cons, mode='lines', name='consumed')
                    fig_power.update_layout(width=400)
                    st.plotly_chart(fig_power)

                with col5:
                    # Battery Output
                    fig_battopt = px.area(x=date_range, y=y_battopt, labels={'x': 'Date', 'y': 'Battery Charge (kWh)'}, title='BATTERY CHARGE')
                    fig_battopt.update_layout(
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        width=600,
                        height=400,
                        showlegend=False  # Hide legend
                    )
                    fig_battopt.update_layout(width=400)
                    st.plotly_chart(fig_battopt)


                # Get definitiion for weather WMO codes
                wmo_url = 'https://gist.githubusercontent.com/stellasphere/9490c195ed2b53c707087c8c2db4ec0c/raw/76b0cb0ef0bfd8a2ec988aa54e30ecd1b483495d/descriptions.json'
                wmo_description = requests.get(wmo_url).json()

                # Forecast header
                st.subheader('7 Day Energy Forecast')
                st.markdown('')  # Empty markdown line for spacing
                st.markdown('')  # Empty markdown line for spacing

                # WEATHER
                daily_forecasts = np.array(weather).reshape(7, 24)

                # Define the range of daytime hours (for example, 7 AM to 7 PM)
                start_hour = 7
                end_hour = 19

                # Find the mode for daytime hours for each day
                daily_modes = []
                for day_data in daily_forecasts:
                  daytime_data = day_data[start_hour:end_hour] # Slice for daytime hours
                  mode = np.bincount(daytime_data).argmax()
                  daily_modes.append(mode)

                weekly_forecast = {}
                for index, day in enumerate(daily_modes):
                    image = wmo_description[f'{day}']['day']['image']
                    description = wmo_description[f'{day}']['day']['description']
                    weekly_forecast[index] = []
                    weekly_forecast[index].append(image)
                    weekly_forecast[index].append(description)

                # Split the columns for 7 images for 7 days of week
                mon, tue, wed, thu, fri, sat, sun = st.columns(7)

                mon.text('⌁ DAY 01')
                mon.image(weekly_forecast[0][0])
                mon.text(weekly_forecast[0][1])

                tue.text('⌁ DAY 02')
                tue.image(weekly_forecast[1][0])
                tue.text(weekly_forecast[1][1])

                wed.text('⌁ DAY 03')
                wed.image(weekly_forecast[2][0])
                wed.text(weekly_forecast[2][1])

                thu.text('⌁ DAY 04')
                thu.image(weekly_forecast[3][0])
                thu.text(weekly_forecast[3][1])

                fri.text('⌁ DAY 05')
                fri.image(weekly_forecast[4][0])
                fri.text(weekly_forecast[4][1])

                sat.text('⌁ DAY 06')
                sat.image(weekly_forecast[5][0])
                sat.text(weekly_forecast[5][1])

                sun.text('⌁ DAY 07')
                sun.image(weekly_forecast[6][0])
                sun.text(weekly_forecast[6][1])


                # FOOTER
                # Tracker cards
                st.divider()
                st.subheader('Model Performance')
                st.markdown('')  # Empty markdown line for spacing

                foot1, foot2, foot3 = st.columns(3)
                foot1.metric("Average User Annual Savings", "£230 💷")
                foot2.metric("Mean Average Error", "£0.64 📈")
                foot3.metric("R Squared:", "0.92 ✅")
                st.markdown('')  # Empty markdown line for spacing
                st.balloons()
                st.markdown("---")


                # lottie_url = 'https://assets5.lottiefiles.com/packages/lf20_V9t630.json'
                # st_lottie(lottie_url, key="user", height=100)


page_dashboard()
