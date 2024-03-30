

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dateutil import parser


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
        background-color: #FF7C52;
    }
    </style>
    """
    """
    <style>
    [data-testid="stApp"] {
        background-color: #FF7C52;
    }
    </style>
    """
    """
    <style>
    [data-testid="column"]{
        background-color: #EEF0F4;
        border-radius: 5px;
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


# HOME: MARKET logo
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing

image = Image.open('app/assets/logo.png')
st.image(image, use_column_width=True)

st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing
st.markdown('')  # Empty markdown line for spacing


# ABOUT: brief description
st.header('What is Market', divider='grey')
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

# INPUT: user variable (or create your custom profile)
st.header('Form', divider='grey')


with st.form(key='params_for_api'):
    Postcode = st.text_input("Postcode", "")
    House_price = st.number_input("House price", step=10000)
    Income = st.number_input("Income", step=1)
    Battery_Size = st.number_input("Battery Size", step=1)
    Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    House_type = ["Bungalow","Terraced house", "Detached house", "Flat or maisonette", "Semi-detached house"]
    selected_date = st.date_input('Select a date', datetime.today())

    selected_option = st.selectbox("Select an option", House_type)

    ########Test_API_predict
    params = {
        #'date': f'{selected_date} 00:00:00',
        'battery_size': Battery_Size,
        'battery_charge': Battery_Charge,
        'solar_size': Income
    }

    api_url = 'http://127.0.0.1:8000/predict'
    # api_url = 'http://127.0.0.1:8000/predict?date=2024-01-03%2018:30:05&battery_size=5&battery_charge=1'

    complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])

    st.write(f'params passed to API are {Battery_Size}, {Battery_Charge} and {Income}') #{selected_date}

    st.write(f'complete url is {complete_url}')

    # if st.form_submit_button('Submit'):
    #     response = requests.get(api_url, params=params)
    #     prediction = response.json()
    #     st.write(prediction)

    if st.form_submit_button('Submit'):
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

        # saleprice = pd.DataFrame(data['prediction_data']['SalePrice_p/kwh'])
        # buyprice = pd.DataFrame(data['prediction_data']['PurchasePrice_p/kwh'])
        # power_gen = pd.DataFrame(data['prediction_data']['Generation_kwh'])
        # power_cons = pd.DataFrame(data['prediction_data']['Consumption_kwh'])

        # st.write(res_opt_batt.head())
        # st.write(buyprice.df)
        # st.write(power_gen.df)
        # st.write(power_cons.df)

        x_sale, y_sale = zip(*saleprice.items()) # unpack a list of pairs into two tuples
        x_buy, y_buy = zip(*buyprice.items())
        x_gen, y_gen = zip(*power_gen.items())
        x_cons, y_cons = zip(*power_cons.items())

        x_battopt, y_battopt = zip(*res_opt_batt.items())
        x_bpopt, y_bpopt = zip(*res_opt_buyprice.items())
        x_spopt, y_spopt = zip(*res_opt_sellprice.items())
        x_basep, y_basep = zip(*res_opt_baseprice.items())

        dates = pd.to_datetime(x_buy)

        fig = plt.figure();
        plt.plot(dates, y_sale,label = 'sell_price');
        plt.plot(dates, y_buy, label = 'buy price');
        plt.legend()
        start_date = (x_buy[0]) #.floor('D')
        start_datetimeobj = parser.isoparse(start_date)
        start_datetime = start_datetimeobj.strftime('%Y-%m-%d %H:%M:%S')
        st.code(start_datetimeobj)
        st.code(start_datetime)
        end_date = x_buy[-1] #.floor('D')
        end_datetimeobj = parser.isoparse(end_date)
        st.code(end_datetimeobj)
        plt.xticks(pd.date_range(start=start_datetimeobj, end=end_datetimeobj, freq='2D'))
        st.pyplot(fig)

        fig_power = plt.figure();
        plt.plot(dates, y_gen,label = 'Power_gen');
        plt.plot(dates, y_cons, label = 'Power_cons');
        plt.legend()
        plt.xticks(pd.date_range(start=start_datetimeobj, end=end_datetimeobj, freq='2D'))
        st.pyplot(fig_power)

        fig_battopt = plt.figure();
        plt.plot(x_battopt, y_battopt,label = 'Battery_Opt');
        # plt.plot(x_cons, y_cons, label = 'Power_cons');
        plt.legend()
        st.pyplot(fig_battopt)

        fig_priceopt = plt.figure();
        plt.plot(x_bpopt, y_bpopt,label = 'Buy_Price_Opt');
        plt.plot(x_spopt, y_spopt, label = 'Sell_Price_Opt');
        plt.plot(x_basep, y_basep, label = 'Base_Price');
        plt.legend()
        st.pyplot(fig_priceopt)



st.button("Submit")

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

# Define CSS style
css = """
    <style>
        .container {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
        }
        column {
            background-color: #808080;
        }
    </style>
"""

# Get definitiion for weather WMO codes
from IPython.display import Image
import requests
wmo_url = 'https://gist.githubusercontent.com/stellasphere/9490c195ed2b53c707087c8c2db4ec0c/raw/76b0cb0ef0bfd8a2ec988aa54e30ecd1b483495d/descriptions.json'
wmo_description = requests.get(wmo_url).json()
image = wmo_description['2']['day']['image']
total = round(data['kwh'].sum())
col1, col2, col3 = st.columns(3)
col1.image(image)
col2.metric("Avg Daily Power",total, 'kWh')
col3.metric("Weather",wmo_description['2']['day']['description'])
