import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

import matplotlib.pylab as plt

# import datetime
import requests

st.set_page_config(page_title="Market", initial_sidebar_state="collapsed")


st.markdown(

    """
    <style>
    [data-testid="stHeader"] {
        background-color: #FFA500;
    }
    </style>
    """
    """
    <style>
    [data-testid="stApp"] {
        background-color: #FFA500;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Scrolling battery
st.markdown(
    """
    <div id="progress" style="background-color: #ddd; height: 20px; width: 10px;"></div>
    """,
    unsafe_allow_html=True
)

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scroll Battery</title>
    <style>
        #progress {
            position: fixed;
            top: 50%;
            right: 10px; /* Adjust as needed */
            transform: translateY(-50%);
            background-color: #ddd;
            height: 40px;
            width: 20px;
        }
    </style>
    <style>
        #battery {
            position: fixed;
            top: 50%;
            right: 11px; /* Adjust as needed */
            transform: translateY(-50%);
            background-color: #5BC236;
            height: 35px;
            width: 18px;
        }
    </style>
</head>
<body>
    <div id="progress"></div>
    <div id="battery"></div>
</body>
</html>
<script>
    window.addEventListener('scroll', function() {
        var battery = document.getElementById("battery");
        var scrollHeight = document.documentElement.scrollHeight;
        var clientHeight = document.documentElement.clientHeight;
        var scrollTop = document.documentElement.scrollTop;
        var percentageScrolled = (scrollTop / (scrollHeight - clientHeight)) * 100;
        var newHeight = percentageScrolled + "%";
        battery.style.height = newHeight;
    });
</script>
"""

st.markdown(html_code, unsafe_allow_html=True)


st.markdown(
        """
        <h1 style='text-align: center; font-size: 60px; color: white;'>Market</h1>
        <h3 style='text-align: center;'>Social Solar Trading</h3>
        """,
        unsafe_allow_html=True
    )


st.header('About', divider='grey')

st.subheader("""Empowerment: 💪""")
st.write("""
Empower users to participate in renewable energy ownership, promoting sustainability and community engagement.
""")
st.subheader("""Optimization: ⏱️""")
st.write("""
Optimize users' trading decisions by leveraging weather forecasts and solar energy output predictions to maximize profitability.
""")
st.subheader("""Accessibility:🤙""")
st.write("""
         Make renewable energy trading accessible to all, regardless of technical expertise and type of accomodation.
""")

st.header('Form', divider='grey')

with st.form(key='params_for_api'):
    Postcode = st.text_input("Postcode", "")
    House_price = st.number_input("House price", step=10000)
    Income = st.number_input("Income", step=10000)
    Battery_Size = st.number_input("Battery Size", step=1)
    Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
    House_type = ["Bungalow","Terraced house", "Detached house", "Flat or maisonette", "Semi-detached house"]
    selected_date = st.date_input('Select a date', datetime.today())

    selected_option = st.selectbox("Select an option", House_type)

    ########Test_API_predict
    params = {
        #'date': f'{selected_date} 00:00:00',
        'battery_size': Battery_Size,
        'battery_charge': Battery_Charge
    }

    api_url = 'http://127.0.0.1:8000/predict'
    # api_url = 'http://127.0.0.1:8000/predict?date=2024-01-03%2018:30:05&battery_size=5&battery_charge=1'

    complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])

    st.write(f'params passed to API are {Battery_Size}, and {Battery_Charge}') #{selected_date}

    st.write(f'complete url is {complete_url}')

    # if st.form_submit_button('Submit'):
    #     response = requests.get(api_url, params=params)
    #     prediction = response.json()
    #     st.write(prediction)

    if st.form_submit_button('Submit'):
        response = requests.get(api_url, params=params)
        prediction = response.json()
        data = prediction
        saleprice = data['SalePrice_p/kwh']
        buyprice = data['PurchasePrice_p/kwh']
        power_gen = data['Generation_kwh']
        power_cons = data['Consumption_kwh']
        x_sale, y_sale = zip(*saleprice.items()) # unpack a list of pairs into two tuples
        x_buy, y_buy = zip(*buyprice.items())
        x_gen, y_gen = zip(*power_gen.items())
        x_cons, y_cons = zip(*power_cons.items())

        fig = plt.figure();
        plt.plot(x_sale, y_sale,label = 'sell_price');
        plt.plot(x_buy, y_buy, label = 'buy price');
        plt.legend()
        # plt.show()
        # st.write(prediction)
        fig_power = plt.figure();
        plt.plot(x_gen, y_gen,label = 'Power_gen');
        plt.plot(x_cons, y_cons, label = 'Power_cons');
        plt.legend()
        st.pyplot(fig)
        st.pyplot(fig_power)



########


st.header('Your Output', divider='grey')
day = st.slider("Select Days", 1,7,1)
# data = pd.read_csv("/home/freddieoxland/code/hramzan01/market/notebooks/data/final_prediction.csv")

# Create a Plotly figure
fig = go.Figure()

# Add a line trace to the figure
# fig.add_trace(go.Scatter(y=data['kwh'][0:(day*24)], mode='lines', name='Line Chart'))

# Set title and axes labels
fig.update_layout(title='Line Chart', xaxis_title='X-axis', yaxis_title='Y-axis')

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
col2.metric("power",total, 'kWh')
col3.metric("Weather",wmo_description['2']['day']['description'])
