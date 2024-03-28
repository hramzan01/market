import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
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

Postcode = st.text_input("Postcode", "")
House_price = st.number_input("House price", step=10000)
Income = st.number_input("Income", step=10000)
Battery_Size = st.number_input("Battery Size", step=1)
Battery_Charge =  st.number_input("Battery Charge", step=1, min_value=0, max_value=100)
House_type = ["Bungalow","Terraced house", "Detached house", "Flat or maisonette", "Semi-detached house"]
selected_date = st.date_input('Select a date', datetime.today())

selected_option = st.selectbox("Select an option", House_type)

st.button("Submit")

# Output: User visuals (this is user dashboard)
st.header('Your Output', divider='grey')

day = st.slider("Select Days", 1,7,1)
root = os.getcwd()
data = pd.read_csv(os.path.join(root, 'data/processed_data/final_prediction.csv'))

# Create a Plotly figure
fig = go.Figure()

# Add a line trace to the figure
fig.add_trace(go.Scatter(y=data['kwh'][0:(day*24)], mode='lines',fill='tozeroy', name='Line Chart'))

# Set title and axes labels
fig.update_layout(title='Line Chart', xaxis_title='X-axis', yaxis_title='Y-axis')
fig.update_layout(
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
    paper_bgcolor='rgba(0, 0, 0, 0)'   # Transparent background
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
