import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(page_title="Market", layout="wide", initial_sidebar_state="collapsed")


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
</head>
<body>
    <div id="progress"></div>
</body>
</html>
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
Postcode = st.text_input("Postcode", "")
House_price = st.text_input("House price", "")
Income = st.text_input("Income", "")
House_type = ["Bungalow","Terraced house", "Detached house", "Flat or maisonette", "Semi-detached house"]

selected_option = st.selectbox("Select an option", House_type)

st.button("Submit")
