import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px
from PIL import Image
import os


st.set_page_config(page_title="Market", initial_sidebar_state="collapsed", layout='wide')

# define temporary variables from profile page
battery_size = 5
battery_charge = 3
postcode = 'E1 5DY'

# Return Lat & Lon from postcode
base_url = 'https://api.postcodes.io/postcodes'
response = requests.get(f'{base_url}/{postcode}').json()

lat = response['result']['latitude']
lon = response['result']['longitude']


with st.form(key='params_for_api'):
    # Test_API_predict
    params = {
        #'date': f'{selected_date} 00:00:00',
        'battery_size': 5,
        'battery_charge': 3
    }

    api_url = 'http://127.0.0.1:8000/predict'
    complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])
    # complete_url
    
    # Generate Dashboard when submit is triggered
    if st.form_submit_button('Predict', use_container_width=True):
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
                    size=16,
                    color='orange'
                ),
                text=['London']
            ))

            fig.update_layout(
                autosize=True,
                margin=dict(l=5, r=5, t=0, b=20),
                hovermode='closest',
                mapbox=dict(
                    style='carto-positron',
                    bearing=0,
                    center=dict(
                        lat=lat,
                        lon=lon
                    ),
                    pitch=0,
                    zoom=17
                ),
                width=1280,
                height=400
            )
            return fig
        
        st.write(london_map(lat, lon))
        
        # Split the remaining space into three columns
        col1, col2, col3 = st.columns(3)
        
        # Display images
        image1 = Image.open('app/assets/logo.png')
        image2 = Image.open('app/assets/logo.png')
        image3 = Image.open('app/assets/logo.png')


        
        # First column: Buy vs Sell Price
        with col1:
            # Buy vs Sell Price
            st.image(image1, use_column_width=True)
            fig = px.line(x=dates, y=y_sale, labels={'x': 'Date', 'y': 'Price'}, title='Buy vs Sell Price')
            fig.add_scatter(x=dates, y=y_buy, mode='lines', name='Buy Price')
            st.plotly_chart(fig)

        with col2:
            # Power gen vs power con
            st.image(image2, use_column_width=True)
            fig_power = px.line(x=dates, y=[y_gen, y_cons], labels={'x': 'Date', 'y': 'Power'}, title='Power Generation vs Consumption')
            st.plotly_chart(fig_power)

        with col3:
            # Battery Output
            st.image(image3, use_column_width=True)
            fig_battopt = px.area(x=x_battopt, y=y_battopt, labels={'x': 'Date', 'y': 'Battery Output'}, title='Battery Output')
            fig_battopt.update_layout(width=400)
            st.plotly_chart(fig_battopt)
            
 
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
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            paper_bgcolor='rgba(0, 0, 0, 0)'   # Transparent background
        # Transparent background
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
