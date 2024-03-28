import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import plotly.express as px

st.set_page_config(page_title="Market", initial_sidebar_state="collapsed")

with st.form(key='params_for_api'):
    # Test_API_predict
    params = {
        #'date': f'{selected_date} 00:00:00',
        'battery_size': 5,
        'battery_charge': 3
    }

    api_url = 'http://127.0.0.1:8000/predict'
    complete_url = api_url + '?' + '&'.join([f"{key}={value}" for key, value in params.items()])
    
    # Generate Dashboard when submit is triggered
    if st.form_submit_button('Predict'):
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

        '''VISUALS'''
        # Buy vs Sell Price
        fig = px.line(x=dates, y=y_sale, labels={'x': 'Date', 'y': 'Price'}, title='Buy vs Sell Price')
        fig.add_scatter(x=dates, y=y_buy, mode='lines', name='Buy Price')
        st.plotly_chart(fig)

        # Power gen vs power con
        fig_power = px.line(x=dates, y=[y_gen, y_cons], labels={'x': 'Date', 'y': 'Power'}, title='Power Generation vs Consumption')
        st.plotly_chart(fig_power)

        # Battery Output
        fig_battopt = px.line(x=x_battopt, y=y_battopt, labels={'x': 'Date', 'y': 'Battery Output'}, title='Battery Output')
        st.plotly_chart(fig_battopt)

        # Buy price vs sell price vs base price
        fig_priceopt = px.line(x=x_bpopt, y=y_bpopt, labels={'x': 'Date', 'y': 'Price'}, title='Buy vs Sell vs Base Price')
        fig_priceopt.add_scatter(x=x_spopt, y=y_spopt, mode='lines', name='Sell Price')
        fig_priceopt.add_scatter(x=x_basep, y=y_basep, mode='lines', name='Base Price')
        st.plotly_chart(fig_priceopt)
