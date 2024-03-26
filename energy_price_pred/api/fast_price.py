import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from energy_price_pred.energypricepred import *

from market.ml_logic.optimiser_model_variable_inputs import *

battery_size = 5 # total size
battery_charge = 1 # initial charge amount
date = '2024-01-03 18:30:05' #datetime(2024,1,3,18,30,5)

app = FastAPI()

@app.get("/predict")
def predict(date, battery_size, battery_charge):
    """predicting Buy/Sell price as first pass"""
    # df_price, train, test = process_df(file, previous_days=12*30)
    # model, forecast_y_df, df_cv, df_p = ml_model(train)
    # pred_y_df_sell, hourly_data_sell, hourly_data_buy, date1, date2, forecast_days = pred(df_price, model, forecast_start_date='2024-03-23', forecast_days=7)

    date=pd.Timestamp(date)
    return run_full_model_api(date, int(battery_size), int(battery_charge)) #{'eval_full_model_res': evaluate_full_model(date, battery_size, battery_charge)}  #

@app.get("/")
def root():
    return {'greeting': 'Hello there'}
