import pandas as pd
from fastapi import FastAPI
from energy_price_pred.energypricepred import *


app = FastAPI()

@app.get("/predict")
def predict():
    """predicting Buy/Sell price as first pass"""
    df_price, train, test = process_df(file, previous_days=12*30)
    model, forecast_y_df, df_cv, df_p = ml_model(train)
    pred_y_df_sell, hourly_data_sell, hourly_data_buy, date1, date2, forecast_days = pred(df_price, model, forecast_start_date='2024-03-23', forecast_days=7)
    return {'hourly_data_sell': hourly_data_sell}
    # (date, battery charge, size, ACORN='A')

@app.get("/")
def root():
    return {'greeting': 'Hello'}
