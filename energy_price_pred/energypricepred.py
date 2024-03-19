import os
import pandas as pd
import requests
import time

import numpy as np

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse

"""
Downloading London wholesale electricity prices files from energy-stats
"""

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def download_file(file, save_path):
    response = requests.get(file)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("File downloaded successfully")
    else:
        print("Failed to download file. Status code:", response.status_code)


def process_df(file, previous_days=6*30, split_ratio=0.9, resample_rate='H'):
    """Returns original df, train and test
    Also resamples the data from 30 min to hourly intervals"""
    # download_file(file, save_path)
    df = pd.read_csv(file)
    print(str(df.columns[0]))
    column_names=['date_time', 'time', 'Letter', 'City', 'Price']
    df.columns = column_names
    df_price = pd.DataFrame(df[['date_time', 'Price']])
    df_price.columns = ['ds', 'y']
    df_price['ds'] = df_price['ds'].str.slice(stop=-6)
    df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y-%m-%d %H:%M:%S')
    start_date = str(df_price['ds'][df.index[-1]] - timedelta(days=previous_days))
    end_date = str(df_price['ds'][df.index[-1]])
    df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]

    # df_price = df_price.resample('H').mean()

    check_start_date=df_price['ds'][df_price.index[0]]
    check_end_date=df_price['ds'][df_price.index[-1]]
    print(str(len(df_price)))
    print(f'df created including time history of electricity export prices between {check_start_date} and {check_end_date} i.e. for the last {(check_end_date - check_start_date)} days')
    split_index = round(df_price.shape[0]*split_ratio)
    train = df_price.iloc[:split_index]
    test = df_price.iloc[split_index:]
    return df_price, train, test

def ml_model(train, forecast_days=7, seasonality_mode = 'multiplicative', year_seasonality_mode=4, freq='30min'):
    """Returns trained prophet model and forecasting for """
    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=year_seasonality_mode, interval_width=0.95)
    model.fit(train)
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq=freq)
    forecast = model.predict(future)
    forecast_y_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    start_date=train['ds'][train.index[0]]
    end_date=train['ds'][train.index[-1]]

    initial=round(((end_date - start_date).days)/2)
    horizon=round(initial/5)
    period = round(horizon/2)

    df_cv = cross_validation(model = model, initial=f'{initial} days', horizon=f'{horizon} days', period=f'{period} days')
    df_p = performance_metrics(df_cv)
    return model, forecast_y_df, df_cv, df_p

def pred(df_price, model, forecast_start_date='2024-03-18', forceast_end_date='2024-03-25'):
    # forecast_days = forceast_end_date-forecast_start_date
    date1 = datetime.strptime(forceast_end_date, "%Y-%m-%d").date()
    date2 = datetime.strptime(forecast_start_date, "%Y-%m-%d").date()
    forecast_days = int((date1 - date2).days)
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq='30min')
    forecast = model.predict(future)
    pred_y_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(forecast_days)
    return pred_y_df,date1, date2, forecast_days



url = 'https://files.energy-stats.uk/csv_output/'
dir = os.path.join(os.getcwd(), 'raw_data')
csv_name = 'csv_agileoutgoing_C_London.csv'
file = os.path.join(url, csv_name)
save_path = os.path.join(dir, csv_name)

# create_folder_if_not_exists(dir)
# download_file(file, save_path)
# df_price, train, test = process_df(file)
# model, forecast_y_df, df_cv, df_p = ml_model(train)
# pred(df_price, model)
