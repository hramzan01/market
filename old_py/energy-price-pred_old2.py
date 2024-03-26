import os
import pandas as pd
import requests

import numpy as np

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from datetime import timedelta
from datetime import datetime

"""
Downloading London wholesale electricity prices files from energy-stats
"""

url = 'https://files.energy-stats.uk/csv_output/'

dir = os.path.join(os.getcwd(), 'raw_data')

csv_name = 'csv_agileoutgoing_C_London.csv'

file = os.path.join(url, csv_name)
save_path = os.path.join(dir, csv_name)



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


def process_df(file, days=6*30, split_ratio=0.9):
    """Returns original df, train and test"""
    # download_file(file, save_path)
    df = pd.read_csv(file)
    print(str(df.columns[0]))
    column_names=['date_time', 'time', 'Letter', 'City', 'Price']
    df.columns = column_names
    df_price = pd.DataFrame(df[['date_time', 'Price']])
    df_price.columns = ['ds', 'y']
    df_price['ds'] = df_price['ds'].str.slice(stop=-6)
    df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y-%m-%d %H:%M:%S')
    start_date = str(df_price['ds'][df.index[-1]] - timedelta(days=days))
    end_date = str(df_price['ds'][df.index[-1]])
    df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]
    check_start_date=df_price['ds'][df_price.index[0]]
    check_end_date=df_price['ds'][df_price.index[-1]]
    print(str(len(df_price)))
    print(f'df created including time history of electricity export prices between {check_start_date} and {check_end_date} i.e. for the last {(check_end_date - check_start_date)} days')
    split_indices = round(df_price.shape[0]*split_ratio)
    train = df_price.iloc[:split_ratio]
    test = df_price.iloc[split_ratio:]
    return df_price, train, test

def ml_model(train, forecast_days=14, seasonality_mode = 'multiplicative', year_seasonality_mode=4):
    """Returns trained prophet model and forecasting for """
    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=year_seasonality_mode, interval_width=0.95)
    model.fit(train)
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq='30min')
    forecast = model.predict(future)
    forecast_y_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    df_cv = cross_validation(model = model, initial='180 days', horizon='30 days', period='3 days')
    df_p = performance_metrics(df_cv)
    return model, forecast_y_df, df_cv, df_p


# Example usage:


download_file(file, save_path)

process_df(file)


# print(df(file)[0])
