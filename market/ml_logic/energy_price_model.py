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
Downloading, creating a model, and foreacsting London energy prices
"""


def create_folder_if_not_exists(folder_path):
    """
    Creating folder for London wholesale energy prices
    """
    # if folder exists, do nothing. If it doesn';t exist, create.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def download_file(file, save_path):
    """
    Downloading London wholesale electricity prices files from energy-stats
    """
    response = requests.get(file)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("File downloaded successfully")
    else:
        print("Failed to download file. Status code:", response.status_code)

def create_train_test_set(file, d, previous_days=6*30, resample_rate='H'):
    """
    AEOXLEY. Returns original df, and train set
    Also resamples the data from 30 min to hourly intervals
    """
    n_test = 7 #days
    split_ratio = 1 - n_test/(previous_days + n_test)

    # convert file to pandas dataframe
    df = pd.read_csv(file)

    # restructure the data
    column_names=['date_time', 'time', 'Letter', 'City', 'Price']
    df.columns = column_names
    df_price = pd.DataFrame(df[['date_time', 'Price']])
    df_price.columns = ['ds', 'y']
    df_price['ds'] = df_price['ds'].str.slice(stop=-6)
    df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y-%m-%d %H:%M:%S')

    # check date is in required range
    start_date_dt = d - timedelta(days=previous_days)
    end_date_dt = d + timedelta(days=n_test)
    start_date = str(start_date_dt)
    end_date = str(end_date_dt)

    if start_date_dt - df_price['ds'][0] > timedelta(0) and end_date_dt - df_price['ds'][len(df_price['ds'])-1] < timedelta(0):
        print(f'Specified date {d} is in correct range')
    else:
        print(f'Specified date {d} is out of the range')
        print(end_date)
        return

    # Create train and test set
    df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]
    check_start_date=df_price['ds'][df_price.index[0]]
    check_end_date=df_price['ds'][df_price.index[-1]]
    print(f'df created including time history of electricity export prices between {check_start_date} and {check_end_date} i.e. for the last {(check_end_date - check_start_date)} days')
    split_index = round(df_price.shape[0]*split_ratio) - 1
    train = df_price.iloc[:split_index]
    test = df_price.iloc[split_index:-1]
    test = test.iloc[::2,:]
    print('Train and Test data created')
    return df_price, train, test


def process_df(file, d, previous_days=6*30, split_ratio=0.9, resample_rate='H'):
    """
    Returns original df, train and test
    Also resamples the data from 30 min to hourly intervals
    """
    df = pd.read_csv(file)
    print(str(df.columns[0]))
    column_names=['date_time', 'time', 'Letter', 'City', 'Price']
    df.columns = column_names
    df_price = pd.DataFrame(df[['date_time', 'Price']])
    df_price.columns = ['ds', 'y']
    df_price['ds'] = df_price['ds'].str.slice(stop=-6)
    df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y-%m-%d %H:%M:%S')
    # TODO add an assert that the inputted date is in the required range

    start_date = str(d - timedelta(days=previous_days))
    end_date = str(d)
    df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]
    check_start_date=df_price['ds'][df_price.index[0]]
    check_end_date=df_price['ds'][df_price.index[-1]]
    print(str(len(df_price)))
    print(f'df created including time history of electricity export prices between {check_start_date} and {check_end_date} i.e. for the last {(check_end_date - check_start_date)} days')
    split_index = round(df_price.shape[0]*split_ratio)
    train = df_price.iloc[:split_index]
    test = df_price.iloc[split_index:]
    return df_price, train, test

def ml_model(train, forecast_days=7, seasonality_mode = 'multiplicative', year_seasonality_mode=4, freq='h'):
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
    print(f'performing cross-validation using, initial: {initial}, horizon:{horizon}, and period:{period}')
    df_cv = cross_validation(model = model, initial=f'{initial} days', horizon=f'{horizon} days', period=f'{period} days')
    df_p = performance_metrics(df_cv)
    return model, forecast_y_df, df_cv, df_p

def pred(df_price, model, forecast_start_date='2024-03-18', forecast_end_date='2024-03-25', freq='h'):
    # forecast_days = forceast_end_date-forecast_start_date
    date1 = datetime.strptime(forecast_end_date, "%Y-%m-%d").date()
    date2 = datetime.strptime(forecast_start_date, "%Y-%m-%d").date()
    forecast_days = int((date1 - date2).days)
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq= freq)
    forecast = model.predict(future)
    pred_y_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(forecast_days)
    return pred_y_df,date1, date2, forecast_days

def energy_model_run(date):
    # Set ups for files
    url = 'https://files.energy-stats.uk/csv_output/'
    dir = os.path.join(os.getcwd(), 'raw_data')
    csv_name = 'csv_agileoutgoing_C_London.csv'
    file = os.path.join(url, csv_name)
    save_path = os.path.join(dir, csv_name)
    forecast_start_date=f'{year}-{month}-{day}'
    # TODO update day +7 to take into account end of months
    forecast_end_date=f'{year}-{month}-{day + 7}'

    download_file(file, save_path)
    df_price, train, test = process_df(file, d=date, previous_days=6*30, split_ratio=0.9, resample_rate='H')
    model, forecast_y_df, df_cv, df_p = ml_model(train, forecast_days=7, seasonality_mode = 'multiplicative', year_seasonality_mode=4, freq='h')
    pred_y_df,date1, date2, forecast_days = pred(df_price, model, forecast_start_date=forecast_start_date, forecast_end_date=forecast_end_date, freq='h')
    return pred_y_df['yhat'][len(pred_y_df['yhat'])-168:]


if __name__ == '__main__':
    year = 2024
    month = 3
    day = 10
    date = datetime(year,month,day)


    url = 'https://files.energy-stats.uk/csv_output/'
    dir = os.path.join(os.getcwd(), 'raw_data')
    csv_name = 'csv_agileoutgoing_C_London.csv'
    file = os.path.join(url, csv_name)
    save_path = os.path.join(dir, csv_name)

    download_file(file, save_path)
    df_price, train, test = create_train_test_set(file, d=date, previous_days=6*30, resample_rate='H')
    print(train)
    print(test)
    #pred = energy_model_run(date)
    #print(pred)
