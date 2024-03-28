'''
Energy_price_model
Predicts the energy price from a given date in the UK
'''

# imports
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
from prophet.serialize import model_to_json, model_from_json

# Stop Prophet outputting lots of information
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

"""
Downloading, creating a model, and foreacsting London energy prices
"""

def create_folder_if_not_exists(folder_path):
    """
    Creating folder for London wholesale energy prices
    """
    # if folder exists, do nothing. If it doesn't exist, create.
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


def create_train_test_set(file, d, previous_days=6*30, forecast_days = 7):
    """
    Returns original df, and train set
    Also resamples the data from 30 min to hourly intervals
    """
    # define train test split ratio
    split_ratio = 1 - forecast_days/(previous_days + forecast_days)

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
    end_date_dt = d + timedelta(days=forecast_days)
    start_date = str(start_date_dt)
    end_date = str(end_date_dt)
    if start_date_dt - df_price['ds'][0] > timedelta(0):
        print(f'Specified date {d} is in the correct range')
    else:
        print(f'Specified date {d} is out of the range')
        print('Please enter a different date')
        print(end_date)
        return

    # check if the full testing set exists
    if df_price['ds'].iloc[-1] <= end_date_dt:
        train = df_price[(df_price['ds']>start_date) & (df_price['ds']<= d)]
        test = 'Date not applicable for test set'
        return train, test

    else:
        # Create train and test set
        df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]
        check_start_date=df_price['ds'][df_price.index[0]]
        check_end_date=df_price['ds'][df_price.index[-1]]
        split_index = round(df_price.shape[0]*split_ratio) - 1
        train = df_price.iloc[:split_index]
        test = df_price.iloc[split_index:-1]
        #print(test)
        # sample test set so it takes every other entry - hourly results and return df
        train = train.iloc[1:]
        train = train.iloc[::2,:]
        test = test.iloc[::2,:]
        test.set_index('ds', inplace = True)

        print('Cost data processed')
        return train, test


def ml_model(train, forecast_days=7, seasonality_mode = 'multiplicative', year_seasonality_mode=4, freq='h'):
    """
    Returns trained prophet model and forecasting for energy prices
    """
    # Set up prophet model
    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=year_seasonality_mode, interval_width=0.95)
    model.fit(train)

    # Make forecast
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq=freq)
    forecast = model.predict(future)
    forecast_y_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Forecasting dates
    start_date=train['ds'][train.index[0]]
    end_date=train['ds'][train.index[-1]]

    # Return prediction
    y_pred = forecast_y_df.iloc[-forecast_days*24 :]
    y_pred.set_index('ds', inplace = True)
    print('Prediction Successful')
    return model, y_pred


def energy_model_run(date, forecast_days = 7):
    '''
    A function to run the full model and return test and forecasted data
    '''
    # Set ups for files
    url = 'https://files.energy-stats.uk/csv_output/'
    dir = os.path.join(os.getcwd(), 'raw_data')
    csv_name = 'csv_agileoutgoing_C_London.csv'
    file = os.path.join(url, csv_name)
    save_path = os.path.join(dir, csv_name)

    # download the latest file
    download_file(file, save_path)

    # Run the model
    train, test = create_train_test_set(file, d=date, previous_days=36*30, forecast_days = forecast_days)
    # Line removed for model checking AEOXLEY
    #train, test = create_train_test_set(file, d=date, previous_days=6*30, forecast_days = forecast_days)
    model, forecast_y_df = ml_model(train, forecast_days=forecast_days, seasonality_mode = 'multiplicative', year_seasonality_mode=4, freq='h')
    print('Model finished')
    #return test, forecast_y_df[['ds', 'yhat']]
    return test, forecast_y_df[['yhat']]


def price_save_model(date, forecast_days = 7):
    '''
    A function to preprocess the data and save the final model for future use
    '''
    # Set ups for files
    url = 'https://files.energy-stats.uk/csv_output/'
    dir = os.path.join(os.getcwd(), 'raw_data')
    csv_name = 'csv_agileoutgoing_C_London.csv'
    file = os.path.join(url, csv_name)
    save_path = os.path.join(dir, csv_name)

    # download the latest file
    download_file(file, save_path)

    # preprocess the data
    train, test = create_train_test_set(file, d=date, previous_days=6*30, forecast_days = forecast_days)

    # train the model
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4, interval_width=0.95)
    model.fit(train)

    with open('market/models/price_model.json', 'w') as fout:
        fout.write(model_to_json(model))  # Save model
    print('Cost model saved')
    return


def price_load_model(date, forecast_days = 7):
    '''
    A function to laod a saved model and run a one week prediction
    '''
    # load model
    with open('market/models/price_model.json', 'r') as fin:
        model = model_from_json(fin.read())  # Load model

    # Make forecast
    horizon = 24*forecast_days
    future = model.make_future_dataframe(periods = horizon, freq='h')
    forecast = model.predict(future)
    forecast_y_df = forecast[['ds', 'yhat']]

    # Return prediction
    y_pred = forecast_y_df.iloc[-horizon :]
    y_pred.set_index('ds', inplace = True)
    print('Cost forecasted')
    return y_pred


if __name__ == '__main__':
    #year = 2024
    #month = 3
    #day = 10
    #date = datetime(year,month,day)
    date = datetime.now()
    date = date.replace(minute = 0, second = 0, microsecond = 0)

    #test, forecast_y_df = energy_model_run(date, forecast_days = 7)
    price_save_model(date, forecast_days = 7)
    y_pred = price_load_model(date, forecast_days = 7)
