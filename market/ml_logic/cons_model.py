import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


def cons_model(X, d):
    '''
    A model which takes in X, ACORN value for the house and d, the dat which is being investigated.
    Returns the predicted and actual energy data in lists
    '''

    # Data import
    X = X.upper()
    path = '/home/adam/code/hramzan01/market/raw_data/ACORN_A_processed.csv'
    cwd = os.getcwd()
    path = cwd + f'/raw_data/ACORN_{X}_processed.csv'
    data = pd.read_csv(path)

    # Preprocess date to reset minutes and sceonds to 0
    # TODO change datetime from 2013 if needed
    d = datetime(2013, d.month, d.day, d.hour, 0, 0)

    # Processing input data for profit
    # TODO: update data processing so data comes into the model in the right format
    # remove excess columns
    data = data.drop(columns=['Unnamed: 0'])
    # put date into correct format
    data['date'] = data['date'].apply(lambda x: x[:19])
    date_format = '%Y-%m-%d %H:%M:%S'
    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, date_format))
    # renaming columns for Prophet
    data = data.rename(columns = {'date':'ds','Average energy(kWh/hh)':'y'})
    data = data.iloc[1:,:]
    data = data.iloc[::2,:]
    data = data.reset_index().drop(columns=['index'])
    print(data.columns)

    # Define train and test data
    date_index = data.index[data['ds'] == d][0]
    print(date_index)
    X_train = data.iloc[:date_index]

    # Create Prophet model
    m = Prophet()
    m.fit(X_train)

    # Forecast one week data
    future = m.make_future_dataframe(periods=7*24, freq='h')
    forecast = m.predict(future)

    # Create return strings of forecasted and real energy consumption
    prediction = forecast[['ds', 'yhat']].iloc[date_index:]
    actual = data.iloc[date_index:date_index + 168]

    # return forecasted and real energy consumption
    return actual,prediction

actual, prediction = cons_model('A', d=datetime(2014,5,6,18,30,5))
print(actual)
print(prediction)
