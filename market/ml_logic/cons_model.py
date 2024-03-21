import pandas as pd
import numpy as np
from datetime import datetime
import os

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


def cons_model(X ='A', date=datetime(2024,3,19,18,00,0)):
    '''
    A model which takes in X, ACORN value for the house and d, the dat which is being investigated.
    Returns the predicted and actual energy data in lists
    '''
    # TODO: add an extra input as number of days
    # Data import
    X = X.upper()
    path = '/home/adam/code/hramzan01/market/raw_data/ACORN_A_processed.csv'
    cwd = os.getcwd()
    path = cwd + f'/raw_data/ACORN_{X}_processed.csv'
    data = pd.read_csv(path)

    # Preprocess date to reset minutes and sceonds to 0
    # TODO change datetime from 2013 if needed
    d = datetime(2013, date.month, date.day, date.hour, 0, 0)

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
    # remove the first row and take only hourly samples
    data = data.iloc[1:,:]
    data = data.iloc[::2,:]
    data = data.reset_index().drop(columns=['index'])
    print('Energy consumption data processed')

    # Define train and test data
    date_index = data.index[data['ds'] == d][0]
    X_train = data.iloc[:date_index]

    # Create Prophet model
    m = Prophet()
    m.fit(X_train)
    print('Energy consumption model created')

    # Forecast one week data
    future = m.make_future_dataframe(periods=7*24, freq='h')
    forecast = m.predict(future)
    print('Energy consumption forecast complete')

    # Create return strings of forecasted and real energy consumption
    prediction = forecast[['ds', 'yhat']].iloc[date_index:]
    actual = data.iloc[date_index:date_index + 168]
    # resccale to current time period
    time_diff = date - d
    prediction['ds'] += time_diff
    prediction.set_index('ds', inplace = True)
    actual['ds'] += time_diff
    actual.set_index('ds', inplace = True)

    # return forecasted and real energy consumption
    return actual, prediction

if __name__ == '__main__':
    actual, prediction = cons_model('A', date=datetime(2018,5,6,18,0,0))
    print(actual)
    print(prediction)
    #d=datetime(2014,5,6,18,30,5)
    #print(d.date())
