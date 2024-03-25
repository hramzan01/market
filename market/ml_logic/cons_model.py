import pandas as pd
import numpy as np
from datetime import datetime
import os

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.serialize import model_to_json, model_from_json


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

def cons_save_model(X ='A', date=datetime(2024,3,19,18,00,0)):
    '''
    Use the consumption data to run the model and save the model
    '''
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

    with open('market/models/consumption_model.json', 'w') as fout:
        fout.write(model_to_json(m))  # Save model
    return


def cons_load_model(date, forecasted_days = 7):
    '''
    Load the model and run the forcast
    '''
    # Load the model
    with open('market/models/consumption_model.json', 'r') as fin:
        m = model_from_json(fin.read())  # Load model

    # reset the date to a matching date in the required time period
    d = datetime(2013, date.month, date.day, date.hour, 0, 0)

    # Forecast one week data
    horizon = forecasted_days*24
    future = m.make_future_dataframe(periods=horizon, freq='h')
    forecast = m.predict(future)
    print('Energy consumption forecasted')

    # Create return strings of forecasted and real energy consumption
    prediction = forecast[['ds', 'yhat']].iloc[-horizon :]
    # resccale to current time period
    time_diff = date - d
    prediction['ds'] += time_diff
    prediction.set_index('ds', inplace = True)
    return prediction


if __name__ == '__main__':
    d = date=datetime(2018,5,6,18,0,0)
    cons_save_model('A', date=d)
    prediction = cons_load_model(date=d, forecasted_days = 7)
    print(prediction)
