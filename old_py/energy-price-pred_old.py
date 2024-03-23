import os
import pandas as pd
import requests

import numpy as np

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from datetime import timedelta



"""
Downloading files from energy-stats
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

# Example usage:


download_file(file, save_path)

def df(file, days=6*30):
    download_file(file, save_path)
    df = pd.read_csv(file)
    column_names=['date_time', 'time', 'Letter', 'City', 'Price']
    df.columns = column_names
    df_price = pd.DataFrame(df[['date_time', 'Price']])
    df_price.columns = ['ds', 'y']
    df_price['ds'] = df_price['ds'].str.slice(stop=-6)
    df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y-%m-%d %H:%M:%S')
    start_date = str(df['ds'][df.index[-1]] - timedelta(days=days))
    end_date = str(df['ds'][df.index[-1]])
    df_price = df_price[(df_price['ds']>start_date) & (df_price['ds']<= end_date)]
    return df_price

df_price = df(file)

days=6*30   # Looking at past 6 months

start_date = df_price['ds'][df_price.index[-1]] - timedelta(days=6*30)
end_date = df_price['ds'][df_price.index[-1]]

df_test = df[(df['ds']>start_date) & (df['ds']<= end_date)]

date = train['ds'][train.index[-1]]- timedelta(days=days)
date_end = date + timedelta(days=days+30)

split_ratio = round(df.shape[0]*0.9)
