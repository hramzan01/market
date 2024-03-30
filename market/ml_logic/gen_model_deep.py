'''
gen_model_efficient
Creates and runs a RNN model to predict PV energy generation
'''

# Import libraries
import pandas as pd
import os
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime
from datetime import timedelta
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, layers


warnings.simplefilter('ignore')

n_input = 24
batch_size = 64

def load_raw_data():
    '''
    function retrievs UKPV hourly datset and merges to combine time series with location and meta data
    '''
    # Load in Data
    df_spec = pd.read_csv('data/raw_data/metadata.csv')
    df_energy_ldn = pd.read_csv('/home/adam/code/hramzan01/market/raw_data/hourly_generation_ldn.csv')

    # rename of change type
    df_energy_ldn.rename(columns={'datetime': 'timestamp'}, inplace=True)
    df_energy_ldn['timestamp'] = pd.to_datetime(df_energy_ldn['timestamp'], utc=True)

    # Merge spec with energy
    df_merged = pd.merge(df_energy_ldn,df_spec,how='left',on='ss_id')
    df_merged['formatted_timestamp'] = df_merged['timestamp'].dt.strftime('%Y-%m-%dT%H:%M')

    print('--raw data loaded--')
    print(df_merged.head(3))
    return df_merged

def append_weather_params():
    '''
    function reads historical weather forecast for london area and appends to existing DF
    '''
    # Define list of properties to iterate through as chunks
    df_merged = load_raw_data()
    id_list = df_merged.ss_id.unique()

    # define hourly parameters to loop through each chunk
    hourly_params = ["temperature_2m", "weather_code", "cloud_cover", "is_day", "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation"]

    # Preprocess hourly time data to create an index mapping timestamps to indices
    data = pd.read_json('data/raw_data/weather_api.json')
    hourly_time = data['hourly']['time']
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(hourly_time)}

    # Modify the get_solar_feature function to use the index mapping
    def get_solar_feature(row, param, data, timestamp_index):
        solar_feature = data['hourly'][param]
        timestamp = row['formatted_timestamp']

        # Check if timestamp exists in the index
        if timestamp in timestamp_index:
            idx = timestamp_index[timestamp]
            return solar_feature[idx]
        else:
            return np.nan

    # Wrap entire code into for loop to iterate through each property
    for id in id_list:
        df_merged_multiple = df_merged[df_merged['ss_id']== id]

        data = pd.read_json('data/raw_data/weather_api.json')

        # loop through each weather param and populate weather features to DF
        for param in hourly_params:
            df_merged_multiple[param] = df_merged_multiple.apply(lambda row: get_solar_feature(row, param, data, timestamp_index), axis=1)
        print(f'completed property id:{id}')

        # # Export DataFrame to CSV and concatenate
        # csv_filename = f"../data/processed_data/ldn_energy_supply.csv"
        # if os.path.exists(csv_filename):
        #     df_merged_multiple.to_csv(csv_filename, mode='a', header=False, index=False)
        # else:
        #     df_merged_multiple.to_csv(csv_filename, index=False)

    print('--------chunking complete!------')

    return df_merged


def get_training_data():
    '''
    function preprocesses the feature engineered dataset to be passed into RNN model
    '''
    # Set train test split date
    d = datetime(2020, 1, 1)

    # define training data of all properties
    file_path = f'{os.getcwd()}/raw_data/ldn_energy_supply.csv'
    #file_path = f'{os.getcwd()}/market/models/ldn_energy_supply.csv'
    training_data = pd.read_csv(file_path, low_memory=False)

    # Line removed
    #training_data = pd.read_csv('data/processed_data/ldn_energy_supply.csv')

    # define sample set of 1 property
    id_list = training_data.ss_id.unique()
    training_sample = training_data[training_data['ss_id'] == id_list[0]]
    training_sample.drop_duplicates(inplace=True)

    # fill NA in target to ensure no empty rows before training
    training_sample.isnull().sum()
    training_sample['generation_wh'].fillna(0, inplace=True)

    # Preprocess for RNN
    # Step 1: Split data into training and testing sets
    X = training_sample[[
        'is_day',
        'cloud_cover',
        'weather_code',
        "temperature_2m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "terrestrial_radiation"
    ]].values

    y = training_sample[['generation_wh']].values

    dates = training_sample[['timestamp']]
    dates['timestamp'] = pd.to_datetime(dates['timestamp']).dt.tz_localize(None)

    X_train = X[dates['timestamp'] - d < timedelta(0)]
    X_test = X[dates['timestamp'] - d > timedelta(0)]

    y_train = y[dates['timestamp'] - d < timedelta(0)]
    y_test = y[dates['timestamp'] - d > timedelta(0)]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test


def get_test_data(d):
    d = d.replace(year = 2020)

    # define training data of all properties
    file_path = f'{os.getcwd()}/raw_data/ldn_energy_supply.csv'
    #file_path = f'{os.getcwd()}/market/models/ldn_energy_supply.csv'
    training_data = pd.read_csv(file_path, low_memory=False)

    # Line removed
    #training_data = pd.read_csv('data/processed_data/ldn_energy_supply.csv')

    # define sample set of 1 property
    id_list = training_data.ss_id.unique()
    training_sample = training_data[training_data['ss_id'] == id_list[0]]
    training_sample.drop_duplicates(inplace=True)

    # fill NA in target to ensure no empty rows before training
    training_sample.isnull().sum()
    training_sample['generation_wh'].fillna(0, inplace=True)

    # Preprocess for RNN
    # Step 1: Split data into training and testing sets
    X = training_sample[[
        'is_day',
        'cloud_cover',
        'weather_code',
        "temperature_2m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "terrestrial_radiation"
    ]].values

    y = training_sample[['generation_wh']].values

    dates = training_sample[['timestamp']]
    dates['timestamp'] = pd.to_datetime(dates['timestamp']).dt.tz_localize(None)
    #print(dates['timestamp'][dates['timestamp'] - d >= timedelta(0)].iloc[:24])
    X_test = X[dates['timestamp'] - d >= timedelta(0)]
    X_test = X_test[:168]
    y_test = y[dates['timestamp'] - d >= timedelta(0)]
    y_test = y_test[:168]
    dates = dates[dates['timestamp'] - d >= timedelta(0)]
    dates = dates[:168]

    return X_test, y_test, dates


def preprocess_data(X_train, X_test, y_train, y_test):
    # Scale X
    Xscaler = MinMaxScaler(feature_range=(0, 1))
    Xscaler.fit(X_train)
    scaled_X_train = Xscaler.transform(X_train)
    scaled_X_test = Xscaler.transform(X_test)
    # Save X scaler
    scaler_filename = f'{os.getcwd()}/raw_data/X_scaler.save'
    joblib.dump(Xscaler, scaler_filename)

    # Scale y
    Yscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler.fit(y_train)
    scaled_y_train = Yscaler.transform(y_train)
    scaled_y_test = Yscaler.transform(y_test)
    # Save y scaler
    scaler_filename = f'{os.getcwd()}/raw_data/Y_scaler.save'
    joblib.dump(Yscaler, scaler_filename)

    print('--training data loaded--')

    return scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test


def train_model(scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test):
    '''
    Train a model based on the training data set
    Added in the custom activation
    '''
    es = EarlyStopping(patience=10, restore_best_weights=True)


    def custom_activation(x):
        return tf.maximum(x, 0)

    model = Sequential()
    model.add(layers.Dense(32, input_dim=9, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='linear'))
    model.add(layers.Dense(1, activation=custom_activation))
    model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

    model.fit(scaled_X_train, scaled_y_train, batch_size=64, validation_split=0.1, epochs=1000, callbacks=[es])
    print('--model trained sucessfuly--')

    # Save the model
    model.save("market/models/deep_model.keras")
    print('model saved')
    return model


def weekly_validation(d):
    '''
    define 7 days period for validation based on custom date
    outputs the test and predicted data for PV energy generation
    '''
    d = d.replace(minute = 0, second = 0, microsecond = 0)
    # Get test data
    X_test, y_test, dates = get_test_data(d)

    # Load Scalers
    scaler_filename = "market/models/X_scaler.save"
    Xscaler = joblib.load(scaler_filename)
    scaler_filename = "market/models/Y_scaler.save"
    Yscaler = joblib.load(scaler_filename)

    # Scale X_test
    scaled_X_test = Xscaler.transform(X_test)

    # specify custom activation function
    def custom_activation(x):
        return tf.maximum(x, 0)

    # load the model
    file_path = f'{os.getcwd()}/market/models/deep_model.keras'
    model = tf.keras.models.load_model(file_path, custom_objects={'custom_activation': custom_activation})
    predictions = model.predict(scaled_X_test)

    # Inverse transform predictions and true values
    predictions_inverse = Yscaler.inverse_transform(predictions)

    # specify date set
    dates.reset_index(inplace = True)
    dates.drop(columns=['index'], inplace = True)
    date_df = pd.to_datetime(dates['timestamp']).dt.tz_localize(None)
    pred = pd.DataFrame(predictions_inverse, columns = ['predict'])
    pred_df = pd.merge(date_df, pred, left_index = True, right_index = True)
    test = pd.DataFrame(y_test, columns = ['test'])
    pred_df = pd.merge(pred_df, test, left_index = True, right_index = True)

    pred_df.rename(columns={'timestamp' : 'date'}, inplace = True)
    pred_df['date'] = pred_df['date'].apply(lambda x: x.replace(year = d.year))

    # Fill in date gaps
    first_date = d
    last_date = pred_df['date'].iloc[len(pred_df['date'])-1]
    last_date = last_date.replace(year = d.year)
    df = pd.date_range(start = first_date, end=last_date, freq = 'h').to_frame(name='date')
    df.reset_index(inplace= True)
    df.drop(columns=['index'], inplace=True)
    df_results = pd.merge(df, pred_df, how = 'left', on = 'date')
    df_results = df_results[:168]
    # Fill in empty data gaps with 0
    df_results['test'] = df_results['test'].fillna(0)
    df_results['predict'] = df_results['predict'].fillna(0)

    return df_results


def get_prediction():
    '''
    this function calls a 7 week forecast from API then preprocesses before passing through model for prediction
    Uses a pretrained model and pretrained scalers
    '''
    def custom_activation(x):
        return tf.maximum(x, 0)

    # load scalers
    scaler_filename = "market/models/X_scaler.save"
    Xscaler = joblib.load(scaler_filename)

    scaler_filename = "market/models/Y_scaler.save"
    Yscaler = joblib.load(scaler_filename)

    # load the model
    file_path = f'{os.getcwd()}/market/models/deep_model.keras'
    loaded_model = tf.keras.models.load_model(file_path, custom_objects={'custom_activation': custom_activation})

    # Get 7 day forecast from API
    base_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 51.42,
        "longitude": -0.19,
        "hourly": [
            "is_day",
            "cloud_cover",
            "weather_code",
            "temperature_2m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation"
        ],
        "timezone": "Europe/London"
    }

    # save response as Dataframe
    responses = requests.get(base_url, params).json()
    future_forecast = pd.DataFrame()
    for param in params['hourly']:
        future_forecast[param] = responses['hourly'][param]

    # convert to numpy array
    forecast_values = future_forecast.values

    # Scale features and target
    scaled_forecast = Xscaler.transform(forecast_values)

    # Predict with saved model
    forecast_prediction = loaded_model.predict(scaled_forecast)
    forecast_prediction_actual = Yscaler.inverse_transform(forecast_prediction)

    prediction = pd.DataFrame(forecast_prediction_actual, columns = ['kwh'])
    forecast = future_forecast[['weather_code']]
    pred_df = pd.merge(prediction, forecast, left_index = True, right_index = True)
    return pred_df

def save_gen_model():
    X_train, X_test, y_train, y_test = get_training_data()
    scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = preprocess_data(X_train, X_test, y_train, y_test)
    train_model(scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test)

def run_gen_model():
    final_prediction = get_prediction()
    return final_prediction


if __name__ == '__main__':
    ''''
    Uncomment required steps
    '''
    #X_train, X_test, y_train, y_test = get_training_data()
    #scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = preprocess_data(X_train, X_test, y_train, y_test)
    #model = train_model(scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test)
    #final_prediction = get_prediction()
    #print(final_prediction)
    #d = datetime(2024,3,28,00,00,0) # start date of evaluation
    d = datetime(2023,3,4,00,00,0)
    #print(d)
    df_results = weekly_validation(d)
    #print(df_results.iloc[10:])
    #print(df_results.columns)

    #save_gen_model()
