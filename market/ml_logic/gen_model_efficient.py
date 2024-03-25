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

warnings.simplefilter('ignore')

n_input = 24
batch_size = 64

def load_raw_data():
    '''
    function retrievs UKPV hourly datset and merges to combine time series with location and meta data
    '''
    # Load in Data
    df_spec = pd.read_csv('data/raw_data/metadata.csv')
    df_energy_ldn = pd.read_csv('data/raw_data/hourly_generation_ldn.csv')

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
    # define training data of all properties
    file_path = f'{os.getcwd()}/market/models/ldn_energy_supply.csv'
    training_data = pd.read_csv(file_path)

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

    y = training_sample['generation_wh'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


    # Step 2: Scale features
    Xscaler = MinMaxScaler(feature_range=(0, 1))
    Xscaler.fit(X_train)
    scaled_X_train = Xscaler.transform(X_train)
    scaled_X_test = Xscaler.transform(X_test)
    # Save X scaler
    scaler_filename = "market/models/X_scaler.save"
    joblib.dump(Xscaler, scaler_filename)


    # Scale the Y target
    Yscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler.fit(y_train.reshape(-1, 1))  # Reshape y_train for MinMaxScaler
    scaled_y_train = Yscaler.transform(y_train.reshape(-1, 1))
    scaled_y_test = Yscaler.transform(y_test.reshape(-1, 1))
    # Save Y scaler
    scaler_filename = "market/models/Y_scaler.save"
    joblib.dump(Yscaler, scaler_filename)

    # Step 3: Reshape target (y) for Keras
    scaled_y_train = scaled_y_train.reshape(-1)
    scaled_y_test = scaled_y_test.reshape(-1)

    # Step 4: Manipulate target array
    scaled_y_train = np.insert(scaled_y_train, 0, 0)
    scaled_y_train = np.delete(scaled_y_train, -1)

    # Step 5: Create tf.data.Dataset
    n_input = 24  # Number of samples/rows/timesteps to look in the past to forecast the next sample
    batch_size = 64  # Number of timeseries samples in each batch


    def create_dataset(X, y, length, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.window(length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((
            x.batch(length),
            y.skip(length - 1)
        )))
        return dataset.batch(batch_size).prefetch(1)

    train_dataset = create_dataset(scaled_X_train, scaled_y_train, length=n_input, batch_size=batch_size)
    test_dataset = create_dataset(scaled_X_test, scaled_y_test, length=n_input, batch_size=batch_size)

    print('--training data loaded--')

    return training_sample, scaled_y_test, scaled_X_train, create_dataset, train_dataset, test_dataset, Xscaler, Yscaler


def train_model():
    training_sample, scaled_y_test, scaled_X_train, create_dataset, train_dataset, test_dataset, Xscaler, Yscaler = get_training_data()

    # RNN Architecture
    # Custom activation function to ensure non-negative predictions
    def custom_activation(x):
        return tf.maximum(x, 0)

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(units=32, activation='tanh', input_shape=(n_input, scaled_X_train.shape[1])),
        tf.keras.layers.Dense(1, activation=custom_activation)
    ])

    # Compilation
    model.compile(loss='mse', optimizer='adam')

    # Fit
    model.fit(train_dataset, epochs=5, verbose=0)
    print('--model trained sucessfuly--')

    # Save the model
    model.save("../models/rnn_model.keras")
    print('model saved')
    return model


def get_prediction():
    '''
    this function calls a 7 week forecast from API then preprocesses before passing through model for prediction
    '''
    # Load the model params and model
    #training_sample, scaled_y_test, scaled_X_train, create_dataset, train_dataset, test_dataset, Xscaler, Yscaler = get_training_data()

    def create_dataset(X, y, length, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.window(length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((
            x.batch(length),
            y.skip(length - 1)
        )))
        return dataset.batch(batch_size).prefetch(1)

    def custom_activation(x):
        return tf.maximum(x, 0)

    # And now to load...
    scaler_filename = "market/models/X_scaler.save"
    Xscaler = joblib.load(scaler_filename)

    scaler_filename = "market/models/Y_scaler.save"
    Yscaler = joblib.load(scaler_filename)

    # load the model
    file_path = f'{os.getcwd()}/market/models/rnn_model.keras'
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

    # Create TensorFlow dataset
    forecast_dataset = create_dataset(scaled_forecast, np.zeros_like(forecast_values[:,0]), length=n_input, batch_size=batch_size)

    # Predict with saved model
    forecast_prediction = loaded_model.predict(forecast_dataset)

    # Inverse transform the scaled forecasted predictions to get actual KWH value
    forecast_prediction_actual = Yscaler.inverse_transform(forecast_prediction)

    # Calculate the difference in lengths
    length_diff = len(scaled_forecast) - len(forecast_prediction_actual)

    # Append zeros to forecast_prediction to match the length of scaled_forecast(missing last 24 hours)
    forecast_prediction_extended = np.append(forecast_prediction_actual, np.zeros(length_diff))
    future_forecast['kwh'] = forecast_prediction_extended

    # Save prediction as df
    final_prediction = future_forecast[[
        'weather_code',
        'kwh'
    ]]
    return final_prediction


def weekly_validation(d):
    '''
    define 7 days period for validation based on custom date
    '''
    # pass in variables for other functions
    training_sample, scaled_y_test, scaled_X_train, create_dataset, train_dataset, test_dataset, Xscaler, Yscaler = get_training_data()

    # get predictions from model
    def custom_activation(x):
        return tf.maximum(x, 0)

    # load the model
    file_path = f'{os.getcwd()}/market/models/rnn_model.keras'
    model = tf.keras.models.load_model(file_path, custom_objects={'custom_activation': custom_activation})
    #model = tf.keras.models.load_model("models/rnn_model.keras", custom_objects={'custom_activation': custom_activation})
    predictions = model.predict(test_dataset)

    # define y actual and y pred
    # Step 7: Inverse transform predictions and true values
    scaled_y_test_inverse = Yscaler.inverse_transform(scaled_y_test.reshape(-1, 1)).flatten()
    predictions_inverse = Yscaler.inverse_transform(predictions).flatten()

    # Use limiter so that length of pred and actual match
    limiter = len(predictions_inverse)
    df_validation = pd.DataFrame(
        {'test': scaled_y_test_inverse[:limiter],
        'predict': predictions_inverse}
    )

    df_validation['date'] = training_sample.timestamp[:limiter]
    df_validation['date'] = pd.to_datetime(df_validation['date']).dt.tz_localize(None)
    print(len(df_validation))
    print(df_validation.head(10))

    first_date = df_validation['date'].iloc[0]
    last_date = df_validation['date'].iloc[len(df_validation['date'])-1]
    diff_s = (last_date - first_date).total_seconds()
    hours = divmod(diff_s, 3600)[0]

    print(first_date) # 2015-05-31
    print(last_date) # 2018-12-12
    print(diff_s)
    print(hours)
    # TODO make full time dateaframe so that all dates in range will match

    index_ = df_validation[df_validation['date'] == d].index.item()
    print(index_)
    # Select the next 7 rows from the matched index
    weekly_validation = df_validation.iloc[index_:index_+168]

    return weekly_validation


def run_gen_model():
    final_prediction = get_prediction()
    return final_prediction


if __name__ == '__main__':
    ''''
    Uncomment required steps
    '''
    # load_raw_data()
    # append_weather_params()
    #get_training_data()
    #train_model()
    final_prediction = get_prediction()
    print(final_prediction)
    #final_prediction = run_gen_model()




    #d = datetime(2015,5,31,16,0,0) # start date of evaluation
    #weekly_validation = weekly_validation(d)
    #print(weekly_validation)
