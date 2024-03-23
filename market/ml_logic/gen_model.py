# imports
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import os


def custom_activation(x):
    '''Set up a custom activation function for RNN'''
    return tf.maximum(x, 0)


def create_dataset(X, y, length, batch_size):
    ''' set up the datatset'''
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((
        x.batch(length),
        y.skip(length - 1)
    )))
    return dataset.batch(batch_size).prefetch(1)


def predict_gen():
    ''' Predcit the energy generation from solar panels'''


    def custom_activation(x):
        return tf.maximum(x, 0)

    # load the model
    file_path = f'{os.getcwd()}/market/models/rnn_model.keras'
    print(file_path)

    loaded_model = tf.keras.models.load_model(file_path, custom_objects={'custom_activation': custom_activation})

    # get the weather forecast data from the API
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
    responses = requests.get(base_url, params).json()

    # format the API data
    future_forecast = pd.DataFrame()
    for param in params['hourly']:
        future_forecast[param] = responses['hourly'][param]

    forecast_values = future_forecast.values

    # Scale features and target
    Xscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler = MinMaxScaler(feature_range=(0, 1))


    # Line added AEOXLEY as approximation to check code
    #TODO: save the fitted files
    Xscaler.fit(forecast_values)


    scaled_forecast = Xscaler.transform(forecast_values)

    # Create prediction dataset
    n_input = 24  # Number of samples/rows/timesteps to look in the past to forecast the next sample
    batch_size = 64  # Number of timeseries samples in each batch
    forecast_dataset = create_dataset(scaled_forecast, np.zeros_like(forecast_values[:,0]), length=n_input, batch_size=batch_size)
    forecast_prediction = loaded_model.predict(forecast_dataset)
    forecast_prediction_actual = Yscaler.inverse_transform(forecast_prediction)

    # Calculate the difference in lengths
    length_diff = len(scaled_forecast) - len(forecast_prediction_actual)

    # Append zeros to forecast_prediction to match the length of scaled_forecast(missing last 24 hours)
    forecast_prediction_extended = np.append(forecast_prediction_actual, np.zeros(length_diff))
    future_forecast['kwh'] = forecast_prediction_extended

    final_prediction = future_forecast[[
        'weather_code',
        'kwh'
    ]]
    return final_prediction

#final_prediction.to_csv('../../raw_data/final_prediction.csv')
#final_prediction
if __name__=='__main__':
    final_prediction = predict_gen()
    final_prediction
