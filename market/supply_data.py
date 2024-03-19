# Import relevant libraries
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# define training data of all properties
training_data = pd.read_csv('../data/processed_data/ldn_energy_supply.csv')

# define sample set of 1 property
id_list = training_data.ss_id.unique()
training_sample = training_data[training_data['ss_id'] == id_list[0]]
training_sample.drop_duplicates(inplace=True)
training_sample['generation_wh'].fillna(0, inplace=True)


# //////////////////////////////
# Preprocess for RNN
# Step 1: Extract features (X) and target (y)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Scale features and target
Xscaler = MinMaxScaler(feature_range=(0, 1))
Xscaler.fit(X_train)
scaled_X_train = Xscaler.transform(X_train)
scaled_X_test = Xscaler.transform(X_test)

# Scale the Y
Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train.reshape(-1, 1))  # Reshape y_train for MinMaxScaler
scaled_y_train = Yscaler.transform(y_train.reshape(-1, 1))
scaled_y_test = Yscaler.transform(y_test.reshape(-1, 1))

# Step 3: Reshape target (y) for Keras
scaled_y_train = scaled_y_train.reshape(-1)
scaled_y_test = scaled_y_test.reshape(-1)

# Step 4: Manipulate target array
scaled_y_train = np.insert(scaled_y_train, 0, 0)
scaled_y_train = np.delete(scaled_y_train, -1)

# Step 5: Create tf.data.Dataset
n_input = 25  # Number of samples/rows/timesteps to look in the past to forecast the next sample
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

# //////////////////////
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

# ////////////////////////
# Step 6: Make predictions
predictions = model.predict(test_dataset) # replace with inputs from front end
