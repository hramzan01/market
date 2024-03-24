import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from energy_price_pred.energypricepred import *

from market.ml_logic.optimiser_model_variable_inputs import *

app = FastAPI()

@app.get("/predict")
def predict(date: str, # '2024-03-24 06:30:00'
            battery_size: int, # 5 total size
            battery_charge: int): # 1 initial charge amount
    """predicting Buy/Sell price as first pass"""

    date=pd.Timestamp(date)
    res = run_full_model_api(date, int(battery_size), int(battery_charge))
    res_pd = pd.DataFrame.from_dict(res['predicted_data']['SalePrice_p/kwh']) #pd.DataFrame.from_dict(res['predicted_data'])
    key_list=[]
    value_list=[]
    data = res_pd['SalePrice_p/kwh']
    for key, value in data.items():
        key_list.append(key)
        value_list.append(value)
    return {'res':f'key is {key_list[0]} and value is {value_list[0]}'}


@app.get("/")
def root():
    return {'greeting': 'Hello there'}
