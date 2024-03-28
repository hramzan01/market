import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from energy_price_pred.energypricepred import *

from market.ml_logic.optimiser_model_variable_inputs_copy_ri import *

import matplotlib.pylab as plt

app = FastAPI()

@app.get("/predict")
def predict(battery_size: int, # 5 total size
            battery_charge: int): # 1 initial charge amount
    """predicting Buy/Sell price as first pass"""

    # date=pd.Timestamp(date)
    output_keys = ['SalePrice_p/kwh', 'PurchasePrice_p/kwh', 'Generation_kwh', 'Consumption_kwh']

    res = run_full_model_api(int(battery_size), int(battery_charge))
    res_pd = pd.DataFrame.from_dict(res['predicted_data']['SalePrice_p/kwh']) #pd.DataFrame.from_dict(res['predicted_data'])
    res_pd_all= pd.DataFrame.from_dict(res['predicted_data'])
    key_list=[]
    value_list=[]
    data_saleprice = res_pd['SalePrice_p/kwh']
    for key, value in data_saleprice.items():
        key_list.append(key)
        value_list.append(value)
    #return {'res':f'key is {key_list[0]} of type {type(key_list[0])} and value is {value_list[0]}'} #{"res columns": res_pd}
    return res_pd_all


@app.get("/")
def root():
    return {'greeting': 'Hello there'}


# data = (predict(5, 1))
# saleprice = data['SalePrice_p/kwh']
# buyprice = data['PurchasePrice_p/kwh']
# x_sale, y_sale = zip(*saleprice.items()) # unpack a list of pairs into two tuples
# x_buy, y_buy = zip(*buyprice.items())

# plt.figure();
# plt.plot(x_sale, y_sale);
# plt.plot(x_buy, y_buy);
# plt.show()
