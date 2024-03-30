import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from energy_price_pred.energypricepred import *
import json
from market.ml_logic.optimsier_model_var_in_deep import *

import matplotlib.pylab as plt

app = FastAPI()

@app.get("/predict")
def predict(battery_size: int, # 5 total size
            battery_charge: int): # 1 initial charge amount
    """predicting Buy/Sell price as first pass"""

    # date=pd.Timestamp(date)
    output_keys = ['SalePrice_p/kwh', 'PurchasePrice_p/kwh', 'Generation_kwh', 'Consumption_kwh']

    res = run_full_model_api(int(battery_size), int(battery_charge))

    res_pred_saleprice = pd.DataFrame.from_dict(res['predicted_data']['SalePrice_p/kwh']) #pd.DataFrame.from_dict(res['predicted_data'])
    res_pred_all= res['predicted_data']['SalePrice_p/kwh']#pd.DataFrame.from_dict(res['predicted_data'])
    res_opt_batt= pd.DataFrame(res['optimised_battery_storage'])
    # res_opt_buyprice= pd.DataFrame(res['optimised_energy_purchase_price'])
    # res_opt_sellprice= pd.DataFrame(res['optimised_energy_sold_price'])
    # res_opt_baseprice= pd.DataFrame(res['baseline_hourly_price'])
    # res_weather_code = res['weather_code']
    # res_weather_code_reset = res_weather_code.reset_index()
    # res_weather_code_reset = res_weather_code_reset.drop(columns=['ds'])
    # res_weather_code_reset = res_weather_code_reset['weather_code'].tolist()

    # # output = {'prediction_data': res_pred_all}#, 'res_opt_batt': res_opt_batt, 'res_opt_buyprice': res_opt_buyprice,
    # #           #'res_opt_sellprice': res_opt_sellprice, 'res_opt_baseprice': res_opt_baseprice}#,
    # #           #'res_weather_code': res_weather_code_reset}
    # output = {'res_opt_batt': res_opt_batt, 'res_opt_buyprice': res_opt_buyprice,
    #           'res_opt_sellprice': res_opt_sellprice, 'res_opt_baseprice': res_opt_baseprice,
    #           'res_weather_code': res_weather_code_reset}
    # return output# {'keys': str(res.keys())}

    # results_dict = {
    #     'SalePrice': res_pred_saleprice,
    #     ''
    # }
    output = {'prediction_data': res_pred_all}
    return output

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
