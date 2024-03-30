'''
Optimsier model variable inputs
Runing the final optimiser model
Improved by using inputs of acorn group,
Just runs a prediction

Requirements in saving models:
raw_data/ldn_energy_supply.csv
raw_data/ACORN_A_processed.csv
'''

# Imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from datetime import datetime
from datetime import timedelta
import os
import time

from market.ml_logic.cons_model import *
from market.ml_logic.energy_price_model import *
from market.ml_logic.gen_model_deep import *

import warnings
warnings.simplefilter('ignore')

global battery_size, battery_charge, time_points



def data_collect_save_models(d, acorn = 'A'):
    '''
    This function runs the models and saves the data for a given date
    Where d is today's date
    '''
    d=d.replace(hour = 0, minute = 0, second = 0)
    cons_save_model(X ='A', date=d)
    price_save_model(date = d, forecast_days = 7)
    #save_gen_model()


def data_collect_prediction(d_input, acorn = 'A'):
    '''
    This function takes in the start date of interest
    and collects the predictions from the three other models
    Energy consumption, PV Energy Gen, Energy Price
    The function outputsthe predicted data
    '''
    # Format input date to be an hourly date
    d=d_input.replace(hour = 0, minute = 0, second = 0)

    # Run the price model
    price_pred = price_load_model(d, forecast_days = 7)
    price_pred.rename(columns={'yhat':'SalePrice_p/kwh'}, inplace= True)
    # Run the consumption model
    cons_prediction = cons_load_model(d, forecasted_days = 7)
    cons_prediction.rename(columns={'yhat':'Consumption_kwh'}, inplace= True)
    # Run the Generation model
    gen = get_prediction()
    gen['ds']=price_pred.reset_index()['ds']
    gen.set_index('ds', inplace = True)
    weather_code = gen.drop(columns =['kwh'])
    gen.drop(columns = ['weather_code'], inplace = True)
    gen.rename(columns={'kwh':'Generation_kwh'}, inplace = True)
    gen = gen / 500

    # Combine the data into a predicted dataframe
    price_buy = (price_pred[['SalePrice_p/kwh']] * 2)
    price_buy = price_buy.rename(columns={'SalePrice_p/kwh':'PurchasePrice_p/kwh'})
    predicted_df = pd.concat([price_pred, price_buy, gen, cons_prediction], axis = 1)

    # Store the data for future use
    file_path = f'{os.getcwd()}/market/models/model_data.csv'
    predicted_df.to_csv(file_path)
    # Return the final dataframes
    return predicted_df, weather_code


def optimiser_model(data, battery_charge, battery_size):
    '''
    A model which takes in a dataframe with the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a prediction based on when to buy and sell
    along with the total profitability of the period
    '''
    # Input data must be in the form:
    # SalePrice_p/kwh    PurchasePrice_p/kwh    Generation_kwh    Consumption_kwh

    # load data if using repeatedly for efficient use:
    #file_path = f'{os.getcwd()}/market/models/model_data.csv'
    #data = pd.read_csv(file_path, index_col='ds')

    # convert data into numpy array
    df = np.array(data)

    # set up profit function
    def profit(x_input):
        '''
        Function to be minimised for the optimsation problem
        '''
        cost_punishment = 0 # initial cost punishment
        cost_punishment_increment = 1000 # p

        x0 = x_input[0:time_points]
        x1 = x_input[time_points:]

        battery = np.zeros(time_points+1)
        battery[0] = battery_charge

        for i in range(len(battery)-1):
            battery[i+1] = battery[i] + df[i,2] - df[i,3] + x0[i] - x1[i]
            if battery[i + 1] > battery_size:
                cost_punishment += cost_punishment_increment
            if battery[i+1] < 0:
                cost_punishment += cost_punishment_increment

        buy = x0[:] * df[:,1]
        sell = x1[:] * df[:,0]

        cost = np.sum(buy - sell) + cost_punishment
        battery_benefit = battery[time_points] * np.mean(df[i,0])
        return cost - battery_benefit

    # Model set up
    x0 = np.array(df[:,3]) # initial purchase amount
    x1 =  np.array(df[:,2]) # initial sale amount
    # Improvement on X0 and X1 initial guesses
    for i in range(time_points):
        # if generation is more than consumption
        if df[i,2] > df[i,3]:
           x0[i] = 0
           x1[i] = df[i,2] - df[i,3]
        elif df[i,2] < df[i,3]:
            #loss is from purchase
            x0[i] = df[i,3] - df[i,2]
            x1[i] = 0
        else:
            x0[i] = df[i,2]
            x1[i] = df[i,2]

    # Set bounds
    # lower bound for x0 is 0, upper bound is 3 (assumptino set from grid)
    # lower bound for x1 is 0, upper bound is the PV energy generation
    lb =np.concatenate((np.ones(time_points)*0, np.ones(time_points)*0),axis = 0)
    ub =np.concatenate((np.ones(time_points)*3, df[:,2]), axis = 0)

    bounds = Bounds(lb=lb, ub=ub)
    # concatanate x0 and x1 for the model
    x_input = np.concatenate((x0,x1),axis=0)

    # Model Run: minimisation. maxiter = 100000.
    res = minimize(
        profit,
        x_input,
        bounds = bounds,
        method='nelder-mead',
        options={'xatol': 1e-12, 'maxiter':100000, 'disp': True}
        )
    # Work out the minimum cost for energy from the minimisation
    price_week = profit(res.x)

    # set up function to run the optimal model
    def battery_storage(x_input):
        '''
        Function to be minimised for the optimsation problem
        '''
        # initialise
        x0 = x_input[0:time_points]
        x1 = x_input[time_points:]
        battery = np.zeros(time_points+1)
        battery[0] = battery_charge
        # Calculate battery level, buy amount, sell amount
        for i in range(len(battery)-1):
            battery[i+1] = battery[i] + df[i,2] - df[i,3] + x0[i] - x1[i]
        buy = x0[:] * df[:,1]
        sell = x1[:] * df[:,0]
        # Work out final profit
        cost = np.sum(buy - sell)
        battery_benefit = battery[time_points] * np.mean(df[i,0])
        return battery

    # Run the optimal scenario
    battery_store = battery_storage(res.x)

    # Find the energy bought and sold
    price_energy_bought = res.x[: time_points] * df[:,1]
    price_energy_sold = res.x[time_points :] * df[:,0]
    print('Model optimsied')
    return price_week, battery_store, price_energy_bought, price_energy_sold


def baseline_model(data):
    '''
    A model which takes in the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a baseline profitability
    '''
    # Input data must be in the form:
    # SalePrice_£/kwh	PurchasePrice_£/kwh	Generation_kwh	Consumption_kwh

    print('Basline model running')
    df = np.array(data)
    df = np.concatenate((df,np.zeros((time_points,1))),axis=1)
    for i in range(time_points):
        # if generation is more than consumption
        if df[i,2] > df[i,3]:
            #profit is from sales
            df[i,4] = (df[i,3] - df[i,2]) * df[i,0]
        elif df[i,2] < df[i,3]:
            #loss is from purchase
            df[i,4] = (df[i,3] - df[i,2]) * df[i,1]
        else:
            df[i,4] = 0
    baseline_price = df[:,4]
    baseline_cost = np.sum(df[:,4])
    return baseline_cost, baseline_price

def baseline_model_no_solar(data):
    '''
    A model which takes in the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a baseline profitability assuming no PV generation
    '''
    # Input data must be in the form:
    # SalePrice_£/kwh	PurchasePrice_£/kwh	Generation_kwh	Consumption_kwh
    df = np.array(data)
    df = np.concatenate((df,np.zeros((time_points,1))),axis=1)
    for i in range(time_points):
        df[i,4] = df[i,3] * df[i,1]
    baseline_price_no_solar = df[:,4]
    baseline_cost_no_solar = np.sum(df[:,4])
    return baseline_price_no_solar, baseline_cost_no_solar



def run_full_model_unsaved(battery_size = 10, battery_charge = 1, acorn = 'A'):
    '''
    This function runs the full model and for optimising profit
    The model outputs the cost for one week based on the optimised scenario
    And outputs the cost for one week for the baseline scenario
    This model assumes
    '''
    # Find current date and time
    date = datetime.now()
    date = date.replace(minute = 0, second = 0, microsecond = 0)
    # Save the new model
    data_collect_save_models(date, acorn = acorn)
    # Make the  prediction
    predicted_df, weather_code = data_collect_prediction(d_input = date, acorn = acorn)
    predicted_df = predicted_df.iloc[:time_points]
    # Optimise for profit
    price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(predicted_df,battery_charge=battery_charge, battery_size = battery_size)
    # Compare to baseline
    baseline_cost, baseline_price = baseline_model(predicted_df)
    return price_week, baseline_cost


def run_full_model_saved(battery_size=10, battery_charge=1, acorn = 'A'):
    '''
    This function runs the full model and for optimising profit
    The model outputs the cost for one week based on the optimised scenario
    And outputs the cost for one week for the baseline scenario
    '''
    # Find current date and time
    date = datetime.now()
    date = date.replace(minute = 0, second = 0, microsecond = 0)
    # Make the  prediction
    predicted_df, weather_code = data_collect_prediction(d_input = date, acorn = acorn)
    predicted_df = predicted_df.iloc[:time_points]
    # Optimise for profit
    price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(predicted_df,battery_charge=battery_charge, battery_size = battery_size)
    # Compare to baseline
    baseline_cost, baseline_price = baseline_model(predicted_df)
    return price_week, baseline_cost


def run_full_model_api_unsaved(battery_size, battery_charge, acorn = 'A'):
    '''
    This function runs the full model and for optimising profit
    The model outputs the cost for one week based on the optimised scenario
    And outputs the cost for one week for the baseline scenario
    This model assumes
    '''
    # Find current date and time
    date = datetime.now()
    date = date.replace(minute = 0, second = 0, microsecond = 0)
    # Save the new model
    data_collect_save_models(date, acorn = acorn)
    # Make the  prediction
    predicted_df, weather_code  = data_collect_prediction(d_input = date, acorn = acorn)
    predicted_df = predicted_df.iloc[:time_points]
    # Optimise for profit
    price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(predicted_df,battery_charge=battery_charge, battery_size = battery_size)
    # Compare to baseline
    baseline_cost, baseline_price = baseline_model(predicted_df)
    # format the data for the api
    api_output = {
        'predicted_data':predicted_df,
        'predicted_hourly_price':price_week,
        'optimised_battery_storage':battery_store,
        'optimised_energy_purchase_price':price_energy_bought,
        'optimised_energy_sold_price':price_energy_sold,
        'baseline_cost':baseline_cost,
        'baseline_hourly_price':baseline_price,
        'weather_code': weather_code
    }
    return api_output


def run_full_model_api(battery_size, battery_charge, acorn = 'A'):
    '''
    This function runs the full model and for optimising profit
    The model outputs the cost for one week based on the optimised scenario
    And outputs the cost for one week for the baseline scenario for the api
    '''
    # Find current date and time
    date = datetime.now()
    date = date.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    # Make the  prediction
    #predicted_df = data_collect_prediction(d_input = date, acorn = acorn)
    predicted_df, weather_code  = data_collect_prediction(d_input = date, acorn = acorn)
    predicted_df = predicted_df.iloc[:time_points]
    # Optimise for profit
    price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(predicted_df,battery_charge=battery_charge, battery_size = battery_size)
    # Compare to baseline
    baseline_cost, baseline_price = baseline_model(predicted_df)
    baseline_price_no_solar, baseline_cost_no_solar = baseline_model_no_solar(predicted_df)


    # format the data for the api
    api_output = {
        'predicted_data':predicted_df,
        'predicted_hourly_price':price_week,
        'optimised_battery_storage':battery_store,
        'optimised_energy_purchase_price':price_energy_bought,
        'optimised_energy_sold_price':price_energy_sold,
        'baseline_cost':baseline_cost,
        'baseline_hourly_price':baseline_price,
        'baseline_price_no_solar':baseline_price_no_solar,
        'baseline_cost_no_solar':baseline_cost_no_solar,
        'weather_code': weather_code
    }
    return api_output


if __name__ == '__main__':
    battery_size = 10 # total size
    battery_charge = 5 # initial charge amount
    time_points = 3*24 # hours

    start = time.time()
    run_full_model_api_unsaved(battery_size, battery_charge, acorn = 'A')
    start = time.time()
    api_output = run_full_model_api(battery_size, battery_charge, acorn = 'A')
    #price_week, baseline_cost = run_full_model_unsaved()
    #price_week, baseline_cost = run_full_model_saved()
    end = time.time()

    # print statements
    print(f'The model took {round((end - start),1)} seconds to run')
    #print(api_output['weather_code'])
    print('Baseline cost no solar:')
    print(f"{round(api_output['baseline_cost_no_solar']/100,2)}")
    print('Baseline cost solar:')
    print(f"£{round(api_output['baseline_cost']/100,2)}")
    print('Cost our prediction:')
    print(f"£{round(api_output['predicted_hourly_price']/100,2)}")
    #print(f'The week cost using our model is £{round(price_week/100,2)}')
    #print(f'The week cost not using our model is £{round(baseline_cost/100,2)}')
