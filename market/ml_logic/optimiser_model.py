import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from datetime import datetime
from datetime import timedelta
import os


from cons_model import cons_model
from energy_price_model import *
from gen_model_updated import *

'''
Runing all other ML Models
'''


def data_collect(d):
    '''
    This function takes in the start date of interest
    and collects the predictions from the three other models
    Energy consumption, PV Energy Gen, Energy Price
    The function outputs two pandas dataframes
    One dateaframe for the actual data and one dataframe for the predicted data
    '''
    # Format input date to be an hourly date
    d=d.replace(minute = 0, second = 0)

    # Run the price model
    price_actual, price_pred = energy_model_run(d, forecast_days = 7)
    price_actual.rename(columns={'y':'SalePrice_£/kwh'}, inplace= True)
    price_pred.rename(columns={'yhat':'SalePrice_£/kwh'}, inplace= True)

    # Run the consumption model
    cons_actual, cons_prediction = cons_model('A', date = d)
    cons_actual.rename(columns={'y':'Consumption_kwh'}, inplace= True)
    cons_prediction.rename(columns={'yhat':'Consumption_kwh'}, inplace= True)

    # Run the Generation model
    gen = run_gen_model()
    gen['ds']=price_actual.reset_index()['ds']
    gen.set_index('ds', inplace = True)
    gen.drop(columns = ['weather_code'], inplace = True)
    gen.rename(columns={'kwh':'Generation_kwh'}, inplace = True)
    gen = gen / 150


    # TODO: link the generation model here
    # Removed AEOXLEY
    #gen = pd.read_csv(f'{os.getcwd()}/raw_data/final_prediction.csv')
    #gen['ds']=price_actual.reset_index()['ds']
    #gen.drop(columns = ['Unnamed: 0', 'weather_code'], inplace = True)
    #gen.set_index('ds', inplace = True)
    #gen.rename(columns={'kwh':'Generation_kwh'}, inplace = True)
    #gen = gen / 150


    # Combine the data into an actual dataframe
    # TODO: concatanate the consumption data. make sure it comes in one dataframe
    price_buy = (price_actual[['SalePrice_£/kwh']] * 2)
    price_buy = price_buy.rename(columns={'SalePrice_£/kwh':'PurchasePrice_£/kwh'})
    actual_df = pd.concat([price_actual, price_buy, gen, cons_actual['Consumption_kwh']], axis = 1)

    # Combine the data into a predicted dataframe
    price_buy = (price_pred[['SalePrice_£/kwh']] * 2)
    price_buy = price_buy.rename(columns={'SalePrice_£/kwh':'PurchasePrice_£/kwh'})
    predicted_df = pd.concat([price_pred, price_buy, cons_prediction], axis = 1)

    # Return the dataframes
    return actual_df, predicted_df


def optimiser_model(data):
    '''
    A model which takes in a dataframe with the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a prediction based on when to buy and sell
    along with the total profitability of the period
    '''
    # TODO clean up from notebook form

    # Input data must be in the form:
    # SalePrice_£/kwh	PurchasePrice_£/kwh	Generation_kwh	Consumption_kwh
    # convert data into numpy array
    df = np.array(data)
    time_points = 7*24

    # set up profit function
    def profit(x_input):
        '''
        Function to be minimised for the optimsation problem
        '''
        time_points = 7*24

        battery_size = 5 # kwh, max battery charge
        battery_charge = 1 # kwh, initial battery charge
        cost_punishment = 0 # initial cost punishment
        cost_punishment_increment = 1000 # £

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
        battery_charge = battery[time_points] * np.mean(df[i,0])
        return cost - battery_charge

    # x0 = initial purchase amount
    x0 = np.array(df[:,3])
    #x1 = initial sale amount
    x1 =  np.array(df[:,2])

    for i in range(168):
        # if generation is more than consumption
        if df[i,2] > df[i,3]:
           x0[i] = 0
           x1[i] = df[i,2] - df[i,3]
        elif df[i,2] < df[i,3]:
            #loss is from purchase
            x0[i] = df[i,3] - df[i,2]
            x1[i] = 0
        else:
            df[i,4] = 0


    # x0 is the energy purchased
    # x1 is the energy sold
    # lower bound for x0 is 0, upper bound is 3 (assumptino set from grid)
    # lower bound for x1 is 0, upper bound is the PV energy generation
    lb =np.concatenate((np.ones(time_points)*0, np.ones(time_points)*0),axis = 0)
    ub =np.concatenate((np.ones(time_points)*3, df[:,2]), axis = 0)
    bounds = Bounds(lb=lb, ub=ub)

    # concatanate x0 and x1 for the model
    x_input = np.concatenate((x0,x1),axis=0)

    # run the minimisation
    res = minimize(
        profit,
        x_input,
        bounds = bounds,
        method='nelder-mead',
        options={'xatol': 1e-12, 'maxiter':100000, 'disp': True}
        )
    # Work out the minimum cost for energy from the minimisation
    price_week = profit(res.x)

    # ste up function to run the optimal model
    def battery_storage(x_input):
        '''
        Function to be minimised for the optimsation problem
        '''
        time_points = 24*7
        x0 = x_input[0:time_points]
        x1 = x_input[time_points:]

        battery = np.zeros(time_points+1)
        # initial battery charge
        battery[0] = 1
        # battery size
        battery_size = 5
        cost_punishment = 0
        for i in range(len(battery)-1):
            battery[i+1] = battery[i] + df[i,2] - df[i,3] + x0[i] - x1[i]
            if battery[i + 1] > battery_size:
                cost_punishment += 1000

        buy = x0[:] * df[:,1]
        sell = x1[:] * df[:,0]


        cost = np.sum(buy - sell) + cost_punishment
        battery_charge = battery[time_points] * np.mean(df[i,0])
        return battery, (cost - battery_charge)

    # Run the optimsal scenario
    (battery_store, cost) = battery_storage(res.x)


    # Find the energy bought and sold
    price_energy_bought = res.x[: time_points] * df[:,1]
    price_energy_sold = res.x[time_points :] * df[:,0]

    return price_week, battery_store, price_energy_bought, price_energy_sold


def baseline_model(data):
    '''
    A model which takes in the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a baseline profitability
    '''
    # Input data must be in the form:
    # SalePrice_£/kwh	PurchasePrice_£/kwh	Generation_kwh	Consumption_kwh

    df = np.array(data)
    df = np.concatenate((df,np.zeros((168,1))),axis=1)
    for i in range(168):
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


if __name__ == '__main__':
    actual_df, predicted_df = data_collect(datetime(2024,1,3,18,30,5))
    price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(actual_df)
    print('Battery Storage for the week:')
    print(battery_store)
    print(f'The week cost using our model is £{round(price_week,0)/100}')
    baseline, baseline_price = baseline_model(actual_df)
    print(f'The week cost not using our model is £{round(baseline,0)/100}')
