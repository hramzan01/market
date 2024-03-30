'''
Optimiser model evaluation
Runing the final optimiser model
Improves by using inputs of acorn group, date.
'''

# Imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from datetime import datetime
from datetime import timedelta
import os

from cons_model import cons_model
from energy_price_model import *
from market.ml_logic.gen_model_deep import *


import warnings
warnings.simplefilter('ignore')

global battery_size, battery_charge, time_points


def data_collect(d,solar_size, acorn = 'A'):
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
    price_actual.rename(columns={'y':'SalePrice_p/kwh'}, inplace= True)
    price_pred.rename(columns={'yhat':'SalePrice_p/kwh'}, inplace= True)

    # Run the consumption model
    cons_actual, cons_prediction = cons_model(acorn, date = d)
    cons_actual.rename(columns={'y':'Consumption_kwh'}, inplace= True)
    cons_prediction.rename(columns={'yhat':'Consumption_kwh'}, inplace= True)

    # Run the Generation model
    gen = weekly_validation(d)
    #gen['date'] = gen['date'].apply(lambda x: x.replace(year = d.year))
    gen.set_index('date', inplace = True)
    gen_actual = gen[['test']]/1000 * solar_size
    gen_actual.rename(columns={'test':'Generation_kwh'}, inplace = True)
    gen_pred = gen[['predict']]/1000 * solar_size
    gen_pred.rename(columns={'predict':'Generation_kwh'}, inplace = True)
    # Combine the data into an actual dataframe
    price_buy = (price_actual[['SalePrice_p/kwh']] * 2)
    price_buy = price_buy.rename(columns={'SalePrice_p/kwh':'PurchasePrice_p/kwh'})
    actual_df = pd.concat([price_actual, price_buy, gen_actual, cons_actual['Consumption_kwh']], axis = 1)

    # Combine the data into a predicted dataframe
    price_buy = (price_pred[['SalePrice_p/kwh']] * 2)
    price_buy = price_buy.rename(columns={'SalePrice_p/kwh':'PurchasePrice_p/kwh'})
    predicted_df = pd.concat([price_pred, price_buy, gen_pred, cons_prediction], axis = 1)

    # Store the data for future use
    file_path = f'{os.getcwd()}/market/models/model_data.csv'
    actual_df.to_csv(file_path)

    # Return the final dataframes
    return actual_df, predicted_df


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
            x0[i] = 0
            x1[i] = 0
    # Set bounds
    # lower bound for x0 is 0, upper bound is 3 (assumption set from grid)
    # lower bound for x1 is 0, upper bound is the PV energy generation
    lb =np.concatenate((np.ones(time_points)*0, np.ones(time_points)*0),axis = 0)
    ub =np.concatenate((np.ones(time_points)*3, df[:,2]), axis = 0)
    bounds = Bounds(lb=lb, ub=ub)
    # concatanate x0 and x1 for the model
    x_input = np.concatenate((x0,x1),axis=0)
    print('minimiser started')
    # Model Run: minimisation. maxiter = 100000.
    res = minimize(
        profit,
        x_input,
        bounds = bounds,
        method='nelder-mead',
        options={'xatol': 1e-14, 'maxiter':100000, 'disp': True}
        )

    # Work out the minimum cost for energy from the minimisation
    price_week = profit(res.x)
    print('minimiser finished')
    return price_week


def baseline_model(data):
    '''
    A model which takes in the results of three seperate models:
    Energy consumption, PV Energy Gen, Energy Price
    and outputs a baseline profitability
    '''
    # Input data must be in the form:
    # SalePrice_£/kwh	PurchasePrice_£/kwh	Generation_kwh	Consumption_kwh

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
    return baseline_cost_no_solar


def multiple_evaluate_full_model(battery_size, battery_charge,solar_size, acorn = 'A'):
    '''
    Function for running the evaluation function multiple times and saving the reults
    '''
    d = datetime(2023,3,4,00,00,0)
    td = timedelta(days=3)
    df = pd.DataFrame({'actual_price_week':[],'pred_price_week':[],'actual_baseline_cost':[],'pred_baseline_cost':[], 'act_baseline_cost_no_solar':[],'pred_baseline_cost_no_solar':[]}, index = [])

    for i in range(3): # 90
        print(f'model {i} evaluation in progress...')
        actual_df, predicted_df = data_collect(d, solar_size)
        actual_df = actual_df.iloc[:time_points]
        predicted_df = predicted_df.iloc[:time_points]
        # Use actual data
        actual_price_week  = optimiser_model(actual_df, battery_charge,battery_size)
        # Use predicted data
        pred_price_week= optimiser_model(predicted_df,battery_charge,battery_size)
        # calculate baseline data
        actual_baseline_cost, baseline_price = baseline_model(actual_df)
        pred_baseline_cost, baseline_price = baseline_model(predicted_df)

        act_baseline_cost_no_solar = baseline_model_no_solar(actual_df)
        pred_baseline_cost_no_solar = baseline_model_no_solar(predicted_df)
        # append to dataframe
        df_to_add = pd.DataFrame([[actual_price_week,pred_price_week,actual_baseline_cost,pred_baseline_cost, act_baseline_cost_no_solar, pred_baseline_cost_no_solar]], columns = ['actual_price_week','pred_price_week','actual_baseline_cost','pred_baseline_cost', 'act_baseline_cost_no_solar','pred_baseline_cost_no_solar'],index =[d])
        df = pd.concat([df, df_to_add])
        d = d + td
        df.to_csv(f'{os.getcwd()}/market/models/profit_results.csv')
    return df


def predict_model_accuracy(d, battery_size, battery_charge, acorn = 'A'):
    '''
    This function runs the full model for optimising profit
    The model outputs the cost for one week based on the optimised scenario
    And iterates through running the actual and predicted data
    '''
    actual_df, predicted_df = data_collect(d)
    #print(actual_df)
    # SalePrice_p/kwh    PurchasePrice_p/kwh    Generation_kwh    Consumption_kwh

    ##Use actual data
    actual_price_week, battery_store, price_energy_bought, price_energy_sold = optimiser_model(actual_df,battery_charge=battery_charge, battery_size = battery_size)
    print('Test 1/5 Done...')
    ## Use predicted data
    pred_price_week, battery_store_pred, price_energy_bought_pred, price_energy_sold_pred = optimiser_model(predicted_df,battery_charge=battery_charge, battery_size = battery_size)
    print('Test 2/5 Done...')
    ## use actual price data
    price_df = pd.concat([actual_df['SalePrice_p/kwh'], actual_df['PurchasePrice_p/kwh'], predicted_df['Generation_kwh'], predicted_df['Consumption_kwh']], axis = 1)
    price_actual_price_week, battery_store_pred, price_energy_bought_pred, price_energy_sold_pred = optimiser_model(price_df,battery_charge=battery_charge, battery_size = battery_size)
    print('Test 3/5 Done...')
    gen_df = pd.concat([predicted_df['SalePrice_p/kwh'], predicted_df['PurchasePrice_p/kwh'], actual_df['Generation_kwh'], predicted_df['Consumption_kwh']], axis = 1)
    gen_actual_price_week, battery_store_pred, price_energy_bought_pred, price_energy_sold_pred = optimiser_model(gen_df,battery_charge=battery_charge, battery_size = battery_size)
    print('Test 4/5 Done...')
    cons_df = pd.concat([predicted_df['SalePrice_p/kwh'], predicted_df['PurchasePrice_p/kwh'], predicted_df['Generation_kwh'], actual_df['Consumption_kwh']], axis = 1)
    cons_actual_price_week, battery_store_pred, price_energy_bought_pred, price_energy_sold_pred = optimiser_model(cons_df,battery_charge=battery_charge, battery_size = battery_size)
    print('Test 5/5 Done...')
    return actual_price_week, pred_price_week, price_actual_price_week, gen_actual_price_week, cons_actual_price_week


if __name__ == '__main__':
    battery_size = 10 # total size
    battery_charge = 5 # initial charge amount
    time_points = 3*24 # hours
    solar_size = 5

    df = multiple_evaluate_full_model(battery_size, battery_charge,solar_size, acorn = 'A')
    print(df)
