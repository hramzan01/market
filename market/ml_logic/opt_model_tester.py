import pandas as pd
import numpy as np

from optimiser_model import optimiser_model

'''
This is a file for testing the optimiser model
'''

# set up the number of time points
time_points = 24*7

# Create Data
# create sales data. Scale between 0.5 and 1
sell = np.random.random((time_points,1))/5+0.2
sell_price = pd.DataFrame(data=sell, columns=['SalePrice_£/kwh'])

# create purchase data. Use a two time sell price multiplier
purchase_price = sell_price.rename(columns={'SalePrice_£/kwh':'PurchasePrice_£/kwh'})
purchase_price['PurchasePrice_£/kwh'] = purchase_price['PurchasePrice_£/kwh'].apply(lambda x: x*2)

# create generation data. Average of 0.1 kwh per hour.
generation = np.random.random((time_points,1))/2
generation = pd.DataFrame(data=generation, columns=['Generation_kwh'])

# create comsumption data. Average of 0.3 kwh per hour.
consumption = np.random.random((time_points,1))/5+0.2
consumption = pd.DataFrame(data=consumption, columns=['Consumption_kwh'])

# Combine all data into a final dataframe
df = pd.concat([sell_price, purchase_price, generation, consumption], axis=1)

#price_week, battery_store, energy_bought, energy_sold = optimiser_model(df)
#print(price_week)

df = np.array(df)
df = np.concatenate((df,np.zeros((168,1))),axis=1)
print(df)
for i in range(168):
    # if generation is more than consumption
    if df[i,2] > df[i,3]:
        #profit is from sales
        df[i,4] = (df[i,2] - df[i,3]) * df[i,0]
    elif df[i,2] < df[i,3]:
        #loss is from purhcase
        df[i,4] = (df[i,2] - df[i,3]) * df[i,1]
    else:
        df[i,4] = 0
print(df)
baseline = np.sum(df[:,4])
print(baseline)
