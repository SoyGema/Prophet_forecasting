#Import statements for dealing with data 

import pandas as pd
import numpy as np 
from fbprophet import Prophet

#Python
#Load your dataset 
df = pd.read_csv('..../Prophet3.csv')

#Rename columns for forecasting 
#You should have two columns 
df.columns = ['ds', 'y']
df.columns.values[1]
df.columns.values[0]


#log-transform the y variable
#to linearize a relationship
#simplify the number and complexity of interaction terms
df['y'] = np.log(df['y'])
df.head()

type(df['ds'][0])

#------------------------------#
m = Prophet()
m.fit(df)


#Select the days to make a new dataframe
future = model.make_future_dataframe(periods=50)

#This is the prediction, literally speaking 
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#Plot the results 
%matplotlib inline
c = m.plot(forecast);

#Plot the components
m.plot_components(forecast);
#------------------------------#
