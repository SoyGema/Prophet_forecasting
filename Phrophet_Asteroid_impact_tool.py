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
#------------------------------#


