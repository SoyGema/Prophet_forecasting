#Defining carrying capacity . This number should be chosing knowing more about the area or having 
## Is this a ratio? 

df['cap'] = 6.5
m = Prophet(growth='logistic')
m.fit(df)


# Python
#m.make_future_dataframe(periods=1826)
fcst = m.predict(df)
m.plot(fcst);

