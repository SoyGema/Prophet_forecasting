# Asteroid detection with Prophet forecasting
Prophet is an opensource project leaded by facebook that allows to make timeseries predictions . For more information visit 
https://facebook.github.io/prophet/
Facebook Opensource generates dataframes. This dataframe generation tool is based on forecasting. 
The following graphs shows E-Moid distribution in a time series that goes since first Asteroids registered 
V2.
Prediction with data from 1937 to 2017

![alt tag](https://github.com/SoyGema/Prophet_forecasting/blob/master/v2/Asteroids.png)

Prediction with data from 1997 to 2017

![alt tag](https://github.com/SoyGema/Prophet_forecasting/blob/master/images/descarga%20(1).png)

Prediction with data from 2017

![alt tag](https://github.com/SoyGema/Prophet_forecasting/blob/master/images/descarga%20(2).png)

Acording to new analysis taking into account EMoid and dates , Asteroids are coming nearest earth 
Find the notebook at V2 folder 




It predicts a given magnitude expressed in a timeserie. 
Acording to the documentation , phrophet shines in : 


1.hourly, daily, or weekly observations with at least a few months (preferably a year) of history

2.strong multiple “human-scale” seasonalities: day of week and time of year

3.important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl)

4.A reasonable number of missing observations or large outliers

5.historical trend changes, for instance due to product launches or logging changes

6.trends that are non-linear growth curves, where a trend hits a natural limit or saturates


In this repo you might find Facebook phrophet forecasting tool for Asteroid detection
The repository contains two different jupyter_notebooks in python, a dataset includind dates and AU of distances 
The repository also includes python code .

Find bellow the plotted results of the experiment 

ds > referes to the timeframe

y > refers to the astronomical units of distance 

![alt tag](https://github.com/SoyGema/Prophet_forecasting/blob/master/forecasting.png)
![alt tag](https://github.com/SoyGema/Prophet_forecasting/blob/master/forecasting2.png)
