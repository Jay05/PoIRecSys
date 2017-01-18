# -*- coding: utf-8 -*-
"""
@author: Jay
This is clustering module based on temporal activities - Morning, Afternoon, Evening, Night
"""

import pandas as pd
import datetime
import numpy as np

########## Weeplaces Dataset cleaning #########
weeplace_data = pd.read_csv ('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins FULL.csv')
print(weeplace_data['datetime'].head(3))
weeplace_data1 = weeplace_data.dropna(subset=['userid','placeid','datetime','category'],how='any') #Remove missing values
print(weeplace_data1.shape)
print(weeplace_data1.category.unique())

#### DATA PREPROCESSING STEP 1 #########
def assign_period():
    print('inside function')
    morning_start=datetime.time(6)
    morning_end = datetime.time(12)
    print('inside function1')
    noon_start=datetime.time(12)
    noon_end=datetime.time(17)
    print('inside function2')
    evening_start=datetime.time(17)
    evening_end=datetime.time(22)
    print('inside function4')
    night_start=datetime.time(22)
    night_end=datetime.time(6)
    print('inside function5')
    periods = {'Morning':[morning_start, morning_end], 'Afternoon':[noon_start, noon_end],'Evening':[evening_start, evening_end], 'Night':[night_start, night_end]}
    weeplace_data1['Time_Of_Day'] = 'unknown_period'
    for k, v in periods.items():
      begin = v[0].hour
      end = v[1].hour
      weeplace_data1.loc[(pd.to_datetime(weeplace_data1.datetime).apply(lambda x: x.hour) - begin) % 24 <= (end - begin) % 24, 'Time_Of_Day'] = k
assign_period()

#### DATA PREPROCESSING STEP 2 #########
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Food:.*' ,'Food')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Shops:.*' ,'Shopping')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Arts & Entertainment.*' ,'Art')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Nightlife.*' ,'Outdoor')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Outdoors:.*' ,'Outdoor')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Great Outdoor.*' ,'Outdoor')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Parks.*' ,'Outdoor')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Travel.*' ,'Outdoor')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('College.*' ,'Recreation')
weeplace_data1['category'] = weeplace_data1['category'].str.replace('Home.*' ,'Recreation')
print (weeplace_data1['Time_Of_Day'].head(3))
print(weeplace_data1.category.unique())

#Cluster 1 : Morning
weeplace_data2 = weeplace_data1.groupby('Time_Of_Day').get_group('Morning')
print (weeplace_data2['Time_Of_Day'].head(3))
print (weeplace_data2.shape)
#weeplace_data2 = period_series.to_frame.reset_index()

#Cluster 2 : Afternoon
weeplace_data3 = weeplace_data1.groupby('Time_Of_Day').get_group('Afternoon')
print (weeplace_data3['Time_Of_Day'].head(3))
print (weeplace_data3.shape)
#Cluster 1 : Evening
weeplace_data4 = weeplace_data1.groupby('Time_Of_Day').get_group('Evening')
print (weeplace_data4['Time_Of_Day'].head(3))
print (weeplace_data3.shape)
#Cluster 1 : Night
weeplace_data5 = weeplace_data1.groupby('Time_Of_Day').get_group('Night')
print (weeplace_data5['Time_Of_Day'].head(3))
print (weeplace_data5.shape)

#####Write the data to the file########
weeplace_data2.to_csv('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Morning.csv', index=False)
weeplace_data3.to_csv('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Afternoon.csv', index=False)
weeplace_data4.to_csv('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Evening.csv', index=False)
weeplace_data5.to_csv('C:/Users/User\Data POI\Data-POI\IN - weeplaces\weeplace_checkins_Night.csv', index=False)
print('Done')