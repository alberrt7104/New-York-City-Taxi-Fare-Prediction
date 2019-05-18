import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm  
import math
from igraph import *

picktimey_train=[]
picktimeh_train=[]
picktimey_test=[]
picktimeh_test=[]
empty=[]
key_old=[]
key_new=[]
c=[]
count=0
df_train =  pd.read_csv('training.csv', parse_dates=["pickup_datetime"])
df_test = pd.read_csv('testing.csv')
df_test.head()
df_train.head()
#print(df_train.head())
df_train.dtypes
df_train.describe()
df_test.describe()

# plot histogram of fare

# dfl_train[df_train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
# plt.xlabel('fare $USD')
# pt.title('Histogram')
# plot()

# print('Old size: %d' % len(df_train))
df_train = df_train[df_train.fare_amount>0]
df_train = df_train.reset_index(drop=True)
df_test = df_test[df_train.fare_amount>0]
df_test = df_test.reset_index(drop=True)
# print('New size: %d' % len(df_train))
# print('Old size: %d' % len(df_train))

df_train = df_train.dropna()
df_train = df_train.reset_index(drop=True)
df_test = df_test.dropna()
df_test = df_test.reset_index(drop=True)
# print('New size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount<100] 
df_train = df_train.reset_index(drop=True)
df_test = df_test[df_train.fare_amount<100] 
df_test = df_test.reset_index(drop=True)

df_train= df_train[df_train.passenger_count>0]
df_train = df_train.reset_index(drop=True)
df_test= df_test[df_train.passenger_count>0]
df_test = df_test.reset_index(drop=True)

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1])&(df.pickup_latitude >= BB[2])&(df.pickup_latitude <= BB[3])&(df.dropoff_longitude >= BB[0])& (df.dropoff_longitude <= BB[1]) &(df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
BB = (-74.5, -72.8, 40.5, 41.8)
df_train = df_train[select_within_boundingbox(df_train, BB)]
df_train = df_train.reset_index(drop=True)
df_test= df_test[select_within_boundingbox(df_test, BB)]
df_test = df_test.reset_index(drop=True)
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


df_train['distance_miles'] = distance(df_train.pickup_latitude, df_train.pickup_longitude,df_train.dropoff_latitude, df_train.dropoff_longitude)
df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude,df_test.dropoff_latitude, df_test.dropoff_longitude)

df_train.distance_miles.hist(bins=50, figsize=(12,4))
df_train.distance_miles.describe()
df_test.distance_miles.hist(bins=50, figsize=(12,4))
df_test.distance_miles.describe()

idx = (df_train.distance_miles < 15) & (df_train.fare_amount < 100)
idxt= (df_test.distance_miles < 15) & (df_test.fare_amount < 100)
df_train = df_train[idx]
df_train = df_train.reset_index(drop=True)
df_test = df_test[idxt]
df_test = df_test.reset_index(drop=True)

idx = (df_train.distance_miles >= 0.05)
idxt= (df_test.distance_miles >= 0.05)
df_train = df_train[idx]
df_train = df_train.reset_index(drop=True)
df_test = df_test[idxt]
df_test = df_test.reset_index(drop=True)

jfk = (-73.7822222222, 40.6441666667) 
nyc = (-74.0063889, 40.7141667)
ewr = (-74.175, 40.69) 
lgr = (-73.87, 40.77)
df_train['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0], df_train.dropoff_latitude, df_train.dropoff_longitude)
df_train['pickup_distance_to_nyc'] = distance(nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_nyc'] = distance(nyc[1], nyc[0], df_train.dropoff_latitude, df_train.dropoff_longitude)
df_test['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0], df_test.pickup_latitude, df_test.pickup_longitude)
df_test['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0], df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['pickup_distance_to_nyc'] = distance(nyc[1], nyc[0], df_test.pickup_latitude, df_test.pickup_longitude)
df_test['dropoff_distance_to_nyc'] = distance(nyc[1], nyc[0], df_test.dropoff_latitude, df_test.dropoff_longitude)

df_train['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0], df_train.dropoff_latitude, df_train.dropoff_longitude)
df_train['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0], df_train.dropoff_latitude, df_train.dropoff_longitude)
df_test['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0], df_test.pickup_latitude, df_test.pickup_longitude)
df_test['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0], df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], df_test.pickup_latitude, df_test.pickup_longitude)
df_test['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0], df_test.dropoff_latitude, df_test.dropoff_longitude)



idx = ~((df_train.pickup_distance_to_jfk < 1) | (df_train.dropoff_distance_to_jfk < 1)| (df_train.pickup_distance_to_lgr < 1) | (df_train.dropoff_distance_to_lgr < 1)|(df_train.pickup_distance_to_ewr < 1) | (df_train.dropoff_distance_to_ewr < 1))
df_train=df_train[idx]
df_train = df_train.reset_index(drop=True)
idx = ~((df_train.pickup_distance_to_nyc < 1) | (df_train.dropoff_distance_to_nyc < 1))
df_train=df_train[idx]
df_train = df_train.reset_index(drop=True)

idx = ~((df_test.pickup_distance_to_jfk < 1) | (df_test.dropoff_distance_to_jfk < 1)|(df_test.pickup_distance_to_lgr < 1) | (df_test.dropoff_distance_to_lgr < 1)|(df_test.pickup_distance_to_ewr < 1) | (df_test.dropoff_distance_to_ewr < 1))
df_test=df_test[idx]
df_test = df_test.reset_index(drop=True)
idx = ~((df_test.pickup_distance_to_nyc < 1) | (df_test.dropoff_distance_to_nyc < 1))
df_test=df_test[idx]
df_test = df_test.reset_index(drop=True)

picktimeh_train=df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
picktimey_train=df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
picktimeh_test=df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
picktimey_test=df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)

#print(picktime_h)

x=[]
y=[]
j=[]
k=[]
# print(len(df_train.passenger_count))
# print(len(picktimey_train))
# print(len(df_train.pickup_longitude))
# print(len(df_train.pickup_longitude))
# print(len(df_train.pickup_latitude))
# print(len(df_train.dropoff_longitude))
# print(len(df_train.dropoff_latitude))
df_train['distance_miles'] = distance(df_train.pickup_latitude, df_train.pickup_longitude,df_train.dropoff_latitude, df_train.dropoff_longitude)
df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude,df_test.dropoff_latitude, df_test.dropoff_longitude)

for i in range(len(df_train.passenger_count)):
	if picktimeh_train[i]>12:
		x.append([df_train.distance_miles[i],picktimey_train[i],0,df_train.pickup_longitude[i],df_train.pickup_latitude[i],df_train.dropoff_longitude[i],df_train.dropoff_latitude[i],df_train.passenger_count[i]])
		#x.append([df_train.distance_miles[i],picktimey_train[i],0,df_train.passenger_count[i]])
		y.append(str(df_train.fare_amount[i]))
	else:
		x.append([df_train.distance_miles[i],picktimey_train[i],1,df_train.pickup_longitude[i],df_train.pickup_latitude[i],df_train.dropoff_longitude[i],df_train.dropoff_latitude[i],df_train.passenger_count[i]])
		#x.append([df_train.distance_miles[i],picktimey_train[i],0,df_train.passenger_count[i]])
		y.append(str(df_train.fare_amount[i]))
print(len(y))
for i in range(len(df_test.passenger_count)):
	if picktimeh_train[i]>12:		
		j.append([df_test.distance_miles[i],picktimey_test[i],0,df_test.pickup_longitude[i],df_test.pickup_latitude[i],df_test.dropoff_longitude[i],df_test.dropoff_latitude[i],df_test.passenger_count[i]])
		#j.append([df_test.distance_miles[i],picktimey_test[i],0,df_test.passenger_count[i]])
		k.append(str(df_test.fare_amount[i]))
	else:
		j.append([df_test.distance_miles[i],picktimey_test[i],1,df_test.pickup_longitude[i],df_test.pickup_latitude[i],df_test.dropoff_longitude[i],df_test.dropoff_latitude[i],df_test.passenger_count[i]])
		#j.append([df_test.distance_miles[i],picktimey_test[i],0,df_test.passenger_count[i]])
		k.append(str(df_test.fare_amount[i]))
#print(len(k))
predict=[]
check=[]
sumy=0
rmse=0
# print(y)
clf=svm.SVC()
clf.fit(x,y)
print("finish training")
predict=clf.predict(j)
result=clf.predict(j)
for i in range(len(result)):
    check.append(result[i])

print("finish testing")
for i in range(len(predict)):
	sumy=sumy+(float(predict[i])-float(k[i]))*(float(predict[i])-float(k[i]))
rmse=math.sqrt(sumy/len(predict))
print("RMSE:",rmse)
error=0
for i in range(len(check)):
	if check[i]!=k[i]:
		x.append(j[i])
		y.append(k[i])
		error+=1
#print(error)
dagger=svm.SVC()
dagger.fit(x,y)
daprediction=[]
dasum=0
darmse=0
print("dagger algorithm:")
daprediction=dagger.predict(j)
for i in range(len(daprediction)):
	dasum=dasum+(float(daprediction[i])-float(k[i]))*(float(daprediction[i])-float(k[i]))
darmse=math.sqrt(dasum/len(daprediction))
print("Dagger algorithm ARMSE:",darmse)