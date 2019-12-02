# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import json
import pandas as pd
import datetime as dt
import numpy as np
import quandl
import matplotlib.pyplot as plt
import keras
import tensorflow
import time
import math
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
import h5py

from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

q = quandl.get("ECB/EURCHF", key = "3h5NKbQ9nPGRU9gtkyR9")

q.columns = ["EURCHF"]
df_rate = q
df_rate.head()

ax = df_rate[df_rate.index >= pd.Timestamp("2008-09-06")].plot()
ax.fill_betweenx(ax.get_ylim(),pd.Timestamp("2011-09-06"),pd.Timestamp("2015-01-15"),alpha = 0.1,zorder = -1)

Quandl_code = "https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.json?api_key=3h5NKbQ9nPGRU9gtkyR9"
q_req = requests.get(url = Quandl_code)





q_json = q_req.json()
q_data = q_json["dataset"]["data"]
dates = []
values = []
q_len = len(q_data)
column_len = len(q_data[0])
column = column_len - 1
columns = q_json["dataset"]["column_names"]
print(columns)

df_rates = pd.DataFrame.from_records(q_data, columns = columns)
print(df_rates.head())
df_rates["Date"] = pd.to_datetime(df_rates["Date"])
df_rates.set_index("Date",inplace  = True)

print(df_rates.head())
df_rates.plot()

### only 5Yr
Value = "5 YR"
df_rates = pd.DataFrame.from_records(q_data,columns = columns)
df_data_1 = (df_rates[["Date",Value]]).sort_index(ascending = False)
print(df_data_1.head())
df_data_1.shape

df_data_1_plot = df_data_1.iloc[:,1:2].values
plt.plot(df_data_1_plot,color = "red", label = Value)
plt.title("Historical Data")
plt.xlabel("Index")
plt.ylabel("Yields")
plt.legend()
plt.show()


batch_size = 64
epochs = 120
timesteps = 30
length = len(df_data_1)
length *= 1-0.1


def get_train_length(dataset,batch_size, test_percent):
    length = len(dataset)
    length *= 1-test_percent
    print(length)
    train_length_values = []
    for x in range(int(length) - 100, int(length)):
        modulo = x % batch_size
        if (modulo == 0):
            train_length_values.append(x)
            print(x)
    return (max(train_length_values))
length = get_train_length(df_data_1,batch_size,0.1)
print(length)

upper_train = length + timesteps * 2
df_data_1_train = df_data_1[0:upper_train]
training_set = df_data_1_train.iloc[:,1:2].values
training_set[-10:-1]




sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(np.float64(training_set))
training_set_scaled.shape

X_train = []
y_train = []
print(length + timesteps)
for i in range(timesteps, length + timesteps):
    X_train.append(training_set_scaled[i - timesteps: i, 0])
    y_train.append(training_set_scaled[i:i+timesteps,0])
print(len(X_train))
print(len(y_train))
print(np.array(X_train).shape)
print(np.array(y_train).shape)

### Reshapping
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1],1))


inputs_1_mae = Input(batch_shape =(batch_size, timesteps,1))
lstm_1_mae = LSTM(10,stateful = True, return_sequences = True)(inputs_1_mae)
lstm_2_mae = LSTM(10, stateful = True, return_sequences = True)(lstm_1_mae)
output_1_mae = Dense(units = 1)(lstm_2_mae)

regressor_mae = Model(inputs_1_mae, outputs= output_1_mae)
regressor_mae.compile(optimizer = "adam", loss = "mae")
regressor_mae.summary()

for i in range(epochs):
    regressor_mae.fit(X_train,y_train,shuffle = False, epochs = 1,batch_size = batch_size \
                     ,verbose = 0)
    regressor_mae.reset_states()

def get_test_length(data_set,batch_size):
    test_length_values = []
    for x in range(len(data_set) - 200, len(data_set) - timesteps*2):
        modulo = (x - upper_train)%batch_size
        if (modulo == 0):
            test_length_values.append(x)
            print(x)
    return max(test_length_values)
test_length = get_test_length(df_data_1,batch_size)
upper_test = test_length + timesteps * 2
testset_length = test_length - upper_train
print (upper_train, upper_test, len(df_data_1))
df_data_1_test = df_data_1[upper_train:upper_test]
test_set = df_data_1_test.iloc[:,1:2].values

scaled_real_bcg_values_test = sc.fit_transform(np.float64(test_set))
X_test = []
for i in range(timesteps, testset_length + timesteps):
    X_test.append(scaled_real_bcg_values_test[i - timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_bcg_values_test_mae = regressor_mae.predict(X_test, \
                                                      batch_size = batch_size)
regressor_mae.reset_states()
print(predicted_bcg_values_test_mae.shape)
predicted_bcg_values_test_mae = np.reshape(predicted_bcg_values_test_mae, \
                                          (predicted_bcg_values_test_mae.shape[0],\
                                          predicted_bcg_values_test_mae.shape[1]))
predicted_bcg_values_test_mae = sc.inverse_transform(predicted_bcg_values_test_mae)
y_test = []
for j in range(0,testset_length - timesteps):
    y_test = np.append(y_test, predicted_bcg_values_test_mae[j, timesteps - 1 ])
y_test = np.reshape(y_test,(y_test.shape[0],1))
print(y_test.shape)

plt.rcParams["figure.figsize"] = (20,20)
plt.plot(test_set[timesteps:len(y_test)], color = "red", label = "Prices")
plt.plot(y_test[0:len(y_test) - timesteps], color = "blue", label = "Predicted")

plt.title("Prediction - MAE")
plt.xlabel("Time")
plt.ylabel("Prices")
plt.legend()
plt.show()
err_vector = ((test_set[timesteps:len(y_test)] - y_test[0:len(y_test) - timesteps]))
plt.hist(err_vector)