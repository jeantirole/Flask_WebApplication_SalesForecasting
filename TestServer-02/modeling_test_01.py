from typing import Dict
from warnings import simplefilter
from flask import Flask, render_template, redirect, request, url_for,send_file, make_response
import os
import pandas as pd
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.offline as pyo
from flask import Markup
import plotly.graph_objects as go
import json 
import plotly
import plotly.express as px


# 데코레이터 lib 
from functools import wraps, update_wrapper
from datetime import datetime
################################################################## DATA start
#test_FILEPATH = os.path.join(os.getcwd(), 'train.csv')
test_path = './train.csv'

#df = pd.read_csv(test_FILEPATH)
df = pd.read_csv(test_path)

df = df[(df['store'] == 1) | (df['store'] == 2) | (df['store'] == 3) | (df['store'] == 4) | (df['store'] == 5) ]

#make date split 
df['date'] =  pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week']= df['date'].dt.week
df_01 = df.groupby(['store','year','month','week'])['sales'].sum()
df_01 = df_01.reset_index()
df_01['week_timeline'] = 0

df_01['weekly_timeline'] = [i for i in range(len(df_01))]

for store in df_01.store.unique():
    bystore = df_01[df_01['store'] == store]
    df_01.loc[df_01['store'] == store, 'week_timeline' ] = [i for i in range(len(bystore))]

import tensorflow as tf
## prepare train data 
train_df = df[df['store'] == 1]
train_df_pcs = df.groupby(['date', 'store'])['sales'].sum()
train_df_pcs = train_df_pcs.reset_index() 
train_store_1 = train_df_pcs[train_df_pcs['store'] == 1]
train_store_1 = train_store_1.reset_index()
del train_store_1['index']

fig = px.line(train_store_1, x="date", y="sales", title='Cash Flow',width=800, height=500)
fig.show()


# def min_max_normalize(lst):
#     normalized = []
    
#     for value in lst:
#         normalized_num = (value - min(lst)) / (max(lst) - min(lst))
#         normalized.append(normalized_num)
    
#     return normalized

# normalized_sales = min_max_normalize(train_store_1['sales'])

# plt.plot(a1)
# plt.show()

TRAIN_SIZE = 0.9
WINDOW_SIZE = 30 # 최소주기 : 7 or 최대주기 : 365 or 한달주기 :30
# steps =  len(train_store_1) * TRAIN_SIZE
# int(steps)

# train = normalized_sales[:int(steps)]
# test = normalized_sales[int(steps):]
# len(train) #1460
# len(test) #366

##
train_store_1


##
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)
##
def univariate_data_multistep(dataset, start_index, end_index, history_size, target_size, step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i:i+target_size])
    return np.array(data), np.array(labels)


##
TRAIN_SPLIT = int( len(train_store_1) * 0.9) +1
tf.random.set_seed(13)

uni_data = train_store_1['sales']
uni_data.index = train_store_1['date']

uni_data.plot(subplots=True)
plt.show()

uni_data = uni_data.values
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data - uni_train_mean) / uni_train_std

print(uni_data)

univariate_past_history = 60
univariate_future_target = 14

#uni
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

#multi
x_train_uni, y_train_uni = univariate_data_multistep(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target,7)
x_val_uni, y_val_uni = univariate_data_multistep(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target,7)






len(x_train_uni) #1614
len(y_train_uni) #1614
len(x_val_uni) #152
len(y_val_uni) #152

print('Single window of past history')
print(x_train_uni[0])

print('Target to predict')
print(y_train_uni[0])





def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.axis('auto')
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example').show()

def baseline(history):
    return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Sample Example').show()


BATCH_SIZE = 14
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()




simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 100

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

for x, y in val_univariate.take(1):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()


simple_lstm_model.predict()



### ARIMA ###
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

train_store_1
len_train = int( len(train_store_1) * 0.8)
len_test = len(train_store_1) - len_train

train_store_1.index = train_store_1.date
df_train_arima = train_store_1[:len_train]['sales']
df_test_arima = train_store_1[len_train:]['sales']

df_train_arima.columns = ['sales']



from pmdarima.arima import auto_arima
model = auto_arima(df_train_arima, start_p=0, start_q=0)
model.summary()


from statsmodels.tsa.arima_model import ARIMA
df_train_arima

model_a = ARIMA(df_train_arima, order=(5,1,2))
model_fit = model_a.fit(trend='nc',full_output=True)
fore = model_fit.forecast(steps=366)
print(fore)

# put the prediction on the dataframe 
#df_test_arima = df_test_arima.reset_index()
df_test_arima['prediction'] = 0
df_test_arima['prediction'][0:366] = fore[0]
df_test_arima

# from matplotlib import pyplot as plt
# plt.plot(fore[0], color='blue')
# plt.plot(df_test_arima['sales'][0:7], color='red')
# plt.show()

## 
fig = px.line(df_test_arima[0:366], x="date", y="sales", title='Cash Flow',width=1200, height=700)
fig.add_trace(go.Scatter(y=df_test_arima['prediction'][0:366], x = df_test_arima['date'][0:366], mode="lines"))
fig.show()

# Arima result 
'''
아 좋은데.. 예측이 제대로 안되네 흠.. 
'''

### one more try lstm 


len(train_store_1) # 1826 

train_size = int( len(train_store_1) * 0.8) 
test_size = 1826 - train_size

train = train_store_1[:train_size]
test = train_store_1[train_size:]

len(train) #1460
len(test) #366

train = train_store_1
WINDOW_SIZE = 20

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


from sklearn.model_selection import train_test_split

#train
feature_cols = ['sales']
label_cols = ['sales']

train_feature = train[feature_cols]
train_label = train[label_cols]

train_feature, train_label = make_dataset(train_feature, train_label, 20)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
x_train.shape, x_valid.shape

#test
test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature.shape, test_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )

model.add(Dense(1))


import os

model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                                    epochs=200, 
                                    batch_size=16,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(test_feature)


plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()

# Success ! 
test_disp = test[20:].copy() # 여기가 진짜 정답구간.  window 20 은
test_disp['pred'] = pred
test_disp

fig = px.line(test_disp, x="date", y="sales", title='Cash Flow',width=1200, height=700)
fig.add_trace(go.Scatter(y=test_disp['pred'], x = test_disp['date'], mode="lines",showlegend=False))
fig.show()




###############################################################################
# train, test 나눠놓고 시작. => 시뮬중에는.. test 로 쪼개는 기간과 period 로 던지는 기간이 일치하는게 좋다. 
# 기간선택 : period 변수 => window size => date create 

train_dummy = train_store_1.copy()
train_dummy = train_dummy.reset_index()
del train_dummy['index']
last_date = train_dummy['date'][ len(train_dummy)-1]
last_date

import datetime
predicted_date = last_date + datetime.timedelta(+30)
predicted_date

period = 90

def select_period(last_date, period):
    period = int(period)
    
    stack_dates = []
    for i in range(period):
        day_created = last_date + datetime.timedelta(+(i+1))
        stack_dates.append(day_created)
    
    return stack_dates

A = select_period( last_date, period)


train_dummy_01 = pd.DataFrame({
    'date': A ,
    'store': [1 for i in range(period) ], 
    'sales': [train_dummy['sales'][-period:].sum() / period for i in range(period) ],
} )

print(train_dummy_01)

train_dummy_predicted = pd.concat( [ train_dummy, train_dummy_01  ], axis=0 )

## 예측기간 추가된 그래프 출력 
fig = px.line(train_dummy_predicted[train_dummy_predicted['date'] > '2017-10-01'], x="date", y="sales", title='Cash Flow',width=1200, height=700)
#fig.add_trace(go.Scatter(y=test_disp['pred'][0:period], x = test_disp['date'][0:period], mode="lines",showlegend=False))
#여기서 예측 모델 사용해서 pred 뽑아서 넣어줘야함 

# 기간이 주어졌을 때, period, 모델로 예측하기, 



##################################################################
from sklearn.model_selection import train_test_split

# prediction period 
period = 30

# raw data 
train_store_1

# period split function 
def train_test_split(data,period):
    
    train_size = int( len(data) ) - period
    test_size = period

    train = data[:train_size]
    test = data[train_size:]

    return train, test


train, test = train_test_split(train_store_1,30)


# make data for LSTM 
def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

# make train data 
feature_cols = ['sales'] # 나중에 y 값으로 고를 수 있는 옵션필요 
label_cols = ['sales'] # 

train_feature = train[feature_cols]
train_label = train[label_cols]

train_feature, train_label = make_dataset(train_feature, train_label, 20)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

# make test data 
test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature.shape, test_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, 20)

# Modeling 
# LSTM layout 

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os

# class 
model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )

model.add(Dense(1))

# compile 
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model_LSTM'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# fit 되는 동안의 시간값 html 에 출력 
history = model.fit(x_train, y_train, 
                                    epochs=200, 
                                    batch_size=16,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(test_feature)

# graph 

plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()