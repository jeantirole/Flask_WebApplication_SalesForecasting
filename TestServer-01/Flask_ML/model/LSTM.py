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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 데코레이터 lib 
from functools import wraps, update_wrapper
from datetime import datetime
################################################################## DATA start
#test_FILEPATH = os.path.join(os.getcwd(), 'train.csv')
test_path = './train.csv'

#df = pd.read_csv(test_FILEPATH)
df = pd.read_csv(test_path)

import tensorflow as tf
## prepare train data
df['date'] =  pd.to_datetime(df['date'])
train_df = df[df['store'] == 1]
train_df_pcs = df.groupby(['date', 'store'])['sales'].sum()
train_df_pcs = train_df_pcs.reset_index() 
train_store_1 = train_df_pcs[train_df_pcs['store'] == 1]
train_store_1 = train_store_1.reset_index()
del train_store_1['index']

## normalize train data 
scaler = MinMaxScaler()
reshaped_data = np.reshape(train_store_1['sales'], (-1,1) )

df_scaled = scaler.fit_transform(train_store_1['sales'])
train_store_1['sales'] = df_scaled 


##################################################################

# given prediction period 
period = 360
window_size_param = 20

# raw data 
# train_store_1

train_store_1.date


# period split function 
def train_test_make(data,period):
    
    train_size = int( len(data) ) - period
    test_size = period

    train = data[:train_size]
    test = data[train_size:]

    return train, test


train, test = train_test_make(train_store_1,period)


# make data for LSTM 
def make_dataset(data, label, window_size):
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

train_feature, train_label = make_dataset(train_feature, train_label, window_size_param)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

# make test data 
test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature.shape, test_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, window_size_param)

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


############## 전체 함수화 ################

def execute_model( period , window_size_param  ):

    # raw data 
    # train_store_1

    # period split function 
    def train_test_make(data,period):
        
        train_size = int( len(data) ) - period
        test_size = period

        train = data[:train_size]
        test = data[train_size:]

        return train, test


    train, test = train_test_make(train_store_1,period)


    # make data for LSTM 
    def make_dataset(data, label, window_size):
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

    train_feature, train_label = make_dataset(train_feature, train_label, window_size_param)

    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

    # make test data 
    test_feature = test[feature_cols]
    test_label = test[label_cols]

    test_feature.shape, test_label.shape

    test_feature, test_label = make_dataset(test_feature, test_label, window_size_param)

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
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

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

    return None



execute_model(120, 30 )




