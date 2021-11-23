from typing import Dict
from warnings import simplefilter
from flask import Flask, render_template, redirect, request, url_for,send_file, make_response
import os
import pandas as pd
import io
from io import BytesIO, StringIO
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
import tensorflow as tf
# 데코레이터 lib 
from functools import wraps, update_wrapper
from datetime import datetime
################################################################## DATA start
#test_FILEPATH = os.path.join(os.getcwd(), 'train.csv')
test_path = 'C:\\Eric\\Projects\\02-Webb_h_signal_done2\\train.csv'

#df = pd.read_csv(test_FILEPATH)
df = pd.read_csv(test_path)

## prepare train data
df['date'] =  pd.to_datetime(df['date'])
train_df = df[df['store'] == 1]
train_df_pcs = df.groupby(['date', 'store'])['sales'].sum()
train_df_pcs = train_df_pcs.reset_index() 
train_store_1 = train_df_pcs[train_df_pcs['store'] == 1]
train_store_1 = train_store_1.reset_index()
del train_store_1['index']
train_store_1

## check info of data_frame 
info01 = df.columns
start_day = df['date'][0]
last_day = df['date'][len(df)-1]
category_val_01 = df['store'].unique()
category_val_02 = df['item'].unique()

## random seed fixing 
np.random.seed(70)

## normalize train data 
tr_max = max( train_store_1['sales'])
tr_min = min( train_store_1['sales'])

normalized_sales = [ (i - tr_min)/(tr_max - tr_min) for i in train_store_1['sales']]
train_store_1['normalized_sales'] = normalized_sales
train_store_1.columns = [ 'date' , 'store'  ,'original_sales' , 'sales']

#make date split 
df['date'] =  pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week']= df['date'].dt.week
df_01 = df.groupby(['store','year','month','week'])['sales'].sum()
df_01 = df_01.reset_index()
df_01['week_timeline'] = 0
df_01['weekly_timeline'] = [i for i in range(len(df_01))]
#fig = px.bar(df, x="Vegetables", y="Amount", color="City", barmode="stack", width=600, height=300)


#make df groupby only by item 

df_groupby_item = df.groupby(['date', 'store'])['sales'].sum()
df_groupby_item = df_groupby_item.reset_index()
df_groupby_item
fig = px.line(df_groupby_item, x="date", y="sales", color='store', title='Cash Flow',width=1200, height=500)
fig.update_layout( plot_bgcolor='white')
fig.show()
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
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model_path = 'model_LSTM'
    filename = os.path.join(model_path, 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # fit 되는 동안의 시간값 html 에 출력 
    history = model.fit(x_train, y_train, 
                                        epochs=3, 
                                        batch_size=16,
                                        validation_data=(x_valid, y_valid), 
                                        callbacks=[early_stop, checkpoint])

    model.load_weights(filename)
    pred = model.predict(test_feature)

    # graph 

    plt.figure(figsize=(12, 9))
    plt.plot(test_label, label = 'actual')
    plt.plot(pred, label = 'prediction')
    #plt.legend()
    #plt.show()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    img.seek(0)

    return img



img_file = execute_model(120, 30 )

from PIL import Image
image = Image.open(img_file)
image.show()


#################################

# period was 360

new_pred = tf.reshape(pred, (len(pred)))
new_test_label = tf.reshape(test_label, (len(test_label)))
prediction_df = pd.DataFrame({
    'dates' : [i for i in range(len(pred))],
    'test_label' : new_test_label,
    'pred' : new_pred
})

prediction_df

figprediction = px.line(
	prediction_df,
	x = 'dates',
	y = 'test_label',
    title='Cash Flow Prediction',
    width=1200, height=500)

figprediction.update_layout( 
                            plot_bgcolor='white',
                            )

figprediction.add_trace(go.Scatter(x=prediction_df['dates'], y=prediction_df['pred'],
                    mode='lines',
                    name='prediction'))


figprediction.show()


figprediction.update_layout(paper_bgcolor="rgb(0,0,0,0)", plot_bgcolor='white')

#graphJSON2 = json.dumps(figprediction, cls=plotly.utils.PlotlyJSONEncoder)


#figprediction.show()

# json dumps 는 메모리상 
#json.dumps('./pltolychart11.json', cls=plotly.utils.PlotlyJSONEncoder)

# json dump 는 파일출력 

pred
test_label
train_store_1



#######################################################################

df
group = 3
blank =[3]
blank[-1]

def groupchoice( data, group ):

    train_df_pcs = data.groupby(['date', 'store'])['sales'].sum()
    train_df_pcs = train_df_pcs.reset_index() 
    train_df_pcs = train_df_pcs[train_df_pcs['store'] == group]
    train_df_pcs = train_df_pcs.reset_index()
    del train_df_pcs['index']

    return train_df_pcs

grouped_data = groupchoice(df, group)
grouped_data
# raw data 
# train_store_1

# period split function 
def train_test_make(data,period):
    
    train_size = int( len(data) ) - period
    test_size = period

    train = data[:train_size]
    test = data[train_size:]

    return train, test


train, test = train_test_make(grouped_data,period)


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
                                    epochs=3, 
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
#plt.show()
#img = BytesIO()
plt.savefig('./static/predictionchart.png', format='png', dpi=200)
#img.seek(0)



# to plotly chart 
new_pred = tf.reshape(pred, (len(pred)))
new_test_label = tf.reshape(test_label, (len(test_label)))
prediction_df = pd.DataFrame({
'dates' : [i for i in range(len(pred))],
'test_label' : new_test_label,
'pred' : new_pred
})

figprediction = px.line(
prediction_df,
x = 'dates',
y = 'test_label',
title='Cash Flow',
width=1200, height=500)

figprediction.update_layout( plot_bgcolor='white')

figprediction.add_trace(go.Scatter(x=prediction_df['dates'], y=prediction_df['pred'],
                mode='lines',
                name='prediction'))


graphJSON2 = json.dumps(figprediction, cls=plotly.utils.PlotlyJSONEncoder)



#######################################
group = 2
def groupchoice( data, group ):

    train_df_pcs = data.groupby(['date', 'store'])['sales'].sum()
    train_df_pcs = train_df_pcs.reset_index() 
    train_df_pcs = train_df_pcs[train_df_pcs['store'] == group]
    train_df_pcs = train_df_pcs.reset_index()
    del train_df_pcs['index']

    return train_df_pcs

grouped_data = groupchoice(df, group)

def normalzie_data(data):
# normalzie 
    np.random.seed(70)

    ## normalize train data 
    tr_max = max( data['sales'])
    tr_min = min( data['sales'])

    normalized_sales = [ (i - tr_min)/(tr_max - tr_min) for i in data['sales']]
    data['normalized_sales'] = normalized_sales
    data.columns = [ 'date' , 'store'  ,'original_sales' , 'sales']
    return data

grouped_data = normalzie_data(grouped_data)

# period split function 
def train_test_make(data,period):
    
    train_size = int( len(data) ) - period
    test_size = period

    train = data[:train_size]
    test = data[train_size:]

    return train, test


train, test = train_test_make(grouped_data,period)


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
                                    epochs=3, 
                                    batch_size=16,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(test_feature)