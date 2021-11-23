import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os
from sklearn.model_selection import train_test_split

def preprocessing(df):
    df['date'] = pd.to_datetime(df['date'])
    
    train_df = df.copy()
    train_df = train_df.groupby(['date', 'store'])['sales'].sum()
    train_df = train_df.reset_index() 

    ## normalize train data 
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(train_df['sales'])
    train_df['sales'] = df_scaled 

    return train_df

def make_LSTM_data(df, feature, target, predict_period=30, window_size = 20):
    # train, test 분리 
    train_size = int( len(df) ) - predict_period
    train = df[:train_size]
    test = df[train_size:]
    
    # train LSTM 데이터로 변환
    train_feature, train_label = train[feature], train[target]

    train_feature_list = []
    train_label_list = []
    for i in range(len(train) - window_size):
        train_feature_list.append(np.array(train_feature.iloc[i:i+window_size]))
        train_label_list.append(np.array(train_label.iloc[i+window_size]))
    
    train_list, label_list = np.array(train_feature_list), np.array(train_label_list)
    X_train, X_valid, y_train, y_valid = train_test_split(train_list, label_list, test_size=0.2)
    
    # test LSTM 데이터로 변환
    test_feature, test_label = test[feature], test[target]
    
    test_feature_list = []
    test_label_list = []
    for i in range(len(test) - window_size):
        test_feature_list.append(np.array(test_feature.iloc[i:i+window_size]))
        test_label_list.append(np.array(test_label.iloc[i+window_size]))
    X_test, y_test = np.array(test_feature_list), np.array(test_label_list)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def LSTM_modeling(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # model create
    model = Sequential()
    model.add(LSTM(16, 
            input_shape=(X_train.shape[1], X_train.shape[2]), 
            activation='relu', 
            return_sequences=False)
            )

    model.add(Dense(1))

    # model compile 
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    model_path = 'model_LSTM'
    filename = os.path.join(model_path, 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # model fit
    model.fit(X_train, y_train, 
            epochs=200, 
            batch_size=16,
            validation_data=(X_valid, y_valid), 
            callbacks=[early_stop, checkpoint])

    model.load_weights(filename)
    pred = model.predict(X_test)

    result = pd.DataFrame()
    result['original'] = y_test
    result['predict'] = pred

    return result