from warnings import simplefilter
from flask import Flask, render_template, redirect, request, url_for,send_file, make_response
from io import BytesIO, StringIO
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
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
# 데코레이터 lib 
from functools import wraps, update_wrapper
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os
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
#df_01 = df.groupby(['store','year','month','week'])['sales'].sum()
df_01 = df.groupby(['date','store','year','month','week'])['sales'].sum()

df_01 = df_01.reset_index()
df_01['week_timeline'] = 0
df_01['weekly_timeline'] = [i for i in range(len(df_01))]

#make plotly data
df_groupby_item = df.groupby(['date', 'store'])['sales'].sum()
df_groupby_item = df_groupby_item.reset_index()



#make data info 
info01 = df.columns.values
start_day = df['date'][0]
last_day = df['date'].iloc[len(df['date'])-1]
category_val_01 = df['store'].unique()
category_val_02 = df['item'].unique()[0:10]


# 
for store in df_01.store.unique():
    bystore = df_01[df_01['store'] == store]
    df_01.loc[df_01['store'] == store, 'week_timeline' ] = [i for i in range(len(bystore))]
#grouping
site =None
blank_group=[]
blank_year=[]
blank_month=[]
blank_week=[]


### data prepare for prediction model #######################################
#df['date'] =  pd.to_datetime(df['date'])
train_df = df[df['store'] == 1]

train_df_pcs = df.groupby(['date', 'store'])['sales'].sum()
train_df_pcs = train_df_pcs.reset_index() 
train_store_1 = train_df_pcs[train_df_pcs['store'] == 1]
train_store_1 = train_store_1.reset_index()
del train_store_1['index']
train_store_1

# ## random seed fixing 
# np.random.seed(70)

# ## normalize train data 
# tr_max = max( train_store_1['sales'])
# tr_min = min( train_store_1['sales'])

# normalized_sales = [ (i - tr_min)/(tr_max - tr_min) for i in train_store_1['sales']]
# train_store_1['normalized_sales'] = normalized_sales
# train_store_1.columns = [ 'date' , 'store'  ,'original_sales' , 'sales']
### data prepare for prediction model ####################################### end

### combined function ### 

def execute_model( data, group, period , window_size_param  ):

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

    return graphJSON2




################################################################## DATA end
def nocache(view):
  @wraps(view)
  def no_cache(*args, **kwargs):
    response = make_response(view(*args, **kwargs))
    response.headers['Last-Modified'] = datetime.now()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response      
  return update_wrapper(no_cache, view)


app = Flask(__name__)


# ### type back function
# def long_load(typeback):
#   time.sleep(5) #just simulating the waiting period
#   return "You typed: %s" % typeback

### selection functions 

@app.route('/groupselection', methods=['POST'])
def groupselection(group=None):
    group = request.form['group']
    blank_group.append(group)
    print(blank_group)
    return redirect(url_for('chart2',group=group))

@app.route('/yearselection', methods=['POST'])
def yearselection(year=None):
    year = request.form['year']
    blank_year.append(year)
    print(blank_year)
    return redirect(url_for('chart2',year=year))

@app.route('/monthselection', methods=['POST'])
def monthselection(month=None):
    month = request.form['month']
    blank_month.append(month)
    print(blank_month)
    return redirect(url_for('chart2',month=month))

# main page 에 변수 보내는 test 용 함수 
@app.route('/train', methods=['GET'])
def train(display = None):
    query = request.args.get('anything')
    #outcome = long_load(query)

    dis2 = execute_model(df, int(blank_group[-1]),int(query), 30 )
    time.sleep(6)
    return redirect(url_for('chart2',display=dis2))





### main chart 

@app.route('/chart2', methods=['POST','GET'])
@app.route('/chart2/<int:group>/<int:year>/<int:month>', methods=['GET','POST'])
@nocache
def chart2(num=None,group=None, year=None, month=None, display=None):
    if len(blank_group) > 0  :
        d_target2 = df_01[ df_01['store'] == int(blank_group[-1])]
    else:
        d_target2 = df_01

    if len(blank_year) > 0  :
        d_target3 = d_target2[ d_target2.year == int(blank_year[-1])] # select 된 year 과 동시에 
    else:
        d_target3 = d_target2

    if len(blank_month) > 0  :
        d_target4 = d_target3[ d_target3.month == int(blank_month[-1])] # select 된 year 과 동시에 
    else:
        d_target4 = d_target3

    #fig = px.bar(df, x="Vegetables", y="Amount", color="City", barmode="stack", width=600, height=300)

    
    fig = px.line(d_target4, x="date", y="sales", color='store', title='Cash Flow',width=1200, height=500)
    fig.update_layout( plot_bgcolor='white')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    header="Cash Flow Management"
    description = """
    date time today 
    """
    groups = df_01.store.unique()
    years = df_01.year.unique()
    months = df_01.month.unique()


    if request.args.get('display'):
        display = request.args.get('display')
        

    return render_template('main2.html', graphJSON=graphJSON, header=header,
                            description=description, num=num,groups=groups, group=group, 
                            years=years, year=year, months= months, month = month, display = display,
                            info01 = info01,
                            start_day = start_day,
                            last_day = last_day,
                            category_val_01 = category_val_01 ,
                            category_val_02 = category_val_02 )



if __name__ == '__main__':
    app.run()


