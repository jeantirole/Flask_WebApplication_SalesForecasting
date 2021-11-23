from flask import Blueprint, render_template, request
import os
import pandas as pd
import plotly
import plotly.express as px
import json

from tensorflow.python.eager import def_function
from Flask_ML.model.LSTM_re import preprocessing, make_LSTM_data, LSTM_modeling

test_FILEPATH = os.path.join(os.getcwd(), 'Flask_ML', 'test.csv')
test_df = pd.read_csv(test_FILEPATH)
train_FILEPATH = os.path.join(os.getcwd(), 'Flask_ML', 'train.csv')
train_df = pd.read_csv(train_FILEPATH)
group_name = list(train_df['store'].unique())

bp = Blueprint('predict', __name__)


@bp.route('/LSTM', methods = ['GET'])
def LSTM():
  startdate = request.args.get('startdate')
  enddate = request.args.get('enddate')
  groupname = request.args.get('grouping')
  date_unit = request.args.get('date_unit')

  df = train_df.copy()
  print(groupname != "All")
  print(groupname == "All")
  print("startdate:",startdate,"enddate:",enddate,"groupname:",groupname,"date_unit:",date_unit)
  if startdate != "":
    df = df[ df['date'] >= startdate ]
  if enddate != "":
    df = df[ df['date'] <= enddate ]   
  if groupname != "All":
    df = df[ df['store'] == int(groupname) ]
  
  pre_df = preprocessing(df)
  feature = 'sales'
  target = 'sales'
  X_train, y_train, X_valid, y_valid, X_test, y_test = make_LSTM_data(pre_df, feature, target)
  
  result_df = LSTM_modeling(X_train, y_train, X_valid, y_valid, X_test, y_test)
  

  if groupname != "All":
    fig = px.line(df, x='date', y='sales')
  else:
    fig = px.line(df, x='date', y='sales', color='store')


  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  
  return render_template("index.html", graphJSON=graphJSON, group_name=group_name)
