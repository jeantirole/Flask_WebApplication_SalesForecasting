import pandas as pd

#test_FILEPATH = os.path.join(os.getcwd(), 'train.csv')
test_path = './train.csv'

#df = pd.read_csv(test_FILEPATH)
df = pd.read_csv(test_path)

#make data info 
info01 = df.columns.values
start_day = df['date'][0]
last_day = df['date'].iloc[len(df['date'])-1]
category_val_01 = df['store'].unique()
category_val_02 = df['item'].unique()[0:10]


#make plotly data
df_groupby_item = df.groupby(['date', 'store'])['sales'].sum()
df_groupby_item = df_groupby_item.reset_index()

#make date split 
df_groupby_item['date'] =  pd.to_datetime(df_groupby_item['date'])
df_groupby_item['year'] = df_groupby_item['date'].dt.year
df_groupby_item['month'] = df_groupby_item['date'].dt.month
df_groupby_item['week']= df_groupby_item['date'].dt.week


#make plotly data
df_groupby_item = df.groupby(['date', 'store'])['sales'].sum()
df_groupby_item = df_groupby_item.reset_index()
for store in df_01.store.unique():
    bystore = df_01[df_01['store'] == store]
    df_01.loc[df_01['store'] == store, 'week_timeline' ] = [i for i in range(len(bystore))]
