# DataFrames
import pandas as pd

# Unix time to date
import datetime as dt
from datetime import timedelta

# Binance as Data Source
from binance.client import Client
client=Client()

#Candlesticks chart
import plotly.graph_objects as go

#Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Various
import numpy as np
import warnings
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

# Function to extract the data from binance client and calculate different features of them
def getdata_CHL(symbol='BTCUSDT', freq='1d', lookback='2000'):
    if freq=='1d':
        freqType=' days '
    elif freq=='1h':
        freqType=' hours '
    elif freq=='1w':
        freqType=' weeks'
    elif freq=='1s':
        freqType=' seconds '
    elif freq=='1m':
        freqType=' minutes '
    else: freqType=' days '
    df=pd.DataFrame(client.get_historical_klines(symbol,freq,lookback + freqType + 'ago UTC')) # Contains the 'freq' candlesticks of the last 'lookback' 'freqTypr'
    frame=df.iloc[:,:6] # Extraction of the columns of our interest
    frame.columns=['Date','Open','High','Low','Close','Volume'] # Name of each column according to its values 
    frame[['Open','High','Low','Close','Volume']] = frame[['Open','High','Low','Close','Volume']].astype(float) # Define type as float
    frame.Date = pd.to_datetime(frame.Date, unit = 'ms') # Date from UNIX to normal
    frame['Number of Trades']=df.iloc[:,8]
    frame=frame.set_index('Date') # Set Date as the index
    
    
    # Close return:
    frame['return'] = np.log(frame.Close.pct_change()+1)
    # If we are analysing daily data, we calculate the following SMA and EMA
    if freq=='1d':
        
        # Close SMA, EMA, MACD:
        #SMA
        frame['SMA_7'] = frame.Close.rolling(7).mean()   #SMA expression for 7 periods (week)
        frame['SMA_30'] = frame.Close.rolling(30).mean() #SMA expression for 30 periods (month)
        frame['SMA_90'] = frame.Close.rolling(90).mean() #SMA expression for 90 periods (quarter)
        frame['SMA_365'] = frame.Close.rolling(365).mean() #SMA expression for 365 periods (year)
        #EMA
        frame['EMA_7'] = frame.Close.ewm(span=7).mean()  #EMA expression for 7 periods (week)
        frame['EMA_30'] = frame.Close.ewm(span=30).mean()  #EMA expression for 30 periods (month)
        frame['EMA_90'] = frame.Close.ewm(span=90).mean()  #EMA expression for 90 periods (quarter)
        frame['EMA_365'] = frame.Close.ewm(span=365).mean()  #EMA expression for 365 periods (year)
        #MACD
        frame['MACD'] = frame.Close.ewm(span=12).mean()-frame.Close.ewm(span=26).mean() #Formula de MACD = EMA_12 - EMA_26
        frame['SignalLine'] = frame.MACD.ewm(span=9).mean() #Formula de Signal Line = EMA_9 - EMA_26
        frame['HistogramMACD']=frame.MACD-frame.SignalLine #MACD's histogram is obtained by sustracting MACD and Signal Line
        
        # High return, SMA, EMA, MACD
        frame['returnHigh'] = np.log(frame.High.pct_change()+1)
        #SMA High
        frame['SMA_7High'] = frame.High.rolling(7).mean()   #SMA expression for 7 periods (week)
        frame['SMA_30High'] = frame.High.rolling(30).mean() #SMA expression for 30 periods (month)
        frame['SMA_90High'] = frame.High.rolling(90).mean() #SMA expression for 90 periods (quarter)
        frame['SMA_365High'] = frame.High.rolling(365).mean() #SMA expression for 365 periods (year)
        #EMA High
        frame['EMA_7High'] = frame.High.ewm(span=7).mean()  #EMA expression for 7 periods (week)
        frame['EMA_30High'] = frame.High.ewm(span=30).mean()  #EMA expression for 30 periods (month)
        frame['EMA_90High'] = frame.High.ewm(span=90).mean()  #EMA expression for 90 periods (quarter)
        frame['EMA_365High'] = frame.High.ewm(span=365).mean()  #EMA expression for 365 periods (year)
        #MACD High
        frame['MACDHigh'] = frame.High.ewm(span=12).mean()-frame.High.ewm(span=26).mean() #Formula de MACD = EMA_12 - EMA_26
        frame['SignalLineHigh'] = frame.MACDHigh.ewm(span=9).mean() #Formula de Signal Line = EMA_9 - EMA_26
        frame['HistogramMACDHigh']=frame.MACDHigh-frame.SignalLineHigh #MACD's histogram is obtained by sustracting MACD and Signal Line
        
        # Low return, SMA, EMA, MACD
        frame['returnLow'] = np.log(frame.Low.pct_change()+1)
        #SMA Low
        frame['SMA_7Low'] = frame.Low.rolling(7).mean()   #SMA expression for 7 periods (week)
        frame['SMA_30Low'] = frame.Low.rolling(30).mean() #SMA expression for 30 periods (month)
        frame['SMA_90Low'] = frame.Low.rolling(90).mean() #SMA expression for 90 periods (quarter)
        frame['SMA_365Low'] = frame.Low.rolling(365).mean() #SMA expression for 365 periods (year)
        #EMA Low
        frame['EMA_7Low'] = frame.Low.ewm(span=7).mean() #EMA expression for 7 periods (week)
        frame['EMA_30Low'] = frame.Low.ewm(span=30).mean()  #EMA expression for 30 periods (month)
        frame['EMA_90Low'] = frame.Low.ewm(span=90).mean()  #EMA expression for 90 periods (quarter)
        frame['EMA_365Low'] = frame.Low.ewm(span=365).mean()  #EMA expression for 365 periods (year)
        #MACD Low
        frame['MACDLow'] = frame.Low.ewm(span=12).mean()-frame.Low.ewm(span=26).mean() #Formula de MACD = EMA_12 - EMA_26
        frame['SignalLineLow'] = frame.MACDLow.ewm(span=9).mean() #Formula de Signal Line = EMA_9 - EMA_26
        frame['HistogramMACDLow']=frame.MACDLow-frame.SignalLineLow #MACD's histogram is obtained by sustracting MACD and Signal Line
        
        # This code above looks repetitive, and in fact, it could be optimised to be a for loop, but it wasn't such a pain, just copy, paste, find and replace.
        # On the other hand, subcolums would be in order to easier data manipulation.
     
    # Predictions made today are based on the next day values:
    frame['predOpen']=np.ones(len(frame.index))*np.nan
    frame['predOpen'][:-1]=frame.loc[:,'Open'].values[1:] 
    frame['predHigh']=np.ones(len(frame.index))*np.nan
    frame['predHigh'][:-1]=frame.loc[:,'High'].values[1:] 
    frame['predLow']=np.ones(len(frame.index))*np.nan
    frame['predLow'][:-1]=frame.loc[:,'Low'].values[1:] 
    frame['predClose']=np.ones(len(frame.index))*np.nan
    frame['predClose'][:-1]=frame.loc[:,'Close'].values[1:] 
    
    #In order to have NAN data, uncomment the following statement:
    frame.dropna(inplace = True)
    
    return frame

# Function to sort the data into train and test sets
def train_test_data(symbol='BTCUSDT',freq='1d',lookback='2000'):
    dataset = getdata_CHL(symbol,freq,lookback)
    
    #Close
    Close_features=['Open','High','Low','Close','Volume','Number of Trades',
                    'SMA_7','SMA_30','SMA_90','SMA_365','EMA_7','EMA_30','EMA_90',
                    'EMA_365','MACD','SignalLine', 'HistogramMACD','predClose'] # the last one is the label
    datasetClose = dataset.loc[:,Close_features]
    
    #Split the data into training and test sets
    train_datasetClose = datasetClose.sample(frac=0.8, random_state=0)
    test_datasetClose = datasetClose.drop(train_datasetClose.index)
    
    #Split features from labels
    train_featuresClose = train_datasetClose.copy()
    test_featuresClose = test_datasetClose.copy()

    train_labelsClose = train_featuresClose.pop('predClose')
    test_labelsClose = test_featuresClose.pop('predClose')
    
    #The Normalization layer
    normalizerClose = tf.keras.layers.Normalization(axis=-1)
    normalizerClose.adapt(np.array(train_featuresClose))
    
    #High
    High_features=['Open','High','Low','Close','Volume','Number of Trades',
                    'SMA_7High','SMA_30High','SMA_90High','SMA_365High','EMA_7High','EMA_30High','EMA_90High',
                    'EMA_365High','MACDHigh','SignalLineHigh', 'HistogramMACDHigh','predHigh'] # the last one is the label
    datasetHigh = dataset.loc[:,High_features]
    
    #Split the data into training and test sets
    train_datasetHigh = datasetHigh.sample(frac=0.8, random_state=0)
    test_datasetHigh = datasetHigh.drop(train_datasetHigh.index)
    
    #Split features from labels
    train_featuresHigh = train_datasetHigh.copy()
    test_featuresHigh = test_datasetHigh.copy()

    train_labelsHigh = train_featuresHigh.pop('predHigh')
    test_labelsHigh = test_featuresHigh.pop('predHigh')
    
    #The Normalization layer
    normalizerHigh = tf.keras.layers.Normalization(axis=-1)
    normalizerHigh.adapt(np.array(train_featuresHigh))
    
    #Low
    Low_features=['Open','High','Low','Close','Volume','Number of Trades',
                    'SMA_7Low','SMA_30Low','SMA_90Low','SMA_365Low','EMA_7Low','EMA_30Low','EMA_90Low',
                    'EMA_365Low','MACDLow','SignalLineLow', 'HistogramMACDLow','predLow'] # the last one is the label
    datasetLow = dataset.loc[:,Low_features]
    
    #Split the data into training and test sets
    train_datasetLow = datasetLow.sample(frac=0.8, random_state=0)
    test_datasetLow = datasetLow.drop(train_datasetLow.index)
    
    #Split features from labels
    train_featuresLow = train_datasetLow.copy()
    test_featuresLow = test_datasetLow.copy()

    train_labelsLow = train_featuresLow.pop('predLow')
    test_labelsLow = test_featuresLow.pop('predLow')
    
    #The Normalization layer
    normalizerLow = tf.keras.layers.Normalization(axis=-1)
    normalizerLow.adapt(np.array(train_featuresLow))
    
    return_array=[[train_featuresClose, test_featuresClose, train_labelsClose, test_labelsClose, normalizerClose],
                  [train_featuresHigh, test_featuresHigh, train_labelsHigh, test_labelsHigh, normalizerHigh], 
                  [train_featuresLow, test_featuresLow, train_labelsLow, test_labelsLow, normalizerLow]]
    return return_array

#Function to build a compile a tensorflow model
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# Function to build and compile multiple models 
def bc_multiple_model(norm=[0]):
    if not norm[0]:
        print(norm[0])
        warnings.warn('No model created')
        pass
    else:
        n=len(norm)
        models=[0]*n
        for i in range(n):
            models[i]=build_and_compile_model(norm[i])
        return models

# Function to create, fit, evaluate or use our model for BTC prediction
def model_alpha(data,new=False,fit=False,epochs=100,evaluate=False):
    if new:
        models=bc_multiple_model([data[i][-1] for i in range(len(data))]) 
    else:
        #reference: ttps://www.tensorflow.org/guide/keras/save_and_serialize
        models=[keras.models.load_model('model_alpha_' + str(i)) for i in range(3)]
    if fit:
        history=[0]*3
        for i in range(len(models)):
            train_features = data[i][0]
            train_labels = data[i][2]
            history[i] = models[i].fit(
                train_features,
                train_labels,
                validation_split=0.2,
                verbose=0, epochs=epochs)
            models[i].save('model_alpha_'+str(i))
        return history #List containing the history of each of the 3 models 
    
    elif evaluate:
        evaluation=[0]*3
        for i in range(len(models)):
            test_features = data[i][1]
            test_labels = data[i][3]
            evaluation[i] = models[i].evaluate(test_features, test_labels, verbose=0)
        return evaluation #List containing the evaluation of each of the 3 models 
    
    else:
        pred=[0]*3
        for i in range(len(models)):
            test_features = data[i][1]
            pred[i]=models[i](test_features)
        predAux=[[float(pred[i][j]) for j in range(len(pred[0]))] for i in range(len(pred))] #We transform tf.tensor into float
        completePred=[test_features.loc[:,'Close'].copy().values,predAux[0],predAux[1],predAux[2]]
        df=pd.DataFrame(completePred) 
        df=df.transpose()
        columnsNames=['predOpen','predHigh','predLow','predClose']
        df.columns=columnsNames
        df = df.astype(float)
        df['Date'] = test_features.index.copy() 
        df=df.set_index('Date')
        df.index=df.index + timedelta(days=1)
        return df #Prediction DataFrame of the OHCL values

# Function to plot the history of a model
def plot_loss(history):
  fig = plt.figure()
  fig.patch.set_facecolor('xkcd:mint green')
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 2000])  # Limits may be changed
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
 
# Function to plot the history of the 3 models in model_alpha
def multiple_plot_loss(history):
    n=len(history)
    title=['Close fit','High fit','Low fit']
    for i in range(n):
        plot_loss(history[i])
        plt.title(title[i])
    plt.show

# Function to sort the OHCL prediction dataframe logically
def logical_pred(df):
    high=df.predHigh.values
    low=df.predLow.values
    close=df.predClose.values
    matrix=[list(i) for i in zip(*[high,low,close])]
    for i in range(len(matrix)):
        matrix[i].sort(reverse=True)
    matrix=[list(i) for i in zip(*matrix)]
    matrix

    df_sorted=pd.DataFrame([df.predOpen.values,matrix[0],matrix[2],matrix[1]]).T
    df_sorted['Date'] = df.index.copy() 
    df_sorted=df_sorted.set_index('Date')
    df_sorted.columns=df.columns

    return df_sorted

# Function to plot the candlestick chart of the OHCL predictions
def plot_pred_candlesticks(pred_df):
    df=logical_pred(pred_df)
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df.predOpen,
                    high=df.predHigh,
                    low=df.predLow,
                    close=df.predClose)])
    fig.update_layout(
        height=800,
        title="Predictions Aug 2017-Today",
        yaxis_title='BTC price',
        xaxis_title='Date'
    )
    fig
    return fig

# Function that shows a general view of the achieved model and predictions
def main():
    data=train_test_data()
    history=model_alpha(data,new=True,fit=True,epochs=600)
    multiple_plot_loss(history)
    plot_pred_candlesticks(model_alpha(data))
    pass
