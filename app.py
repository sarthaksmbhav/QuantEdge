import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
start = '2010-01-01'
end = '2019-12-31'
st.title('QuantEdge')
user_input=st.text_input('Enter Stock in box','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2010-2019')
st.write(df.describe())