#importing the relevant libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import time
from pandas import DataFrame
from keras.utils import plot_model
from IPython.display import Image
from IPython.core.display import HTML

#Reading the csv  file from local
data_path = "../quantist/data/pool/"
crl = pd.read_csv(data_path+'crl.csv')

#Computing the price ratio as feature
crl['close_ratio'] = (crl['Close']/crl['Close'].shift(1) -1 ).fillna(0)

#Computing the Volume ratio as feature
crl['Vol_ratio'] = (crl['Volume']/crl['Volume'].shift(1) -1 ).fillna(0)

crl['crl_future_1'] = crl['Close'].shift(-1).fillna(0)

crl.head(),


#Reading the csv file from local
nbi = pd.read_csv(data_path+'nbi.csv')

#Renaming the columns to avoid clash while joining crl and nbi later on
nbi.columns = ['Date','nbi_open','nbi_high','nbi_low','nbi_close','nbi_adjclose','nbi_volume']

#Computing the price ratio difference as feature
nbi['nbi_close_ratio'] = (nbi['nbi_close']/nbi['nbi_close'].shift(1) -1 ).fillna(0)



nbi.head()

df = pd.concat([crl,nbi],axis=1)
df.head()

df.sort_index(ascending = True,inplace = True)

#normalizing closing prices and storing it in a new column
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))

df['close_norm'] = scaler1.fit_transform(df['Close'])
df['nbi_close_norm'] = scaler2.fit_transform(df['nbi_close'])
df['crl_future_norm_1'] = df['close_norm'].shift(-1)

#plottting the arrays
plt.plot(df['close_norm'],'-r',label = 'CRL Closing Price')
plt.plot(df['nbi_close_norm'],'-b',label = 'NBI Closing Price')
plt.title("CRL NYSE & NBI NASDAQ closing stock price from 01-04-2010 to 07-11-2010")
plt.xlabel("Time in Days")
plt.ylabel("Closing Price of stock index in USD")
plt.legend(loc='upper left')
plt.show()

np.corrcoef(df.Close,df.crl_future_1)
np.corrcoef(df.nbi_close,df.crl_future_1)
np.corrcoef(df.nbi_close,df.Close)

#taking 98% of the data points as train. This number can change.
train_size = int(0.98*len(df))

#segregating the inputs and ouput on the test and train data
trainX = df.loc[1:train_size,['close_norm','nbi_close_norm','Vol_ratio','nbi_close_ratio','close_ratio']]
trainY = df.loc[1:train_size,'crl_future_norm_1']
testX = df.loc[1856:1891,['close_norm','nbi_close_norm','Vol_ratio','nbi_close_ratio','close_ratio']]
testY = df.loc[1856:1891,'crl_future_norm_1']

#The inputs needed to be reshaped in the format of  a 3d Tensor with dimesnions = [batchsize,timesteps,features]
trainX = np.reshape(np.array(trainX),(trainX.shape[0],1,trainX.shape[1]))
testX= np.reshape(np.array(testX),(testX.shape[0],1,testX.shape[1]))

model = Sequential()
model.add(LSTM(50 ,batch_input_shape=(1,trainX.shape[1],trainX.shape[2]),return_sequences = True))
model.add(Dropout(0.10))
model.add(LSTM(20))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
epochs = 50
start = time.time()
m = model.fit(trainX, np.array(trainY), epochs = epochs, batch_size=1, verbose=2,validation_split=0.1)
print ("Compilation Time : ", time.time() - start)

# summarize model for loss
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss-Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

testPredict = model.predict(testX,batch_size = 1)
plt.plot(scaler1.inverse_transform(testPredict),'-b',label = 'Predicted CRL  Closing Price')
plt.plot(scaler1.inverse_transform(testY),'-g',label = 'Actual CRL Closing Price')
plt.title("CRL NYSE Actual & Predicted Closing Price from 18-05-2017 to 07-10-2010")
plt.xlabel("Time in Days")
plt.ylabel("Closing Price of stock index in USD")
plt.legend(loc='upper left')
plt.show()

scaler1.inverse_transform(testPredict)
