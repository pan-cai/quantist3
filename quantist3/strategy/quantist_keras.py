# -*- coding: utf-8 -*-

# quantist
# 
# Copyright 2017-2018 Pan Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Unless required

""" 
Author: liupan 
"""

"""
Description:
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class QuantistKeras(object):

    def demo_regressor(self):
        np.random.seed(1337)

        x = np.linspace(-1,1,200)
        np.random.shuffle(x)
        y = 0.5*x + 2 + np.random.normal(0,0.05,(200,))

        plt.scatter(x,y)
        plt.show()

        x_train,y_train = x[:160], y[:160]
        x_test,y_test = x[160:], y[160:]

        model = Sequential()
        model.add(Dense(output_dim=1,units=1))

        model.compile(loss='mse',optimizer='sgd')

        print('training...')
        for step in range(301):
            cost = model.train_on_batch(x_train,y_train)
            if step%100 == 0:
                print('train cost ',cost)

        print('\n testing...')
        cost = model.evaluate(x_test,y_test,batch_size=40)
        print('test cost',cost)
        w,b = model.layers[0].get_weights()
        print('weight=',w,'\n biases=',b)

        y_pred = model.predict(x_test)
        plt.scatter(x_test,y_test)
        plt.plot(x_test,y_pred)
        plt.show()

    #It has tesed
    def demo_google(self,train_data,test_data):


        # 2012 to 2016

        # Part 1- Data Preprocessing

        # importing training set
        #training_set = pd.read_excel(data_path + 'Google_Stock_Price_Train.xls')
        training_set = train_data

        # extract open value from the trainng data
        training_set = training_set.iloc[:, 1:2].values

        # Feature Scaling

        sc = MinMaxScaler()
        training_set = sc.fit_transform(training_set)

        # Getting the input and output
        X_train = training_set[0:1257]
        Y_train = training_set[1:1258]

        # Reshaping
        X_train = np.reshape(X_train, (1257, 1, 1))

        # Part-2 Building RNN

        # Initalizing RNN

        regressor = Sequential()

        regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

        # Adding output layer (default argument)
        regressor.add(Dense(units=1))

        # Compile LSTM
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN on training set
        regressor.fit(X_train, Y_train, batch_size=32, epochs=200)

        # Part 3-Making Prediction and Visualizing Results

        # Getting real Stock price for 2017
        #test_set = pd.read_excel(data_path + 'Google_Stock_Price_Test.xls')
        test_set = test_data
        real_stock_price = test_set.iloc[:, 1:2].values

        # Getting predicted Stock price for 2017
        inputs = real_stock_price
        inputs = sc.transform(inputs)
        inputs = np.reshape(inputs, (20, 1, 1))  # scaling the values

        predicted_stock_price = regressor.predict(inputs)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # scaling to input values

        # Visualize the results
        plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
        plt.plot(predicted_stock_price, color='green', label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()

        # Part 4- Evaluating the RNN
        # since it is linear regression problem we will evaluate RMSE

        import math
        from sklearn.metrics import mean_squared_error
        rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # expressing RMSE in percentage
        rmse = rmse / 800  # 800 becasue it is average value

    def demo_google_2(self):
        # Recurrent Neural Network



        # Part 1 - Data Preprocessing

        # Importing the libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        # importing training set

        data_path = "../quantist/data/pool/"

        dataset_training = pd.read_csv(data_path + "Google_Stock_Price_Train.xls")
        training_set = dataset_training.iloc[:, 1:2].values
        # feature scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        #
        X_train = []
        y_train = []
        for i in range(60, 1258):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        # reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        # building rnn
        # importing libraries
        from keras.models import Sequential
        from keras.layers import LSTM
        from keras.layers import Dense
        from keras.layers import Dropout
        # initialising rnn
        regressor = Sequential()
        # first layer
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        # second layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # third layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # fourth layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        # adding the output layer
        regressor.add(Dense(units=1))
        # compile
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        # fitting the training data
        regressor.fit(X_train, y_train, epochs=100, batch_size=32)
        # whats our test data
        dataset_test = pd.read_csv(data_path + "Google_Stock_Price_Test.xls")
        real_stock_price = dataset_test.iloc[:, 1:2].values
        # Getting the Predicted Stock Price
        dataset_total = pd.concat((dataset_training['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_training) - 60:].values
        inputs = inputs.reshape(-1, 1)  # becoz we did not use iloc method
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, 80):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # making visualizations
        plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()


"""
Sample test
"""
"""
# demo_regressor
q = QuantistKeras()
q.demo_regressor()
"""


