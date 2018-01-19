import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from quantist.strategy.my_answers import *

from quantist.strategy.my_answers import *
from quantist.strategy.my_answers import window_transform_series
from quantist.strategy.my_answers import build_part1_RNN


data_path = "../data/pool/"
#normalized
"""
z = (x - min(x))/(max(x)-min(x))
z2 = (x - mean(x))/s
"""



### load in and normalize the dataset
dataset = np.loadtxt(data_path + 'normalized_apple_prices.csv')



plt.plot(dataset)
plt.xlabel('time period')
plt.ylabel('normalized series value')
plt.show()

odd_nums = np.array([1, 3, 5, 7, 9, 11, 13])
# run a window of size 2 over the odd number sequence and display the results
window_size = 2

X = []
X.append(odd_nums[0:2])
X.append(odd_nums[1:3])
X.append(odd_nums[2:4])
X.append(odd_nums[3:5])
X.append(odd_nums[4:6])

y = odd_nums[2:]

X = np.asarray(X)
y = np.asarray(y)
y = np.reshape(y, (len(y), 1))  # optional

assert (type(X).__name__ == 'ndarray')
assert (type(y).__name__ == 'ndarray')
assert (X.shape == (5, 2))
assert (y.shape in [(5, 1), (5,)])

# print out input/output pairs --> here input = X, corresponding output = y
print('--- the input X will look like ----')
print(X)

print('--- the associated output y will look like ----')
print(y)

# window the data using your windowing function
window_size = 7
X, y = window_transform_series(series=dataset, window_size=window_size)

print(X[:3], y[:3])

# split our dataset into training / testing sets
train_test_split = int(np.ceil(2 * len(y) / float(3)))  # set the split point

# partition the training set
X_train = X[:train_test_split, :]
y_train = y[:train_test_split]

# keep the last chunk for testing
X_test = X[train_test_split:, :]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize]
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

### TODO: create required RNN model
# import keras network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
np.random.seed(0)

# TODO: implement build_part1_RNN in my_answers.py

model = build_part1_RNN(window_size)

# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

# run your model!
model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)

# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))

### Plot everything - the original series as well as predictions on training and testing sets


# plot original series
plt.plot(dataset, color='k')

# plot training set prediction
split_pt = train_test_split + window_size
plt.plot(np.arange(window_size, split_pt, 1), train_predict, color='b')

# plot testing set prediction
plt.plot(np.arange(split_pt, split_pt + len(test_predict), 1), test_predict, color='r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series', 'training fit', 'testing fit'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
