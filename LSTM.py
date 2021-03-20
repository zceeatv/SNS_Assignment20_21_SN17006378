# Part 1 -Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(36)

labels_filename = "data_2021-Jan-25.csv"
column = ["date", "cumCasesBySpecimenDate"]

# Importing the training set
#dataset_train = pd.read_csv('covid19USA-daily.csv')
#training_set = dataset_train.iloc[:, 4:5].values

df = pd.read_csv(labels_filename, usecols=column)
reversed_df = df.iloc[::-1]
normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
reversed_df = reversed_df.to_numpy()

training_set = reversed_df

# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler

#sc = MinMaxScaler(feature_range=(0, 1))
#training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure
X_train = []
y_train = []
for i in range(80, 111):
    X_train.append(training_set[i - 80:i, 1])
    y_train.append(training_set[i, 1])
X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).astype('float32')
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape[1], 1)
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=45, return_sequences=True, input_shape=(X_train.shape[0], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer nd some Dropout regularisation
regressor.add(LSTM(units=45, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=45, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=45))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=80)

# Part 3 - Making the predictions

"""
# Getting the real data
dataset_test = pd.read_csv('usa2.csv')
real_confirmed_rate = dataset_test.iloc[:, 4:5].values

# Getting the predicted data
dataset_total = pd.concat((dataset_train['Confirmed'], dataset_test['Confirmed']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 80:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(80, 90):
    X_test.append(inputs[i - 80:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_confirmed_rate = regressor.predict(X_test)
predicted_confirmed_rate = sc.inverse_transform(predicted_confirmed_rate)
"""