from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import numpy as np
import os
from os.path import dirname, abspath, split
import pandas as pd
import datetime
labels_filename = "data_2021-Mar-18.csv"
column = ["date", "cumCasesBySpecimenDate"]
time_step = 30
forecast_days = 1

def preprocess():
    df = pd.read_csv(labels_filename, usecols=column)
    dates = pd.read_csv(labels_filename, usecols=["date"])
    dates = dates.iloc[::-1]
    dates = dates.values.tolist()
    dates = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates]
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df)-(time_step + forecast_days)
    iteration = 0
    history = []
    forecast = []
    while iteration < numOfDataEntries:
        history.append(reversed_df[iteration:time_step + iteration, 1])
        forecast.append(reversed_df[iteration+time_step:+ iteration+time_step + forecast_days, 1])
        iteration += 1
    history = np.array(history)/normalization_factor
    forecast = np.array(forecast)/normalization_factor
    return history, forecast, normalization_factor, dates, numOfDataEntries

def get_data():
    X, Y, normalization, dates, numOfDataEntries = preprocess()
    dataset_size = len(X)
    training_size = int(dataset_size * 0.85)
    start_date = dates[training_size]
    #validation_size = training_size + int(dataset_size * 0.35)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, training_size, numOfDataEntries

tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date, numOfDataEntries = get_data()

print(tr_X[:])

X = X.astype('float32')
tr_X = tr_X.astype('float32')
te_X = te_X.astype('float32')
tr_Y = tr_Y.astype('float32')
te_Y = te_Y.astype('float32')

act1 = 'relu'
model = Sequential()
model.add(Dense(256, input_dim=time_step))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(96))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())


model.add(Dense(48))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(28))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
"""
model.add(Dense(48))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(12))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(6))
model.add(Activation(act1))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
"""
model.add(Dense(forecast_days))   #Final layer has same number of neurons as classes
model.add(Activation('linear'))

epochs = 30
#batch_size = 64
optimizer = optimizers.Nadam()

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
        #es_callback = EarlyStopping(monitor='val_loss', patience=3)
        # , callbacks=[es_callback]
history = model.fit(tr_X, tr_Y, epochs=epochs)
model.save("B1_NN_Model_new")
#print("Saved Neural Network Model")
"""
        plt.plot(history.history['loss'],marker='x')
        plt.plot(history.history['val_loss'], marker='x')
        plt.title("Learning Rate Curve for B1's CNN Model")
        plt.ylabel('Cost', fontsize='large', fontweight='bold')
        plt.xlabel('Number of Epochs', fontsize='large', fontweight='bold')
        plt.legend(['train', 'test'], loc='upper left')
        plt.rcParams.update({'font.size': 22})
        plt.show()
"""
# Model evaluation
predictions = model.predict(te_X) * normalization
actual = te_Y * normalization

"""
predictions = model.predict(tr_X) * normalization
actual = tr_Y * normalization

predictions = model.predict(X) * normalization
actual = Y*normalization
"""
lengths = np.linspace(1, len(predictions), len(predictions))
plt.figure()
plt.plot_date(dates[start_date:start_date+len(predictions)], predictions[:,-1], xdate=True, label='predictions')
plt.plot_date(dates[start_date:start_date+len(predictions)], actual[:,-1], xdate=True, label='actual')
plt.xlabel('Day')
plt.ylabel('Cumalative Cases')
plt.title('Covid Forecaster predictions')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()
plt.show()

scores = model.evaluate(te_X, te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#return history.history["accuracy"][epochs - 1] * 100, scores[1] * 100
