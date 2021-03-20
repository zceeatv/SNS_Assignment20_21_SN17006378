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
labels_filename = "time_series_covid19_confirmed_global.csv"
column = ["Province/State", "Country/Region", "Lat", "Long"]
time_step = 14
forecast_days = 1
#def loss_function(y_true, y_pred):

def preprocess():
    history = []
    forecast = []
    df = pd.read_csv(labels_filename)
    df = df.drop(columns=column)
    regions = 0
    for index, row in df.iterrows():
        np_row = row.to_numpy()
        numOfDataEntries = len(np_row) - (time_step + forecast_days)
        iteration = 0
        while iteration < numOfDataEntries:
            if np_row[iteration] > 10:
                history.append(np.array(np_row[iteration:time_step + iteration]))
                forecast.append(np_row[iteration + time_step:+ iteration + time_step + forecast_days])
            iteration += 1
        if regions > 30:
            break
        regions += 1
    normalization_factor = np.amax(history)
    history = np.array(history) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return history, forecast, normalization_factor
    #normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
    #reversed_df = reversed_df.to_numpy()
    #print(reversed_df[0:14, 1])
    #for i in reversed_df:
    #    print(i[1])
    #print(len(df))
    """
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
    return history, forecast, normalization_factor
    """
def get_data():
    X, Y, normalization = preprocess()
    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.9)
    validation_size = training_size + int(dataset_size * 0.05)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y, normalization

tr_X, tr_Y, va_X, va_Y, te_X, te_Y, normalization = get_data()
tr_X = tr_X.astype('float32')
va_X = va_X.astype('float32')
te_X = te_X.astype('float32')
tr_Y = tr_X.astype('float32')
va_Y = va_X.astype('float32')
te_Y = te_X.astype('float32')

model = Sequential()
model.add(Dense(256, input_dim=time_step))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(96))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())


model.add(Dense(48))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(28))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(48))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(12))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(6))
model.add(Activation('tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(1))   #Final layer has same number of neurons as classes
model.add(Activation('relu'))

epochs = 1000
#batch_size = 64
optimizer = optimizers.RMSprop()

model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        #es_callback = EarlyStopping(monitor='val_loss', patience=3)
        # , callbacks=[es_callback]
history = model.fit(tr_X, tr_Y, validation_data=(va_X, va_Y), epochs=epochs)
#model.save("B1_NN_Model_new")
print("Saved Neural Network Model")
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
scores = model.evaluate(te_X, te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#return history.history["accuracy"][epochs - 1] * 100, scores[1] * 100
