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
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


def preprocess(days):
    df = pd.read_csv(labels_filename, usecols=column)
    dates = pd.read_csv(labels_filename, usecols=["date"])
    dates = dates.iloc[::-1]
    dates = dates.values.tolist()
    dates = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates]
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df) - (time_step + int(days))
    iteration = 0
    history = []
    forecast = []
    while iteration < numOfDataEntries:
        history.append(reversed_df[iteration:time_step + iteration, 1])
        forecast.append(reversed_df[iteration + time_step:+ iteration + time_step + int(days), 1])
        iteration += 1
    history = np.array(history) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return history, forecast, normalization_factor, dates


def get_data(days):
    X, Y, normalization, dates = preprocess(days)
    dataset_size = len(X)
    training_size = int(dataset_size * 0.575)
    start_date = dates[training_size]
    # validation_size = training_size + int(dataset_size * 0.35)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, training_size


labels_filename = "data_2021-Mar-18.csv"
column = ["date", "cumCasesBySpecimenDate"]
time_step = 30
max_days=10
forecast_days = np.linspace(1, max_days, max_days)
count = 1

fig, ax = plt.subplots(int(max_days/2), 2)

for day in forecast_days:
    tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date = get_data(day)

    X = X.astype('float32')
    tr_X = tr_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_Y = tr_Y.astype('float32')
    te_Y = te_Y.astype('float32')

    activation = 'relu'
    model = Sequential()
    model.add(Dense(256, input_dim=time_step))
    model.add(Activation(activation))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(96))
    model.add(Activation(activation))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(48))
    model.add(Activation(activation))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(28))
    model.add(Activation(activation))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
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
    model.add(Dense(day))  # Final layer has same number of neurons as classes
    model.add(Activation('linear'))

    epochs = 10
    # batch_size = 64
    optimizer = optimizers.Nadam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    # es_callback = EarlyStopping(monitor='val_loss', patience=3)
    # , callbacks=[es_callback]
    history = model.fit(tr_X, tr_Y, epochs=epochs)
    # model.save("B1_NN_Model_new")
    # print("Saved Neural Network Model")

    """
    plt.plot(history.history['loss'])
    plt.title("Learning Rate Curve for B1's CNN Model")
    plt.ylabel('Cost', fontsize='large', fontweight='bold')
    plt.xlabel('Number of Epochs', fontsize='large', fontweight='bold')

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
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
    ax = plt.subplot(int(max_days/2), 2, count)
    ax.plot_date(dates[start_date:start_date + len(predictions)], predictions[:, -1] / 1e6, xdate=True,
                 label='Prediction', linestyle='-', marker=' ', linewidth=2)
    ax.plot_date(dates[start_date:start_date + len(predictions)], actual[:, -1] / 1e6, xdate=True, label='Actual',
                 linestyle='-', marker=' ', linewidth=2)

    plt.title("Predicting " + str(int(day)) + " Days")
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=.06)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    fig.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc='upper right')
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%m/%Y')
    X_axis = plt.gca().xaxis
    X_axis.set_major_locator(locator)
    X_axis.set_major_formatter(fmt)
    count += 1
    # plt.xticks(rotation=30)

    ax.yaxis.set_major_locator(MaxNLocator(5))

plt.show()
