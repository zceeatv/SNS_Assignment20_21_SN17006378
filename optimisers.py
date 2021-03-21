from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


def preprocess():
    df = pd.read_csv(labels_filename, usecols=column)
    dates_array = pd.read_csv(labels_filename, usecols=["date"])
    dates_array = dates_array.iloc[::-1]
    dates_array = dates_array.values.tolist()
    dates_array = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates_array]
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df) - (time_step + forecast_days)
    iteration = 0
    previous = []
    forecast = []
    while iteration < numOfDataEntries:
        previous.append(reversed_df[iteration:time_step + iteration, 1])
        forecast.append(reversed_df[iteration + time_step:+ iteration + time_step + forecast_days, 1])
        iteration += 1
    previous = np.array(previous) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return previous, forecast, normalization_factor, dates_array


def get_data():
    X, Y, normalization_factor, dates_array = preprocess()
    dataset_size = len(X)
    training_size = int(dataset_size * 0.575)
    train_X = X[:training_size]
    train_Y = Y[:training_size]
    test_X = X[training_size:]
    test_Y = Y[training_size:]

    return train_X, train_Y, test_X, test_Y, normalization_factor, X, Y, dates_array, training_size


def plot_cost_vs_epoch(axes, history_array, optimiser_type, frame):
    axes = plt.subplot(2, 2, frame)
    axes.plot(history_array.history['loss'])
    plt.title(str(optimiser_type) + " Performance")
    plt.ylabel('Cost')
    plt.xlabel('Number of Epochs')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.06)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    frame += 1

    return frame


labels_filename = "data_2021-Mar-18.csv"
column = ["date", "cumCasesBySpecimenDate"]
time_step = 30
forecast_days = 1
steps = np.linspace(5, 50, 10)
optimisers = ("Adam", "Nadam", "SGD", "RMSprop")
count = 1

tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date = get_data()
fig, ax = plt.subplots(2, 2)

for optimiser in optimisers:
    X = X.astype('float32')
    tr_X = tr_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_Y = tr_Y.astype('float32')
    te_Y = te_Y.astype('float32')

    activation = 'relu'
    model = Sequential()
    model.add(Dense(256, input_dim=time_step))
    model.add(Activation(activation))

    model.add(Dense(96))
    model.add(Activation(activation))

    model.add(Dense(48))
    model.add(Activation(activation))

    model.add(Dense(28))
    model.add(Activation(activation))
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
    model.add(Dense(forecast_days))  # Final layer has same number of neurons as classes
    model.add(Activation('linear'))

    epochs = 10

    model.compile(loss='mean_squared_error', optimizer=optimiser, metrics=['mse'])
    # es_callback = EarlyStopping(monitor='val_loss', patience=3)
    # , callbacks=[es_callback]
    history = model.fit(tr_X, tr_Y, epochs=epochs)

    count = plot_cost_vs_epoch(ax, history, optimiser, count)

plt.show()
