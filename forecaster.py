from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


def preprocess(days):
    df = pd.read_csv(labels_filename, usecols=column)
    dates_array = pd.read_csv(labels_filename, usecols=["date"])
    dates_array = dates_array.iloc[::-1]
    dates_array = dates_array.values.tolist()
    dates_array = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates_array]
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df) - (time_step + int(days))
    iteration = 0
    previous = []
    forecast = []
    while iteration < numOfDataEntries:
        previous.append(reversed_df[iteration:time_step + iteration, 1])
        forecast.append(reversed_df[iteration + time_step:+ iteration + time_step + int(days), 1])
        iteration += 1
    previous = np.array(previous) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return previous, forecast, normalization_factor, dates_array


def get_data(days):
    X, Y, normalization_factor, dates_array = preprocess(days)
    dataset_size = len(X)
    training_size = int(dataset_size * 0.575)
    train_X = X[:training_size]
    train_Y = Y[:training_size]
    test_X = X[training_size:]
    test_Y = Y[training_size:]

    return train_X, train_Y, test_X, test_Y, normalization_factor, X, Y, dates_array, training_size


def calc_average_accuracy(prediction_values, actual_values, accuracies):
    running_total = 0
    for prediction, known in zip(prediction_values, actual_values):
        running_total += ((known[-1] - abs(prediction[-1] - known[-1])) / known[-1])

    accuracies.append(running_total / len(prediction_values))
    return accuracies


def plot_predictions(dates_array, prediction_values, actual_values, frame, day_num):
    axes = plt.subplot(int(max_days / 2), 2, frame)
    axes.plot_date(dates_array[start_date:start_date + len(prediction_values)], prediction_values[:, -1] / 1e6, xdate=True,
                 label='Prediction', linestyle='-', marker=' ', linewidth=2)
    axes.plot_date(dates_array[start_date:start_date + len(prediction_values)], actual_values[:, -1] / 1e6, xdate=True, label='Actual',
                 linestyle='-', marker=' ', linewidth=2)

    plt.title("Predicting " + str(int(day_num)) + " Days")
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=.06)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    fig.legend(axes.get_legend_handles_labels()[0], axes.get_legend_handles_labels()[1], loc='upper right')
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%m/%Y')
    X_axis = plt.gca().xaxis
    X_axis.set_major_locator(locator)
    X_axis.set_major_formatter(fmt)
    frame += 1

    axes.yaxis.set_major_locator(MaxNLocator(5))

    return frame


def print_accuracy(accuracies, days):
    for value, num_day in zip(accuracies, days):
        print(str(int(num_day)) + " Days in Advance: " + str(round(value*100, 2)) + "%")


labels_filename = "data_2021-Mar-18.csv"
column = ["date", "cumCasesBySpecimenDate"]
time_step = 30
max_days = 10
forecast_days = np.linspace(1, max_days, max_days)
count = 1
accuracy = []
epochs = 10

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
    model.add(Dense(day))  # Final layer has same number of neurons as classes
    model.add(Activation('linear'))

    optimizer = optimizers.Nadam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    # es_callback = EarlyStopping(monitor='val_loss', patience=3)
    # , callbacks=[es_callback]
    history = model.fit(tr_X, tr_Y, epochs=epochs)
    # model.save("B1_NN_Model_new")
    # print("Saved Neural Network Model")

    # Model evaluation
    predictions = model.predict(te_X) * normalization
    actual = te_Y * normalization

    accuracy = calc_average_accuracy(predictions, actual, accuracy)
    count = plot_predictions(dates, predictions, actual, count, day)

print_accuracy(accuracy, forecast_days)
plt.show()
