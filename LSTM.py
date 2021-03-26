
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.dates as mdates


def preprocess():
    """
    Opens csv file containing cumulative cases in the UK since the start of the pandemic and extracts the fields
    for date and cases. Then rearranges the dataset in a rolling method into an array for previous day cases and
    forecasted day cases.

    Example for days = 14 with first data at 1/1/21 (forecasted days = 2):
    The function will first take 14 days of cases from 1/1/21 to 14/1/21, creates an array and adds this to the
    previous array and adds the cases for 15/1/21 to 16/1/21 to the forecast array. In the next iteration, the cases
    for 2/1/21 to 15/1/21 are then added to the previous array and cases for 16/1/21 to 17/1/21 is added to the forecast
    array. The function iterates until all of the dataset has been considered. Note that this introduces overlapping and
    reuse of most of the cases.

    :return: array of previous, forecast cases
             normalisation_factor: the maximum cases number used to normalise the dataset to between 0 and 1
             dates_array: arracy containing all the dates of the dataset
    """
    df = pd.read_csv(labels_filename, usecols=column)   # Opens csv file and extracts the columns for cases and dates
    dates_array = pd.read_csv(labels_filename, usecols=["date"])    # Opens same csv file but only extracts the dates
    dates_array = dates_array.iloc[::-1]    # Reverses the array so that the start of the pandemic is at the start of the array
    dates_array = dates_array.values.tolist()   # converts panda data frame to list
    dates_array = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates_array] # converts string dates to datetime format
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()  # Find the maximum number of cases in the dataset
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df) - (time_step + forecast_days)    # Determines the number of data entries that will be extracted from the dataset
    iteration = 0
    previous = []
    forecast = []
    while iteration < numOfDataEntries:     # Iterates through dataset and builds the input and output datasets for training
        previous.append(reversed_df[iteration:time_step + iteration, 1])   # Creates array for n previous day cases
        forecast.append(reversed_df[iteration + time_step: iteration + time_step + forecast_days, 1])    # Creates array for n forecast day cases
        iteration += 1      # Keeps track of the current data entry being created

    # Normalise all data to between 0 and 1
    previous = np.array(previous) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return previous, forecast, normalization_factor, dates_array


def get_data():
    """
    Called in main to format the dataset into appropriate input and output arrays, splitting the arrays into training
    and testing data sets
    :return: Numpy array for training and testing data sets
             normalisation_factor: the maximum cases number used to normalise the dataset to between 0 and 1
             dates_array: arracy containing all the dates of the dataset
    """
    X, Y, normalization_factor, dates_array = preprocess()
    dataset_size = len(X)
    training_size = int(dataset_size * 0.575)
    train_X = X[:training_size]
    train_Y = Y[:training_size]
    test_X = X[training_size:]
    test_Y = Y[training_size:]

    return train_X, train_Y, test_X, test_Y, normalization_factor, X, Y, dates_array, training_size

def plot_predictions(dates_array, prediction_values, actual_values):
    """
    For building a series of subplots for showing the predicted values and actual values for each Neural Network model
    that were trained using different number of inputs (previous days)
    :param dates_array: Array of all dates within the dataset
    :param prediction_values: Array of predicts from the Neural Network
    :param actual_values: Array of actual case values taken from the dataset

    :return: Counter for tracking subplots
    """
    plt.figure()
    plt.plot_date(dates_array[start_date:start_date + len(prediction_values)], prediction_values[:, -1], xdate=True, label='predictions', linestyle='-', marker=' ', linewidth=2)
    plt.plot_date(dates_array[start_date:start_date + len(prediction_values)], actual_values[:, -1], xdate=True, label='actual', linestyle='-', marker=' ', linewidth=2)
    plt.xlabel('Day')
    plt.ylabel('Cumulative Cases')
    plt.title('Covid Forecaster predictions')
    # Generate grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.legend()
    # Formats appropriate format for the dates in the X axis
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%m/%Y')
    X_axis = plt.gca().xaxis
    X_axis.set_major_locator(locator)
    X_axis.set_major_formatter(fmt)


labels_filename = "data_2021-Mar-18.csv"
column = ["date", "cumCasesBySpecimenDate"]
time_step = 35  # Number of previous days the Neural Network will take in as inputs
forecast_days = 1   # Number of days in advance to be forecasted by neural network
epochs = 100
error = 0.05    # Error margin used to determine the range of acceptable predictions for each known value

# Generate training and testing datasets
tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date = get_data()

X = X.astype('float32')
tr_X = tr_X.astype('float32')
te_X = te_X.astype('float32')
tr_Y = tr_Y.astype('float32')
te_Y = te_Y.astype('float32')
tr_X = tr_X.reshape(len(tr_X), time_step, 1)
te_X = te_X.reshape(len(te_X), time_step, 1)

# Part 2 - Building the RNN
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=45, return_sequences=True, input_shape=(35, 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer nd some Dropout regularisation
regressor.add(LSTM(units=45, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(tr_X, tr_Y, epochs=epochs)

# Make predictions
predictions = regressor.predict(te_X) * normalization
actual = te_Y * normalization
predictions = predictions.astype("int")
actual = actual.astype("int")

# Plot graph of predictions and actual covid cases
plot_predictions(dates, predictions, actual)

plt.show()


