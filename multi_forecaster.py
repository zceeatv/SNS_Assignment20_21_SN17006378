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

    :param days: Number of forecasted days of cases to be used in the training of the model
    :return: array of previous, forecast cases
             normalisation_factor: the maximum cases number used to normalise the dataset to between 0 and 1
             dates_array: arracy containing all the dates of the dataset
    """
    df = pd.read_csv(labels_filename, usecols=column)   # Opens csv file and extracts the columns for cases and dates
    dates_array = pd.read_csv(labels_filename, usecols=["date"]) # Opens same csv file but only extracts the dates
    dates_array = dates_array.iloc[::-1]    # Reverses the array so that the start of the pandemic is at the start of the array
    dates_array = dates_array.values.tolist()   # converts panda data frame to list
    dates_array = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in dates_array]    # converts string dates to datetime format
    reversed_df = df.iloc[::-1]
    normalization_factor = reversed_df["cumCasesBySpecimenDate"].max()      # Find the maximum number of cases in the dataset
    reversed_df = reversed_df.to_numpy()
    numOfDataEntries = len(reversed_df) - (time_step + int(days))   # Determines the number of data entries that will be extracted from the dataset
    iteration = 0
    previous = []
    forecast = []
    while iteration < numOfDataEntries:     # Iterates through dataset and builds the input and output datasets for training
        previous.append(reversed_df[iteration:time_step + iteration, 1])    # Creates array for n previous day cases
        forecast.append(reversed_df[iteration + time_step: iteration + time_step + int(days), 1])      # Creates array for n forecast day cases
        iteration += 1      # Keeps track of the current data entry being created

    # Normalise all data to between 0 and 1
    previous = np.array(previous) / normalization_factor
    forecast = np.array(forecast) / normalization_factor
    return previous, forecast, normalization_factor, dates_array


def get_data(days):
    """
    Called in main to format the dataset into appropriate input and output arrays, splitting the arrays into training
    and testing data sets
    :param days: Number of forecasted days of cases to be used in the training of the model
    :return: Numpy array for training and testing data sets
             normalisation_factor: the maximum cases number used to normalise the dataset to between 0 and 1
             dates_array: arracy containing all the dates of the dataset
    """
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


def calc_performance(prediction_values, actual_values, performances, error_margin):
    """
    Function for determining the performance of the Neural Network by comparing the predicted values from the network
    to known cases. The use of an error margin defines the offset of the predicted value from the real value for which
    is acceptable and considered as a correct prediction
    :param prediction_values: Array of predicted values
    :param actual_values: Array of known values
    :param performances: Array which contains the performance percentage for each trained model (with varying previous days)
    :param error_margin: Fractional value used to define the acceptable range for the predictions
    :return: Updated array of performances with the current Neural Network score
    """
    correct = 0
    for prediction, known in zip(prediction_values, actual_values):     # Goes through each data entry of prediction and actual arrays
        for prediction_item, known_item in zip(prediction, known):      # Nested for loop to then iterate through each prediction and actual values for the current data entry
            if prediction_item > known_item and prediction_item < known_item + (known_item*error_margin):   # Determines whether the predicted value is greater than actual value but lies within the suitable range
                correct += 1
            elif prediction_item < known_item and prediction_item > known_item - (known_item*error_margin): # Determines whether the predicted value is less than actual value but lies within the suitable range
                correct += 1
            elif prediction_item == known_item:
                correct += 1
    performances.append(correct/(len(predictions)*len(predictions[1])))
    return performances


def plot_predictions(dates_array, prediction_values, actual_values, frame, day_num):
    """
    For building a series of subplots for showing the predicted values and actual values for each Neural Network model
    that were trained using different number of inputs (previous days)
    :param dates_array: Array of all dates within the dataset
    :param prediction_values: Array of predicts from the Neural Network
    :param actual_values: Array of actual case values taken from the dataset
    :param frame: Counter that tracks which subplot is being generated
    :param day_num: Number of forecasted days used for the current Neural Network
    :return: Counter for tracking subplots
    """
    axes = plt.subplot(int(max_days / 2), 2, frame)
    axes.plot_date(dates_array[start_date:start_date + len(prediction_values)], prediction_values[:, -1] , xdate=True,
                 label='Prediction', linestyle='-', marker=' ', linewidth=2)
    axes.plot_date(dates_array[start_date:start_date + len(prediction_values)], actual_values[:, -1], xdate=True, label='Actual',
                 linestyle='-', marker=' ', linewidth=2)

    plt.title("Predicting " + str(int(day_num)) + " Days")

    # Generate grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.06)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # Create single legend for entire figure
    fig.legend(axes.get_legend_handles_labels()[0], axes.get_legend_handles_labels()[1], loc='upper right')

    # Formats appropriate format for the dates in the X axis
    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%m/%Y')
    X_axis = plt.gca().xaxis
    X_axis.set_major_locator(locator)
    X_axis.set_major_formatter(fmt)
    frame += 1

    # Set the y axis to have 5 major ticks
    axes.yaxis.set_major_locator(MaxNLocator(5))

    return frame


def print_accuracy(accuracies, days):
    """
    Function for outputing the accuracy values for each Neural Network that were trained for varying number of input
    days.
    :param accuracies:  Array of accuracy values for each Neural Network
    :param days: Array containing the number of output forecasted days used for each network
    """
    for value, num_day in zip(accuracies, days):
        print(str(int(num_day)) + " Days in Advance: " + str(round(value*100, 2)) + "%")


labels_filename = "data_2021-Mar-18.csv"    # Name of CSV file containing dataset
column = ["date", "cumCasesBySpecimenDate"]     # Columns of interest from the dataset
time_step = 35  # Number of previous days the Neural Network will take in as inputs
max_days = 10   # Number of forecasting days the Neural Network will try to predict
forecast_days = np.linspace(1, max_days, max_days)
count = 1   # Counter for keeping track of frame of current subplot to be generated
accuracy = []
epochs = 300
error = 0.05    # Error margin used to determine the range of acceptable predictions for each known value

fig, ax = plt.subplots(int(max_days/2), 2)  # Create figure for subplots

for day in forecast_days:
    # Generate training and testing datasets
    tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date = get_data(day)

    X = X.astype('float32')
    tr_X = tr_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_Y = tr_Y.astype('float32')
    te_Y = te_Y.astype('float32')

    activation = 'linear'
    model = Sequential()
    model.add(Dense(256, input_dim=time_step))       # Number of input neurons depends on the number of previous days the NN will use to generate predictions
    model.add(Activation(activation))

    model.add(Dense(96))
    model.add(Activation(activation))

    model.add(Dense(48))
    model.add(Activation(activation))

    model.add(Dense(28))
    model.add(Activation(activation))

    model.add(Dense(day))  # Final layer has same number of neurons as number of forecasted days
    model.add(Activation('linear'))

    optimizer = optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    # Introducing early stoppage
    # es_callback = EarlyStopping(monitor='val_loss', patience=3), callbacks=[es_callback]
    history = model.fit(tr_X, tr_Y, epochs=epochs)

    # For Saving NN Model
    # model.save("B1_NN_Model_new")

    # Make predictions
    predictions = model.predict(te_X) * normalization
    actual = te_Y * normalization
    predictions = predictions.astype("int")
    actual = actual.astype("int")

    # Calculate accuracy of Neural Network
    accuracy = calc_performance(predictions, actual, accuracy, error)
    # Create subplot for current Neural Network performance
    count = plot_predictions(dates, predictions, actual, count, day)

print_accuracy(accuracy, forecast_days)
plt.show()
