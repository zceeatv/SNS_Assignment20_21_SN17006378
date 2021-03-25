from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


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


def plot_cost_vs_epoch(axes, history_array, optimiser_type, frame):
    """
    For building a series of subplots for showing the cost function of the neural network training vs the epoch
    :param history_array: 2D array containing both epoch and corresponding loss function value
    :param optimiser_type: String of the optimiser used for current Neural Network
    :param frame: Counter that tracks which subplot is being generated
    :return: Counter for tracking subplots
    """
    axes = plt.subplot(2, 2, frame)
    axes.plot(history_array.history['loss'])
    plt.title(str(optimiser_type) + " Performance")
    plt.ylabel('Cost')
    plt.xlabel('Number of Epochs')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.06)

    # Generate grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    frame += 1

    return frame


labels_filename = "data_2021-Mar-18.csv"    # Name of CSV file containing dataset
column = ["date", "cumCasesBySpecimenDate"]     # Columns of interest from the dataset
time_step = 35  # Number of previous days the Neural Network will take in as inputs
forecast_days = 3   # Number of days in advance to be forecasted by neural network
optimisers = ("Adam", "Nadam", "SGD", "RMSprop")
count = 1   # Counter for keeping track of frame of current subplot to be generated
epochs = 300

# Generate training and testing datasets
tr_X, tr_Y, te_X, te_Y, normalization, X, Y, dates, start_date = get_data()
fig, ax = plt.subplots(2, 2)  # Create figure for subplots

for optimiser in optimisers:
    X = X.astype('float32')
    tr_X = tr_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_Y = tr_Y.astype('float32')
    te_Y = te_Y.astype('float32')

    activation = 'relu'
    model = Sequential()
    model.add(Dense(256, input_dim=time_step))      # Number of input neurons depends on the number of previous days the NN will use to generate predictions
    model.add(Activation(activation))

    model.add(Dense(96))
    model.add(Activation(activation))

    model.add(Dense(48))
    model.add(Activation(activation))

    model.add(Dense(28))
    model.add(Activation(activation))

    model.add(Dense(forecast_days))  # Final layer has same number of neurons as classes
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=optimiser, metrics=['mse'])

    # Introducing early stoppage
    # es_callback = EarlyStopping(monitor='val_loss', patience=3), callbacks=[es_callback]

    history = model.fit(tr_X, tr_Y, epochs=epochs)

    # For Saving NN Model
    # model.save("B1_NN_Model_new")

    # Create subplot for current Neural Network learning history
    count = plot_cost_vs_epoch(ax, history, optimiser, count)

plt.show()
