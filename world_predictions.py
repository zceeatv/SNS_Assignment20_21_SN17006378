from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
labels_filename = "time_series_covid19_confirmed_global.csv"
column = ["Province/State", "Country/Region", "Lat", "Long"]
time_step = 14
forecast_days = 1

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


def calc_accuracy(prediction_values, actual_values, error_margin):
    """
    Function for determining the performance of the Neural Network by comparing the predicted values from the network
    to known cases. The use of an error margin defines the offset of the predicted value from the real value for which
    is acceptable and considered as a correct prediction
    :param prediction_values: Array of predicted values
    :param actual_values: Array of known values
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
    return correct/(len(predictions)*len(predictions[1]))


def plot_predictions(prediction_values, actual_values):
    """
    For building a series of subplots for showing the predicted values and actual values for each Neural Network model
    that were trained using different number of inputs (previous days)
    :param prediction_values: Array of predicts from the Neural Network
    :param actual_values: Array of actual case values taken from the dataset

    :return: Counter for tracking subplots
    """
    plt.figure()
    x = np.linspace(1, len(prediction_values), len(prediction_values))
    plt.plot_date(x, prediction_values[:, -1], xdate=True, label='predictions', linestyle='-', marker=' ', linewidth=2)
    plt.plot_date(x, actual_values[:, -1], xdate=True, label='actual', linestyle='-', marker=' ', linewidth=2)
    plt.xlabel('Day')
    plt.ylabel('Cumulative Cases')
    plt.title('Covid Forecaster predictions')
    # Generate grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


tr_X, tr_Y, va_X, va_Y, te_X, te_Y, normalization = get_data()
tr_X = tr_X.astype('float32')
va_X = va_X.astype('float32')
te_X = te_X.astype('float32')
tr_Y = tr_X.astype('float32')
va_Y = va_X.astype('float32')
te_Y = te_X.astype('float32')

model = Sequential()
model.add(Dense(256, input_dim=time_step))
model.add(Activation('linear'))

model.add(Dense(96))
model.add(Activation('linear'))

model.add(Dense(48))
model.add(Activation('linear'))

model.add(Dense(28))
model.add(Activation('linear'))

model.add(Dense(1))   #Final layer has same number of neurons as classes
model.add(Activation('linear'))

epochs = 100
optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        #es_callback = EarlyStopping(monitor='val_loss', patience=3)
        # , callbacks=[es_callback]
history = model.fit(tr_X, tr_Y, validation_data=(va_X, va_Y), epochs=epochs)

#model.save("B1_NN_Model_new")
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
predictions = predictions.astype("int")
actual = actual.astype("int")

# Plot graph of predictions and actual covid cases
plot_predictions(predictions, actual)
accuracy = calc_accuracy(predictions, actual, 0.05)
print("Accuracy of Model: " + str(round(accuracy*100, 2))+"%")
plt.show()
