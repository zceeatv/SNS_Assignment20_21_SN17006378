# SNS Assignment 20/21 SN17006378

### single_forecaster.py
- Finalised Neural Network model
- Preset to 35 previous days (input to NN), 10 forcasted day (output from NN), 300 epochs
- The forecasted days can be adjusted to the users preference 
- Once trained, the model makes predictions on 5 months of cumulative cases
- Plots graph of predictions and actual case numbers
- Calculates accuracy score

### multi_forecaster.py
- Finalised Neural Network model
- Preset to 35 previous days (input to NN), 300 epochs
- Iterates through training process through different number of output forecasted days
- Preprocessing will need to be carried out for each iteration, as to restructure the dataset to be passed to the neural network
- Once trained, the model makes predictions on 5 months of cumulative cases
- Plots graph of predictions and actual case numbers
- Calculates accuracy score

### previous_days.py
- For Hyperparamter optimisation
- Iterates through training process through different number of inputs are passed to the neural network
- Preprocessing will need to be carried out for each iteration, as to restructure the dataset to be passed to the neural network
- Preset to 5 forcasted day (output from NN), 300 epochs 
- Once trained, the model makes predictions on 5 months of cumulative cases
- Plots graph of predictions and actual case numbers
- Calculates accuracy score for each model

### activation_functions.py
- For Hyperparamter optimisation
- Iterates through training process 4 times with different activation functions for all hidden layers (Tanh, Relu, Sigmoid, Linear)
- Preset to 35 previous days (input to NN), 1 forcasted day (output from NN), 300 epochs 
- Once trained, the model makes predictions on 5 months of cumulative cases
- Plots graph of predictions and actual case numbers
- Calculates accuracy score for each activation function


## optimisers.py
- For Hyperparamter optimisation
- Iterates through training process 4 times with different optimisation algorithms (Adam, Nadam,SGD, RMSprop)
- Preset to 35 previous days (input to NN), 3 forcasted day (output from NN), 300 epochs 
- Plots graph of the loss function for the training process for each optimiser


#### System used for training Models
Computational times provided in this READ.me are based off the following system configurations:
 - Intel Core i7-8750H	2.2-4.1GHz	6/12 cores	9 MB cache
 - Nvidia Quadro P1000	640 CUDA 1.49-1.52GHz	4 GB GDDR5
 - Setup of CUDA toolkit allowing Quadro P1000 for use in training tenserflow models
 
### Packages requirements:
- Numpy 1.19.2
- Pandas 1.1.3
- Tensorflow 2.1.0
- Keras 2.3.1
