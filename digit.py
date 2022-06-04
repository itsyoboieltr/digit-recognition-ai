import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, models, Model, Sequential
from matplotlib import pyplot as plt
import tensorflowjs as tfjs

def plot_curve(epochs: int, hist, list_of_metrics: list[str]):
    '''Plot a curve of one or more classification metrics vs. epoch.'''

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    for metric in list_of_metrics:
        x = hist[metric]
        plt.plot(epochs[1:], x[1:], label=metric)

    plt.legend()
    # plt.show()


def create_model(my_learning_rate: float):
    '''Create and compile a deep neural net.'''
    model = models.Sequential()

    # The features are stored in a two-dimensional 28X28 array. 
    # Flatten that two-dimensional array into a a one-dimensional 784-element array.
    model.add(layers.Flatten(input_shape=(28, 28)))

    # Define the first hidden layer.   
    model.add(layers.Dense(units=256, activation='relu'))
    
    # Define a dropout regularization layer. 
    model.add(layers.Dropout(rate=0.2))

    # Define the second hidden layer.   
    model.add(layers.Dense(units=128, activation='relu'))

    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive).
    model.add(layers.Dense(units=10, activation='softmax'))     
         
    # Construct the layers into a model that TensorFlow can execute.  
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.  

    model.compile(optimizer=optimizers.Adam(learning_rate=my_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model    


def train_model(model: Sequential, train_features, train_label, epochs: int, batch_size: int, validation_split: float):
    '''Train the model by feeding it data.'''

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)

    # To track the progression of training, gather a snapshot of the model's metrics at each epoch. 
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


def main():    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train_normalized = x_train / 255.0
    x_test_normalized = x_test / 255.0

    # The following variables are the hyperparameters.
    learning_rate = 0.003
    epochs = 50
    batch_size = 4000
    validation_split = 0.2

    # Establish the model's topography.
    my_model = create_model(learning_rate)

    # Train the model on the normalized training set.
    epochs, hist = train_model(my_model, x_train_normalized, y_train, epochs, batch_size, validation_split)

    # Plot a graph of the metric vs. epochs.
    list_of_metrics_to_plot = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics_to_plot)

    # Evaluate against the test set.
    print('\n Evaluate the new model against the test set:')
    my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

    # Save the model.
    # my_model.save('model', save_format='h5')
    tfjs.converters.save_keras_model(my_model, 'model')


if __name__ == '__main__':
    main()
