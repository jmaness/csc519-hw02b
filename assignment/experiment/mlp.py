#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act
import pandas as pd

from functools import reduce

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 80, 10])
    num_epochs = 125

    # train the network using SGD
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy =    model.SGD(
        training_data=train_data,
        epochs=num_epochs,
        mini_batch_size=128,
        eta=0.0015,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    # Compute accuracy percentages
    training_accuracy_percentage = list(map(lambda x: x / len(train_data[0]), training_accuracy))
    validation_accuracy_percentage = list(map(lambda x: x / len(valid_data[0]), evaluation_accuracy))

    # Plot training and validation loss and accuracy vs. epochs
    epochs = list(range(num_epochs))
    fig = plt.figure()

    axes1 = fig.add_subplot(121)
    axes1.plot(epochs, evaluation_cost, 'g', label='Validation')
    axes1.plot(epochs, training_cost, 'b', label='Training')
    axes1.set_ylabel('Loss')
    axes1.set_xlabel('Epochs')
    axes1.legend()

    axes2 = fig.add_subplot(122)
    axes2.plot(epochs, np.array(validation_accuracy_percentage), 'g', label='Validation')
    axes2.plot(epochs, np.array(training_accuracy_percentage), 'b', label='Training')
    axes2.set_ylabel('Accuracy')
    axes2.set_xlabel('Epochs')
    axes2.legend()

    plt.show()

    # Compute test accuracy
    columns = list(range(10))
    df = pd.DataFrame(columns=columns)

    for data in test_data[0]:
        prediction = model.feedforward(data)
        temp = pd.DataFrame(prediction.transpose(), columns=columns)
        df = df.append(temp)

    df.to_csv("predictions_raw.csv", index=False, header=False)

    test_accuracy = model.accuracy(test_data) / len(test_data[0])
    print("Test accuracy: {:.2%}".format(test_accuracy))

    # One-hot encoding of predictions for the test set
    for i in range(df.shape[0]):
        maximum = df.iloc[i].max()
        for j in range(df.shape[1]):
            if df.iloc[i][j] == maximum:
                df.iloc[i][j] = 1
            else:
                df.iloc[i][j] = 0

    df = df.astype(int)
    df.to_csv("predictions.csv", index=False, header=False)


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
