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

    #
    # l = 1
    # for w in range(network[l]):
    #     print("layer_id: {}, unit_id: {}, weight_id: {}".format(l, 5, w))
    #     model.gradient_check(training_data=train_data, layer_id=l, unit_id=5, weight_id=w)

    # for l in range(3):
    #     for u in range(network[l]):
    #         for w in range(network[l]):
    #             print("layer_id: {}, unit_id: {}, weight_id: {}".format(l, u, w))
    #             try:
    #                 model.gradient_check(training_data=train_data, layer_id=l, unit_id=u, weight_id=w)
    #             except IndexError:
    #                 print("out of bounds")
    #
    #             print("======================================================")

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 20, 10])
    num_epochs = 100

    # train the network using SGD
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy =    model.SGD(
        training_data=train_data,
        epochs=num_epochs,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    # model.save("model_784_20_10.json")

    # model = network2.load("model_784_20_10.json")
    #
    # training_accuracy = model.accuracy(train_data, convert=True)
    # training_cost = model.total_cost(train_data, 0.0)
    # evaluation_accuracy = model.accuracy(valid_data)
    # evaluation_cost = model.total_cost(valid_data, 0.0, convert=True)

    training_accuracy_percentage = list(map(lambda x: x / len(train_data[0]), training_accuracy))
    validation_accuracy_percentage = list(map(lambda x: x / len(valid_data[0]), evaluation_accuracy))


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

    fig.title('Prior to optimization')
    plt.show()


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
