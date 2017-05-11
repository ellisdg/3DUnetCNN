import os
from random import shuffle

import numpy as np

from .utils import pickle_dump, pickle_load


def get_training_and_testing_generators(data_file, batch_size, n_labels, training_file, testing_file, data_split=0.8,
                                        overwrite=False):
    training_list, testing_list = get_validation_split(data_file, data_split=data_split, overwrite=overwrite,
                                                       training_file=training_file, testing_file=testing_file)
    training_generator = data_generator(data_file, training_list, batch_size=batch_size, n_labels=n_labels)
    testing_generator = data_generator(data_file, testing_list, batch_size=batch_size, n_labels=n_labels)
    # Set the number of training and testing samples per epoch correctly
    num_training_steps = len(training_list)//batch_size
    num_validation_steps = len(testing_list)//batch_size
    return training_generator, testing_generator, num_training_steps, num_validation_steps


def get_validation_split(data_file, training_file, testing_file, data_split=0.8, overwrite=False):
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, testing_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(testing_list, testing_file)
        return training_list, testing_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1):
    x_list = list()
    y_list = list()
    while True:
        shuffle(index_list)
        for index in index_list:
            x_list.append(data_file.root.data[index])
            y_list.append(data_file.root.truth[index])
            if len(x_list) == batch_size:
                yield convert_data(x_list, y_list, n_labels=n_labels)
                x_list = list()
                y_list = list()


def convert_data(x_list, y_list, n_labels=1):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels)
    return x, y


def get_multi_class_labels(data, n_labels):
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
