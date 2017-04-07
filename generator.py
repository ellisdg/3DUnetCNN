import os
from random import shuffle

from utils.utils import pickle_dump, pickle_load
from config import config


# TODO: Rescale images to integer
# TODO: include normalization script from raw BRATS data


def get_training_and_testing_generators(data_file, batch_size, data_split=0.8, overwrite=False):
    training_list, testing_list = get_validation_split(data_file, data_split=data_split, overwrite=overwrite)
    training_generator = data_generator(data_file, training_list, batch_size=batch_size)
    testing_generator = data_generator(data_file, testing_list, batch_size=batch_size)
    # Set the number of training and testing samples per epoch correctly
    nb_training_samples = len(training_list)/batch_size * batch_size
    nb_testing_samples = len(testing_list)/batch_size * batch_size
    return training_generator, testing_generator, nb_training_samples, nb_testing_samples


def get_validation_split(data_file, data_split=0.8, overwrite=False):
    if overwrite or not os.path.exists(config["training_file"]):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, testing_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, config["training_file"])
        pickle_dump(testing_list, config["testing_file"])
        return training_list, testing_list
    else:
        print("Loading previous validation split...")
        return pickle_load(config["training_file"]), pickle_load(config["testing_file"])


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, binary=True):
    nb_subjects = len(index_list)
    while True:
        shuffle(index_list)
        nb_batches = nb_subjects/batch_size
        # TODO: Edge case? Currently this is handled by flooring the number of training/testing samples
        for i in range(nb_batches):
            x = data_file.root.data[i*batch_size:(i+1)*batch_size]
            y = data_file.root.truth[i*batch_size:(i+1)*batch_size]
            if binary:
                y[y > 0] = 1
            else:
                raise NotImplementedError("Multi-class labels are not yet implemented")
            yield x, y
