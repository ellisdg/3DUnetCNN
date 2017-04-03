from random import shuffle

from utils.utils import pickle_dump

# TODO: Rescale images to integer
# TODO: include normalization script from raw BRATS data
# TODO: normalize data by subtracting mean and then dividing by standard deviation
# TODO: crop data to the smallest shape image that contains all of the original data
# TODO: set background to zero after resampling


def get_training_and_testing_generators(data_file, batch_size, data_split=0.8):
    nb_samples = data_file.root.data.shape[0]
    sample_list = range(nb_samples)
    training_list, testing_list = split_list(sample_list, split=data_split)
    pickle_dump(training_list, "training_list.pkl")
    pickle_dump(testing_list, "testing_list.pkl")
    training_generator = data_generator(data_file, training_list, batch_size=batch_size)
    testing_generator = data_generator(data_file, testing_list, batch_size=batch_size)
    # Set the number of training and testing samples per epoch correctly
    nb_training_samples = len(training_list)/batch_size * batch_size
    nb_testing_samples = len(testing_list)/batch_size * batch_size
    return training_generator, testing_generator, nb_training_samples, nb_testing_samples


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
