import os
import glob
import pickle
from random import shuffle

import numpy as np
from nilearn.image import resample_img, reorder_img
import nibabel as nib


# TODO: Rescale images to integer
# TODO: include normalization script from raw BRATS data
# TODO: find the smallest shape image that contains all of the original data
# TODO: set background to zero after resampling

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_training_and_testing_generators(data_dir, input_shape, batch_size=1, nb_channels=3, validation_split=0.8,
                                        overwrite=False, saved_folders_file="training_and_testing_folders.pkl"):
    if overwrite or not os.path.exists(saved_folders_file):
        subject_folders = get_subject_folders(data_dir=data_dir)
        training_list, testing_list = split_list(subject_folders, split=validation_split, shuffle_list=True)
        pickle_dump((training_list, testing_list), saved_folders_file)
    else:
        training_list, testing_list = pickle_load(saved_folders_file)
    training_generator = data_generator(training_list, batch_size=batch_size, nb_channels=nb_channels,
                                        input_shape=input_shape)
    testing_generator = data_generator(testing_list, batch_size=batch_size, nb_channels=nb_channels,
                                       input_shape=input_shape)
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


def get_subject_folders(data_dir):
    return glob.glob(os.path.join(data_dir, "*", "*"))


def data_generator(subject_folders, input_shape, batch_size=1, nb_channels=3):
    nb_subjects = len(subject_folders)
    while True:
        shuffle(subject_folders)
        nb_batches = nb_subjects/batch_size
        # TODO: Edge case? Currently this is handled by flooring the number of training/testing samples
        for i in range(nb_batches):
            batch_folders = subject_folders[i*batch_size:(i+1)*batch_size]
            batch = read_batch(batch_folders, input_shape)
            x_train, y_train = get_training_data(batch, nb_channels, truth_channel=3)
            del batch, batch_folders
            yield x_train, y_train


def read_batch(folders, input_shape):
    batch = []
    for folder in folders:
        batch.append(read_subject_folder(folder, input_shape))
    return np.asarray(batch)


def read_subject_folder(folder, image_size):
    flair_image = read_image(os.path.join(folder, "Flair.nii.gz"), image_size=image_size)
    t1_image = read_image(os.path.join(folder, "T1.nii.gz"), image_size=image_size)
    t1c_image = read_image(os.path.join(folder, "T1c.nii.gz"), image_size=image_size)
    truth_image = read_image(os.path.join(folder, "truth.nii.gz"), image_size=image_size,
                             interpolation="nearest")
    return np.asarray([t1_image.get_data(), t1c_image.get_data(), flair_image.get_data(), truth_image.get_data()])


def read_image(in_file, image_size, interpolation='continuous'):
    image = nib.load(in_file)
    return resize(image, new_shape=image_size, interpolation=interpolation)


def resize(image, new_shape, interpolation="continuous"):
    ras_image = reorder_img(image, resample=interpolation)
    input_shape = np.asarray(image.shape, dtype=np.float16)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


def get_truth(batch, truth_channel=3):
    truth = np.array(batch)[:, truth_channel]
    batch_list = []
    for sample_number in range(truth.shape[0]):
        array = np.zeros(truth[sample_number].shape)
        array[truth[sample_number] > 0] = 1
        batch_list.append([array])
    return np.array(batch_list)


def get_training_data(batch, nb_channels, truth_channel):
    return batch[:, :nb_channels], get_truth(batch, truth_channel)
