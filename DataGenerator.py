import os
import glob
from random import shuffle

import SimpleITK as sitk
import numpy as np


def get_training_and_testing_generators(data_dir, batch_size=1, nb_channels=3, truth_channel=3,
                                        background_channel=4, z_crop=15, n_labels=5, validation_split=0.8):
    subject_folders = get_subject_folders(data_dir=data_dir)
    training_list, testing_list = split_list(subject_folders, split=validation_split, shuffle_list=True)
    training_generator = data_generator(training_list, batch_size=batch_size, nb_channels=nb_channels,
                                        truth_channel=truth_channel, background_channel=background_channel,
                                        z_crop=z_crop, n_labels=n_labels)
    testing_generator = data_generator(testing_list, batch_size=batch_size, nb_channels=nb_channels,
                                       truth_channel=truth_channel, background_channel=background_channel,
                                       z_crop=z_crop, n_labels=n_labels)
    return training_generator, testing_generator, len(training_list)/batch_size, len(testing_list)/batch_size


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def get_subject_folders(data_dir):
    return glob.glob(os.path.join(data_dir, "*", "*"))


def data_generator(subject_folders, batch_size=1, nb_channels=3, truth_channel=3, background_channel=4, z_crop=15,
                   n_labels=5):
    nb_subjects = len(subject_folders)
    while True:
        shuffle(subject_folders)
        # TODO: Edge case?
        for i in range(nb_subjects/batch_size):
            batch_folders = subject_folders[i*batch_size:(i+1)*batch_size]
            print(batch_folders)
            batch = read_batch(batch_folders, background_channel, z_crop)
            x_train, y_train = get_training_data(batch, nb_channels, truth_channel, n_labels)
            del batch, batch_folders
            yield x_train, y_train


def read_batch(folders, background_channel, z_crop):
    batch = []
    for folder in folders:
        batch.append(crop_data(read_subject_folder(folder), background_channel=background_channel, z_crop=z_crop))
    return np.asarray(batch)


def read_subject_folder(folder):
    flair_image = sitk.ReadImage(os.path.join(folder, "Flair.nii.gz"))
    t1_image = sitk.ReadImage(os.path.join(folder, "T1.nii.gz"))
    t1c_image = sitk.ReadImage(os.path.join(folder, "T1c.nii.gz"))
    truth_image = sitk.ReadImage(os.path.join(folder, "truth.nii.gz"))
    background_image = sitk.ReadImage(os.path.join(folder, "background.nii.gz"))
    return np.array([sitk.GetArrayFromImage(t1_image),
                     sitk.GetArrayFromImage(t1c_image),
                     sitk.GetArrayFromImage(flair_image),
                     sitk.GetArrayFromImage(truth_image),
                     sitk.GetArrayFromImage(background_image)])


def crop_data(data, background_channel=4, z_crop=15):
    if np.all(data[background_channel, :z_crop] == 1):
        return data[:, z_crop:]
    elif np.all(data[background_channel, data.shape[1] - z_crop:] == 1):
        return data[:, :data.shape[1] - z_crop]
    else:
        upper = z_crop/2
        lower = z_crop - upper
        return data[:, lower:data.shape[1] - upper]


def get_truth(batch, truth_channel=3, n_labels=5):
    truth = np.array(batch)[:, truth_channel]
    batch_list = []
    for sample_number in range(truth.shape[0]):
        sample_list = []
        for label in range(n_labels):
            array = np.zeros(truth[sample_number].shape)
            array[truth[sample_number] == label] = 1
            sample_list.append(array)
        batch_list.append(sample_list)
    return np.array(batch_list)


def get_training_data(batch, nb_channels, truth_channel, n_labels):
    return batch[:, :nb_channels], get_truth(batch, truth_channel, n_labels)

