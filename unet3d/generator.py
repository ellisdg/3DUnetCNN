import os
import copy
from time import sleep
import multiprocessing
from multiprocessing import Manager
from random import shuffle
import itertools

import numpy as np

from .utils import pickle_dump, pickle_load
from unet3d.augment import scale_affine
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y, translate_affine, random_scale_factor
from .normalize import normalize_data


def get_generators_from_data_file(data_file, batch_size=1, validation_batch_size=1, translation_deviation=None,
                                  skip_blank=False, permute=False, normalize=True, preload_validation_data=False,
                                  scale_deviation=None, use_multiprocessing=False):
    training_generator = data_generator_from_data_file(data_file, data_file.get_training_groups(),
                                                       batch_size=batch_size, skip_blank=skip_blank, permute=permute,
                                                       translation_deviation=translation_deviation, normalize=normalize,
                                                       scale_deviation=scale_deviation,
                                                       use_multiprocessing=use_multiprocessing)
    validation_generator = data_generator_from_data_file(data_file, data_file.get_validation_groups(),
                                                         batch_size=validation_batch_size, normalize=normalize,
                                                         use_preloaded=preload_validation_data,
                                                         use_multiprocessing=use_multiprocessing)
    return training_generator, validation_generator


def data_loader(data_file, subject_ids, features_bucket, targets_bucket, batch_size, skip_blank, sleep_time=1,
                buffer_factor=3, **load_data_kwargs):
    while True:
        _subject_ids = subject_ids.tolist()
        shuffle(_subject_ids)
        while len(_subject_ids) > 0:
            if len(features_bucket) < (batch_size * buffer_factor):
                subject_id = _subject_ids.pop()
                features, targets = load_data(data_file=data_file, subject_id=subject_id, **load_data_kwargs)
                if not (skip_blank and np.all(np.equal(targets, 0))):
                    features_bucket.append(features)
                    targets_bucket.append(targets)
            else:
                sleep(sleep_time)


def load_data(data_file, subject_id, use_preloaded=False, translation_deviation=None, scale_deviation=None,
              permute=False, normalize=True):
    if use_preloaded:
        features = data_file.get_supplemental_data(subject_id, "roi_features")
        targets = data_file.get_supplemental_data(subject_id, "roi_targets")
    else:
        roi_affine, roi_shape = data_file.get_roi(subject_id)
        if translation_deviation:
            roi_affine = translate_affine(affine=roi_affine, shape=roi_shape,
                                          translation_scales=random_scale_factor(mean=0,
                                                                                 std=translation_deviation))
        if scale_deviation:
            roi_affine = scale_affine(affine=roi_affine, shape=roi_shape,
                                      scale=random_scale_factor(std=scale_deviation))
        features, targets = data_file.get_roi_data(subject_id, roi_affine=roi_affine, roi_shape=roi_shape)
        if permute:
            features, targets = random_permutation_x_y(features, targets)
    if normalize:
        features = normalize_data(features)
    return features, targets


def data_generator_from_data_file(data_file, subject_ids, batch_size=1, translation_deviation=None, skip_blank=False,
                                  permute=False, normalize=True, use_preloaded=False, scale_deviation=None,
                                  use_multiprocessing=False, sleep_time=1):
    all_subject_ids = np.copy(subject_ids)
    if use_multiprocessing:
        manager = Manager()
        features_bucket = manager.list()
        targets_bucket = manager.list()
        # start filling the buckets
        process = multiprocessing.Process(target=data_loader, kwargs=dict(data_file=data_file,
                                                                          subject_ids=all_subject_ids,
                                                                          features_bucket=features_bucket,
                                                                          targets_bucket=targets_bucket,
                                                                          batch_size=batch_size,
                                                                          sleep_time=sleep_time,
                                                                          skip_blank=skip_blank,
                                                                          use_preloaded=use_preloaded,
                                                                          normalize=normalize,
                                                                          permute=permute,
                                                                          translation_deviation=translation_deviation,
                                                                          scale_deviation=scale_deviation))
        process.start()
        while True:
            x = list()
            y = list()
            while len(x) < batch_size:
                if len(features_bucket) > 0:
                    x.append(features_bucket.pop(0))
                    y.append(targets_bucket.pop(0))
                else:
                    sleep(sleep_time)
            yield np.asarray(x), np.asarray(y)
    else:
        while True:
            x = list()
            y = list()
            subject_ids = all_subject_ids.tolist()
            shuffle(subject_ids)
            while len(subject_ids) > 0:
                subject_id = subject_ids.pop()
                features, targets = load_data(data_file=data_file, subject_id=subject_id, use_preloaded=use_preloaded,
                                              translation_deviation=translation_deviation,
                                              scale_deviation=scale_deviation, permute=permute, normalize=normalize)
                if not (skip_blank and np.all(np.equal(targets, 0))):
                    x.append(features)
                    y.append(targets)
                if len(x) >= batch_size:
                    yield np.asarray(x), np.asarray(y)
                    x = list()
                    y = list()


def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment_flip=True,
                                           augment_distortion_factor=0.25, patch_shape=None, validation_patch_overlap=0,
                                           training_patch_start_offset=None, validation_batch_size=None,
                                           skip_blank=True, permute=False, training_indices=None,
                                           validation_indices=None, weights=None, noise_factor=None,
                                           background_correction=False, augment_translation_deviation=None):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    if training_indices is None or validation_indices is None:
        training_indices, validation_indices = get_validation_split(data_file,
                                                                    data_split=data_split,
                                                                    overwrite=overwrite,
                                                                    training_file=training_keys_file,
                                                                    validation_file=validation_keys_file)

    training_generator = data_generator(data_file, training_indices,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,
                                        weights=weights,
                                        noise_factor=noise_factor,
                                        background_correction=background_correction,
                                        augment_translation_deviation=augment_translation_deviation)
    validation_generator = data_generator(data_file, validation_indices,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          weights=weights)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_indices, patch_shape,
                                                                   skip_blank=skip_blank,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_indices, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False, weights=None, noise_factor=None,
                   background_correction=False, augment_translation_deviation=None):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        weight_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
                     skip_blank=skip_blank, permute=permute, noise_factor=noise_factor,
                     background_correction=background_correction, translation_deviation=augment_translation_deviation)
            if weights is not None:
                weight_list.append(weights[index])
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels, weight_list=weight_list)
                x_list = list()
                y_list = list()
                weight_list = list()


def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, data_file, index, augment_flip=False, augment_distortion_factor=None,
             patch_shape=False, skip_blank=True, permute=False, background_correction=False, noise_factor=None,
             translation_deviation=None):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    if augment_distortion_factor or augment_flip or noise_factor:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor,
                                   noise_factor=noise_factor, background_correction=background_correction,
                                   translation_deviation=translation_deviation)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None, weight_list=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    if weight_list:
        weights = np.asarray(weight_list)
        return x, y, weights
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
