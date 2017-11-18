import os
import collections

import numpy as np
from nilearn.image import new_img_like

from unet3d.utils.utils import resize
from .utils import crop_img, crop_img_to, read_image


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    foreground = get_complete_foreground(in_files)
    return crop_img(foreground, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None):
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]

    crop_slices = get_cropping_parameters([in_files])
    images = list()
    for index, in_file in enumerate(in_files):
        interpolation = "continuous"
        if index in label_indices:
            interpolation = "nearest"
        images.append(read_image(in_file, image_shape=image_shape, crop=crop_slices, interpolation=interpolation))
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001):
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1

    return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


