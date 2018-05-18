import os

import numpy as np
from nilearn.image import new_img_like

from .utils.utils import resize, read_image_files, crop_img_to, read_image
from .utils.nilearn_custom_utils.nilearn_utils import run_with_background_correction, crop_img


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files, background_correction=False, percentile=None, pad=False):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files, background_correction=background_correction,
                                             percentile=percentile)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True, percentile=percentile,
                                                      background_correction=background_correction)
    return crop_img(foreground, return_slices=True, copy=True, pad=pad)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False,
                      background_correction=False, percentile=None):
    if crop:
        crop_slices = get_cropping_parameters([in_files], background_correction=background_correction,
                                              percentile=percentile)
    else:
        crop_slices = None
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices,
                              background_correction=background_correction)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files, background_correction=False, percentile=None):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files, background_correction=background_correction,
                                                              percentile=percentile)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False,
                                     background_correction=False, percentile=None):
    foreground = None
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        foreground = get_image_foreground(image, background_value=background_value, tolerance=tolerance,
                                          array=foreground, background_correction=background_correction,
                                          percentile=percentile)
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground


def get_image_foreground(image, background_value=0, tolerance=1e-5, array=None, background_correction=False,
                         percentile=None):
    if background_correction:
        return run_with_background_correction(get_image_foreground, image, background_value=background_value,
                                              tolerance=tolerance, array=array, background_correction=False,
                                              returns_array=True, reset_background=False, percentile=percentile)
    else:
        if percentile:
            is_foreground = image.get_data() > np.percentile(image.get_data(), percentile)
        else:
            is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                          image.get_data() > (background_value + tolerance))
        if array is None:
            array = np.zeros(is_foreground.shape, dtype=np.uint8)

        array[is_foreground] = 1
        return array


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        mean = data.mean(axis=(1, 2, 3))
        std = data.std(axis=(1, 2, 3))
        data_storage[index] = normalize_data(data, mean, std)
    return data_storage


