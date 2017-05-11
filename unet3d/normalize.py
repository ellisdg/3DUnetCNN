import os

import nibabel as nib
import numpy as np
from nilearn.image import resample_img, reorder_img, new_img_like

from unet3d.utils import crop_img, crop_img_to


def find_downsized_info(subject_folders, input_shape):
    foreground = get_complete_foreground(subject_folders)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_complete_foreground(subject_folders):
    for i, subject_folder in enumerate(subject_folders):
        background_path = os.path.join(subject_folder, "background.nii.gz")
        image = nib.load(background_path)
        image_foreground = image.get_data() == 0
        if i == 0:
            foreground = image_foreground
            reference_image = image
        else:
            foreground[image_foreground] = 1

    return new_img_like(reference_image, foreground)


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
    std = np.asarray(means).std(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


def resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)
