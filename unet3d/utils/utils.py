import pickle
import os

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, resample_img

from .nilearn_custom_utils.nilearn_utils import crop_img_to


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, use_nearest_for_last_file=True):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    image_list = list()
    for i, image_file in enumerate(image_files):
        if (i + 1) == len(image_files) and use_nearest_for_last_file:
            interpolation = "nearest"
        else:
            interpolation = "continuous"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return np.stack([image.get_data() for image in image_list])


def read_image(in_file, image_shape=None, interpolation='continuous', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)
