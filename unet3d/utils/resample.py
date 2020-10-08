import numpy as np
from nilearn.image import new_img_like, resample_to_img

from fcnn.utils.affine import get_extent_from_image, adjust_affine_spacing


def pad_image(image, mode='edge', pad_width=1):
    affine = np.copy(image.affine)
    spacing = np.copy(image.header.get_zooms()[:3])
    affine[:3, 3] -= spacing * pad_width
    if len(image.shape) > 3:
        # just pad the first three dimensions
        pad_width = [[pad_width]*2]*3 + [[0, 0]]*(len(image.shape) - 3)
    data = np.pad(image.get_data(), pad_width=pad_width, mode=mode)
    return image.__class__(data, affine)


def resample_image_to_spacing(image, new_spacing, interpolation='continuous'):
    new_affine = adjust_affine_spacing(image.affine, new_spacing, spacing=image.header.get_zooms()[:3])
    new_shape = np.asarray(np.ceil(np.divide(get_extent_from_image(image), new_spacing)), dtype=np.int)
    new_data = np.zeros(new_shape)
    new_image = new_img_like(image, new_data, affine=new_affine)
    return resample_to_img(image, new_image, interpolation=interpolation)


def resample_image(source_image, target_image, interpolation="linear", pad_mode='edge', pad=False):
    if pad:
        source_image = pad_image(source_image, mode=pad_mode)
    return resample_to_img(source_image, target_image, interpolation=interpolation)


def resample(image, target_affine, target_shape, interpolation='linear', pad_mode='edge', pad=False):
    target_data = np.zeros(target_shape, dtype=image.get_data_dtype())
    target_image = image.__class__(target_data, affine=target_affine)
    return resample_image(image, target_image, interpolation=interpolation, pad_mode=pad_mode, pad=pad)
