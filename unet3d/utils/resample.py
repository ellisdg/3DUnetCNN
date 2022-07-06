import torch
from torch.nn import ReplicationPad3d
import numpy as np
from monai.transforms import SpatialResample

from unet3d.utils.affine import get_extent_from_image, adjust_affine_spacing, get_spacing_from_affine


def pad_image(image, pad_width=1):
    affine = image.affine
    spacing = get_spacing_from_affine(affine)
    affine[:3, 3] -= spacing * pad_width
    rep_pad = ReplicationPad3d(pad_width)
    data = rep_pad(image.get_data())
    return image.make_similar(data, affine)


def resample_image_to_spacing(image, new_spacing, interpolation='continuous'):
    new_affine = adjust_affine_spacing(image.affine, new_spacing, spacing=image.header.get_zooms()[:3])
    new_shape = np.asarray(np.ceil(np.divide(get_extent_from_image(image), new_spacing)), dtype=int)
    new_data = torch.zeros(new_shape)
    new_image = image.make_similar(image, new_data, affine=new_affine)
    return resample_to_img(image, new_image, interpolation=interpolation)


def resample_image(source_image, target_image, interpolation="linear", pad=False):
    if pad:
        source_image = pad_image(source_image)
    return resample_to_img(source_image, target_image, interpolation=interpolation)


def resample(image, target_affine, target_shape, interpolation='linear', pad=False, dtype=None, align_corners=True):
    resampler = SpatialResample(mode=monai_interpolation_mode(interpolation), align_corners=align_corners)

    print(image.shape, target_affine, target_shape)
    data = image.get_data()

    if dtype:
        data = data.to(dtype)
    array, affine = resampler(img=data, src_affine=image.affine,
                              dst_affine=target_affine, spatial_size=target_shape)
    return image.make_similar(array, affine)


def monai_interpolation_mode(interpolation):
    if interpolation == "linear":
        return "bilinear"
    return interpolation


def resample_to_img(source_image, target_image, interpolation='linear', align_corners=True):
    return resample(source_image, target_image.affine, target_image.shape[1:], interpolation=interpolation,
                    align_corners=align_corners, pad=False)
