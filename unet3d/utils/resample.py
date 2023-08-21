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
    data = rep_pad(image)
    return image.make_similar(data, affine)


def resample_image_to_spacing(image, new_spacing, interpolation='bilinear'):
    new_affine = adjust_affine_spacing(image.affine, new_spacing, spacing=image.header.get_zooms()[:3])
    new_shape = np.asarray(np.ceil(np.divide(get_extent_from_image(image), new_spacing)), dtype=int)
    new_data = torch.zeros(new_shape)
    new_image = image.make_similar(image, new_data, affine=new_affine)
    return resample_to_img(image, new_image, interpolation=interpolation)


def resample_image(source_image, target_image, interpolation="trilinear", pad=False):
    if pad:
        source_image = pad_image(source_image)
    return resample_to_img(source_image, target_image, interpolation=interpolation)


def resample(image, target_affine, target_shape, interpolation='trilinear', pad=False, dtype=None, align_corners=True,
             margin=1e-6):
    """
    Resample an image to a target affine and shape.
    :param image: Image to resample.
    :param target_affine: Target affine.
    :param target_shape: Target shape.
    :param interpolation: Interpolation mode.
    :param pad: not implemented
    :param dtype: output data type
    :param align_corners: align_corners parameter for SpatialResample
    :param margin: margin for equality check
    """
    if dtype:
        image = image.to(dtype)
    if (torch.all(torch.abs(image.affine - target_affine) < margin)
            and torch.all(torch.tensor(image.shape[-3:]) == torch.tensor(target_shape))):
        return image
    mode = monai_interpolation_mode(interpolation)
    resampler = SpatialResample(mode=mode, align_corners=align_corners)

    return resampler(img=image, dst_affine=target_affine, spatial_size=target_shape)


def monai_interpolation_mode(interpolation):
    if interpolation == "linear":
        return "bilinear"
    elif interpolation.isdigit():
        return int(interpolation)
    return interpolation


def resample_to_img(source_image, target_image, interpolation='trilinear', align_corners=True):
    return resample(source_image, target_image.affine, target_image.shape[1:], interpolation=interpolation,
                    align_corners=align_corners)
