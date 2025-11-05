import torch
from torch.nn import ReplicationPad3d
from monai.transforms import SpatialResample

from unet3d.utils.affine import get_extent_from_image, adjust_affine_spacing, get_spacing_from_affine


def pad_image(image, pad_width=1):
    affine = image.affine
    spacing = get_spacing_from_affine(affine)
    affine = affine.clone()
    affine[:3, 3] -= spacing * pad_width
    rep_pad = ReplicationPad3d(pad_width)
    # Ensure 5D input to ReplicationPad3d: (N, C, D, H, W)
    x = image
    added_batch = False
    if x.ndim == 4:
        x = x.unsqueeze(0)
        added_batch = True
    data = rep_pad(x)
    if added_batch:
        data = data.squeeze(0)
    return image.make_similar(data, affine)


def resample_image_to_spacing(image, new_spacing, interpolation='bilinear'):
    # Compute spacing from affine to support MetaTensor input
    current_spacing = get_spacing_from_affine(image.affine)
    new_spacing_t = torch.as_tensor(new_spacing, dtype=current_spacing.dtype, device=current_spacing.device)
    new_affine = adjust_affine_spacing(image.affine, new_spacing_t, spacing=current_spacing)
    extent = get_extent_from_image(image)
    # Robust shape: floor with epsilon to avoid float drift causing +1
    eps = torch.finfo(current_spacing.dtype).eps if torch.is_floating_point(current_spacing) else 1e-7
    new_shape_t = torch.floor(extent / new_spacing_t + eps).to(torch.int64)
    d, h, w = int(new_shape_t[0].item()), int(new_shape_t[1].item()), int(new_shape_t[2].item())
    # Prepend channel dimension
    channel_dim = int(image.shape[0]) if image.ndim == 4 else 1
    new_data = torch.zeros((channel_dim, d, h, w), dtype=image.dtype)
    new_image = image.make_similar(new_data, affine=new_affine)
    # Use align_corners=False for downsampling consistency with expected intensity scaling
    return resample_to_img(image, new_image, interpolation=interpolation, align_corners=False)


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
    resampler = SpatialResample(mode=mode, align_corners=align_corners, padding_mode="zeros")

    return resampler(img=image, dst_affine=target_affine, spatial_size=target_shape)


def monai_interpolation_mode(interpolation):
    if interpolation in ("linear", "trilinear"):
        return "bilinear"
    elif isinstance(interpolation, str) and interpolation.isdigit():
        return int(interpolation)
    return interpolation


def resample_to_img(source_image, target_image, interpolation='trilinear', align_corners=True):
    return resample(source_image, target_image.affine, target_image.shape[1:], interpolation=interpolation,
                    align_corners=align_corners)
