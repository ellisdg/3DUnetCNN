import numpy as np
import torch


def calculate_origin_offset(new_spacing, old_spacing):
    return torch.divide(torch.divide(torch.subtract(new_spacing,
                                                    old_spacing),
                                     2),
                        old_spacing)


def get_spacing_from_affine(affine):
    RZS = affine[:3, :3]
    return torch.sqrt(torch.sum(torch.multiply(RZS, RZS), dim=0))


def set_affine_spacing(affine, spacing):
    scale = torch.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = torch.diag(torch.ones(4, dtype=affine.dtype, device=affine.device))
    affine_transform.diagonal().copy_(torch.cat((scale, torch.ones(1, dtype=affine.dtype, device=affine.device))))
    return torch.matmul(affine, affine_transform)


def get_extent_from_image(image):
    # Use last 3 dimensions as spatial dims to support channel-first tensors
    spatial_shape = torch.as_tensor(image.shape[-3:], device=image.device)
    return torch.multiply(spatial_shape, get_spacing_from_affine(image.affine))


def adjust_affine_spacing(affine, new_spacing, spacing=None):
    if spacing is None:
        spacing = get_spacing_from_affine(affine)
    offset = calculate_origin_offset(new_spacing, spacing)
    new_affine = affine.detach().clone()
    translation_affine = torch.diag(torch.ones(4, dtype=affine.dtype, device=affine.device))
    translation_affine[:3, 3] = offset
    new_affine = torch.matmul(new_affine, translation_affine)
    new_affine = set_affine_spacing(new_affine, new_spacing)
    return new_affine


def _shapes_equal(shape, target_shape):
    try:
        a = torch.as_tensor(shape)
        b = torch.as_tensor(target_shape)
        return bool(torch.all(a == b))
    except Exception:
        return shape == target_shape


def resize_affine(affine, shape, target_shape, copy=True):
    if not _shapes_equal(shape, target_shape):
        if copy:
            affine = affine.detach().clone()
        spacing = get_spacing_from_affine(affine)
        scale = torch.divide(
            torch.as_tensor(shape, dtype=spacing.dtype, device=affine.device),
            torch.as_tensor(target_shape, dtype=spacing.dtype, device=affine.device),
        )
        target_spacing = torch.multiply(spacing, scale)
        affine = adjust_affine_spacing(affine, target_spacing)
    return affine


def is_diag(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0


def assert_affine_is_diagonal(affine):
    if not is_diag(affine[:3, :3]):
        raise NotImplementedError("Hemisphere swapping for non-diagonal affines is not yet implemented.")
