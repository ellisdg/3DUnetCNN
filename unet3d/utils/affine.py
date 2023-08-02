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
    affine_transform = torch.diag(torch.ones(4))
    affine_transform.diagonal().copy_(torch.cat((scale, torch.ones(1))))
    return torch.matmul(affine, affine_transform)


def get_extent_from_image(image):
    return torch.multiply(image.shape[:3], get_spacing_from_affine(image.affine))


def adjust_affine_spacing(affine, new_spacing, spacing=None):
    if spacing is None:
        spacing = get_spacing_from_affine(affine)
    offset = calculate_origin_offset(new_spacing, spacing)
    new_affine = affine.detach().clone()
    translation_affine = torch.diag(torch.ones(4))
    translation_affine[:3, 3] = offset
    new_affine = torch.matmul(new_affine, translation_affine)
    new_affine = set_affine_spacing(new_affine, new_spacing)
    return new_affine


def resize_affine(affine, shape, target_shape, copy=True):
    if not np.all(np.equal(shape, target_shape)):
        if copy:
            affine = affine.detach().clone()
        scale = torch.divide(torch.as_tensor(shape), torch.as_tensor(target_shape))
        spacing = get_spacing_from_affine(affine)
        target_spacing = torch.multiply(spacing, scale)
        affine = adjust_affine_spacing(affine, target_spacing)
    return affine


def is_diag(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0


def assert_affine_is_diagonal(affine):
    if not is_diag(affine[:3, :3]):
        raise NotImplementedError("Hemisphere swapping for non-diagonal affines is not yet implemented.")
