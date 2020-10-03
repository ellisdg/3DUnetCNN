import numpy as np


def calculate_origin_offset(new_spacing, old_spacing):
    return np.divide(np.subtract(new_spacing, old_spacing)/2, old_spacing)


def get_spacing_from_affine(affine):
    RZS = affine[:3, :3]
    return np.sqrt(np.sum(RZS * RZS, axis=0))


def set_affine_spacing(affine, spacing):
    scale = np.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = np.diag(np.ones(4))
    np.fill_diagonal(affine_transform, list(scale) + [1])
    return np.matmul(affine, affine_transform)


def get_extent_from_image(image):
    return np.multiply(image.shape[:3], image.header.get_zooms()[:3])


def adjust_affine_spacing(affine, new_spacing, spacing=None):
    if spacing is None:
        spacing = get_spacing_from_affine(affine)
    offset = calculate_origin_offset(new_spacing, spacing)
    new_affine = np.copy(affine)
    translation_affine = np.diag(np.ones(4))
    translation_affine[:3, 3] = offset
    new_affine = np.matmul(new_affine, translation_affine)
    new_affine = set_affine_spacing(new_affine, new_spacing)
    return new_affine


def resize_affine(affine, shape, target_shape, copy=True):
    if copy:
        affine = np.copy(affine)
    scale = np.divide(shape, target_shape)
    spacing = get_spacing_from_affine(affine)
    target_spacing = np.multiply(spacing, scale)
    affine = adjust_affine_spacing(affine, target_spacing)
    return affine


def is_diag(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0


def assert_affine_is_diagonal(affine):
    if not is_diag(affine[:3, :3]):
        raise NotImplementedError("Hemisphere swapping for non-diagonal affines is not yet implemented.")
