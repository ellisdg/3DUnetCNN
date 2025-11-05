import numpy as np
from monai.transforms import GaussianSmooth
import random
from collections.abc import Iterable
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch

from .affine import get_extent_from_image, get_spacing_from_affine, assert_affine_is_diagonal
from .image import get_image
from .resample import resample, resample_to_img
from .nilearn_custom_utils.nilearn_utils import get_background_values
from .utils import copy_image


def _validate_image_has_affine(image):
    """
    Validate that an image has the 'affine' attribute.
    
    Args:
        image (Image or MetaTensor): The image object to validate
        
    Raises:
        TypeError: If the image lacks the 'affine' attribute
    """
    if not hasattr(image, 'affine'):
        raise TypeError(
            f"Input image does not have 'affine' attribute. "
            f"Expected an Image or MetaTensor object, but got {type(image)}. "
            "Ensure your transforms preserve the affine matrix metadata."
        )


def flip_image(image, axis):
    new_data = image.detach().clone()
    new_data = torch.flip(new_data, dims=axis)
    return image.make_similar(data=new_data)


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1., std=0.25):
    return torch.normal(mean, std, size=(n_dim,))


def random_boolean():
    return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None, translation_scale=None):
    if translation_scale is not None:
        image = translate_image(image, translation_scale)
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    return image


def augment_data(data, truth, affine, scale_deviation=None, flip=False, noise_factor=None, background_correction=False,
                 translation_deviation=None, interpolation="linear"):
    background = None
    if background_correction:
        background = get_background_values(data)
        data[:] -= background
    n_dim = len(truth.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    if translation_deviation:
        translation_scale = random_scale_factor(mean=0., std=translation_deviation)
    else:
        translation_scale = None
    data_list = list()
    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        copied_image = copy_image(image)
        distorted_image = distort_image(copied_image, flip_axis=flip_axis, scale_factor=scale_factor,
                                        translation_scale=translation_scale)
        resampled_image = resample_to_img(source_image=distorted_image, target_image=image, interpolation=interpolation)
        data_list.append(resampled_image)
    data = torch.stack(data_list)  # Use torch.stack to preserve Image/MetaTensor properties
    if background_correction and background is not None:
        data = data + background
    if noise_factor is not None:
        data = add_noise(data, sigma_factor=noise_factor)
    truth_image = get_image(truth, affine)
    copied_truth_image = copy_image(truth_image)
    distorted_truth = distort_image(copied_truth_image, flip_axis=flip_axis, scale_factor=scale_factor,
                                    translation_scale=translation_scale)
    resampled_truth = resample_to_img(distorted_truth, truth_image, interpolation="nearest")
    truth_data = resampled_truth
    return data, truth_data


def generate_permutation_keys():
    """
    Return 48 unique keys representing all rotations & reflections of a 3D matrix.

    Each key is a tuple: ((rotate_x, rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    and is used by `permute_data` to realize the transform. For backward compatibility,
    a key whose rotation tuple has only 2 elements is interpreted as (0, ry, rz).

    Implementation detail: we enumerate a superset of candidates using rotations
    k in {0,1,2,3} for all three axes and all flips. We then filter keys down to
    those that actually yield unique permutations on a canonical tensor. We set
    transpose=0 for all generated keys (not needed to span the group) and dedupe
    by realized output. This guarantees exactly 48 unique transforms.
    """
    # Canonical cubic tensor as in tests: shape (1, 4, 4, 4)
    base = torch.arange(4 * 4 * 4, dtype=torch.int64).reshape(1, 4, 4, 4)

    unique_signatures = {}
    unique_keys = []

    for rx in range(4):
        for ry in range(4):
            for rz in range(4):
                for fx in (0, 1):
                    for fy in (0, 1):
                        for fz in (0, 1):
                            transpose = 0
                            key = ((rx, ry, rz), fx, fy, fz, transpose)
                            permuted = permute_data(base, key)
                            sig = tuple(permuted.reshape(-1).tolist())
                            if sig not in unique_signatures:
                                unique_signatures[sig] = key
                                unique_keys.append(key)
                                if len(unique_keys) == 48:
                                    return set(unique_keys)

    return set(unique_keys)


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data,
                 key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = data.detach().clone()
    rotation_tuple, flip_x, flip_y, flip_z, transpose = key

    # Backward compatibility: accept 2-tuple (ry, rz) and treat rx=0
    if len(rotation_tuple) == 2:
        rotate_x = 0
        rotate_y, rotate_z = rotation_tuple
    else:
        rotate_x, rotate_y, rotate_z = rotation_tuple

    # Apply rotations: Rx (about x: dims (2,3)), Ry (about y: dims (1,3)), Rz (about z: dims (1,2))
    if rotate_x != 0:
        data = torch.rot90(data, rotate_x, dims=(2, 3))
    if rotate_y != 0:
        data = torch.rot90(data, rotate_y, dims=(1, 3))
    if rotate_z != 0:
        data = torch.rot90(data, rotate_z, dims=(1, 2))

    # Flips
    if flip_x:
        data = torch.flip(data, dims=(1,))
    if flip_y:
        data = torch.flip(data, dims=(2,))
    if flip_z:
        data = torch.flip(data, dims=(3,))

    # Transpose (swap last two spatial axes) - retained for backward compatibility
    if transpose:
        data = data.transpose(2, 3)

    return data


def random_permutation_x_y(x_data, y_data, channel_axis=0):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_labels, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_labels, x, y, z).
    :param channel_axis: if the channels are not in the first axis of the array (channel_axis != 0) then the channel
    axis will be moved to the first position for permutation and then moved back to the original position.
    :return: the permuted data
    """
    key = random_permutation_key()
    if channel_axis != 0:
        return [torch.moveaxis(permute_data(torch.moveaxis(data, channel_axis, 0), key), 0, channel_axis)
                for data in (x_data, y_data)]
    else:
        return permute_data(x_data, key), permute_data(y_data, key)


def reverse_permute_data(data, key):
    """
    Reverse a permutation applied by `permute_data` without converting to numpy.
    This function works with torch.Tensor or MONAI MetaTensor and avoids
    detaching/cloning which can drop MetaTensor metadata.
    """
    if not torch.is_tensor(data):
        raise TypeError("reverse_permute_data expects a torch.Tensor or MONAI MetaTensor")

    # Compute inverse key (rotation values will be the modular inverse mod 4)
    (rotate_x, rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = reverse_permutation_key(key)

    x = data

    # Reverse transpose first (if original permute transposed at the end)
    if transpose:
        x = x.transpose(2, 3)

    # Reverse flips (same axes as applied originally)
    if flip_z:
        x = torch.flip(x, dims=(3,))
    if flip_y:
        x = torch.flip(x, dims=(2,))
    if flip_x:
        x = torch.flip(x, dims=(1,))

    # Reverse rotations (apply inverse rotation amounts)
    if rotate_z != 0:
        x = torch.rot90(x, rotate_z, dims=(1, 2))
    if rotate_y != 0:
        x = torch.rot90(x, rotate_y, dims=(1, 3))
    if rotate_x != 0:
        x = torch.rot90(x, rotate_x, dims=(2, 3))

    return x


def reverse_permutation_key(key):
    rotation = key[0]
    if len(rotation) == 2:
        # Backward compatible inverse: only (ry, rz) provided
        ry, rz = rotation
        rotation_inv = (0, -ry, -rz)
    else:
        rx, ry, rz = rotation
        rotation_inv = (-rx, -ry, -rz)
    return rotation_inv, key[1], key[2], key[3], key[4]


def add_noise(data, mean=0., sigma_factor=0.1):
    """
    Adds Gaussian noise.
    :param data: input numpy array
    :param mean: mean of the additive noise
    :param sigma_factor: standard deviation of the image will be multiplied by sigma_factor to obtain the standard
    deviation of the additive noise. Assumes standard deviation is the same for all channels.
    :return:
    """
    sigma = torch.abs(torch.multiply(torch.std(data), sigma_factor))
    noise = torch.normal(mean, sigma, size=data.shape)
    return torch.add(data, noise)


def translate_affine(affine, shape, translation_scales, copy=True):
    """
    :param translation_scales: (tuple) Contains x, y, and z translations scales from -1 to 1. 0 is no translation.
    1 is a forward (RAS-wise) translation of the entire image extent for that direction. -1 is a translation in the
    negative direction of the entire image extent. A translation of 1 is impractical for most purposes, though, as it
    moves the image out of the original field of view almost entirely. To perform a random translation, you can
    use numpy.random.normal(loc=0, scale=sigma, size=3) where sigma is the percent of image translation that would be
    randomly translated on average (0.05 for example).
    :return: affine
    """
    if copy:
        affine = affine.detach().clone()
    spacing = get_spacing_from_affine(affine)
    extent = torch.multiply(shape, spacing)
    translation = torch.multiply(translation_scales, extent)
    affine[:3, 3] = affine[:3, 3] + translation
    return affine


def translate_image(image, translation_scales, interpolation="linear"):
    """
    :param image: (NiBabel-like image)
    :param translation_scales: (tuple) Contains x, y, and z translations scales from -1 to 1. 0 is no translation.
    1 is a forward (RAS-wise) translation of the entire image extent for that direction. -1 is a translation in the
    negative direction of the entire image extent. A translation of 1 is impractical for most purposes, though, as it
    moves the image out of the original field of view almost entirely. To perform a random translation, you can
    use numpy.random.normal(loc=0, scale=sigma, size=3) where sigma is the percent of image translation that would be
    randomly translated on average (0.05 for example).
    :return: translated image
    """
    _validate_image_has_affine(image)
    affine = image.affine.detach().clone()
    translation = torch.multiply(translation_scales, get_extent_from_image(image))
    affine[:3, 3] = affine[:3, 3] + translation
    return resample(image, target_affine=affine, target_shape=image.shape, interpolation=interpolation)


def _rotate_affine(affine, shape, rotation):
    """
    Work in progress. Does not work yet.
    :param affine:
    :param shape:
    :param rotation:
    :return:
    """
    assert_affine_is_diagonal(affine)
    # center the image on (0, 0, 0)
    temp_origin = (affine.diagonal()[:3] * torch.tensor(shape)) / 2
    temp_affine = affine.detach().clone()
    temp_affine[:, :3] = temp_origin

    rotation_affine = torch.diag(torch.ones(4))
    theta_x, theta_y, theta_z = rotation
    affine_x = rotation_affine.detach().clone()
    affine_x[1, 1] = torch.cos(theta_x)
    affine_x[1, 2] = -torch.sin(theta_x)
    affine_x[2, 1] = torch.sin(theta_x)
    affine_x[2, 2] = torch.cos(theta_x)
    x_rotated_affine = torch.matmul(affine, affine_x)
    new_affine = x_rotated_affine.detach().clone()
    new_affine[:, :3] = affine[:, :3]
    return new_affine


def find_image_center(image, ndim=3):
    _validate_image_has_affine(image)
    return find_center(image.affine, image.shape, ndim=ndim)


def find_center(affine, shape, ndim=3):
    center_voxel = torch.divide(torch.tensor(shape[:ndim]), 2)
    return torch.matmul(affine, torch.cat((center_voxel, torch.ones(1))))[:ndim]


def scale_image(image, scale, ndim=3, interpolation='linear'):
    _validate_image_has_affine(image)
    affine = scale_affine(image.affine, image.shape, scale=scale, ndim=ndim)
    return resample(image, affine, image.shape, interpolation=interpolation)


def scale_affine(affine, shape, scale, ndim=3):
    """
    This assumes that the shape stays the same.
    :param affine: affine matrix for the image.
    :param shape: current shape of the data. This will remain the same.
    :param scale: iterable with length ndim, int, or float. A scale greater than 1 indicates the image will be zoomed,
    the spacing will get smaller, and the affine window will be smaller as well. A scale of less than 1 indicates
    zooming out with the spacing getting larger and the affine window getting bigger.
    :param ndim: number of dimensions (default is 3).
    :return:
    """
    if not isinstance(scale, Iterable):
        scale = torch.ones(ndim) * scale

    # 1. find the image center
    center = find_center(affine, shape, ndim=ndim)

    # 2. translate the affine
    affine = affine.detach().clone()
    origin = affine[:ndim, ndim]
    t = torch.diag(torch.ones(ndim + 1))
    t[:ndim, ndim] = (center - origin) * (1 - 1 / scale)
    affine = torch.matmul(t, affine)

    # 3. scale the affine
    s = torch.diag(torch.cat(((1 / scale), torch.ones(1))))
    affine = torch.matmul(affine, s)
    return affine


def elastic_transform(image, alpha, sigma, target_image, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Modified from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y, z, c = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), torch.arange(shape[3]),
                             indexing="ij")
    indices = torch.reshape(x+dx, (-1, 1)), torch.reshape(y+dy, (-1, 1)), torch.reshape(z+dz, (-1, 1)), torch.reshape(c, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_target_image = map_coordinates(target_image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape), distored_target_image.reshape(image.shape)


def smooth_img(image, fwhm):
    _validate_image_has_affine(image)
    sigma = torch.divide(fwhm, get_spacing_from_affine(image.affine))
    array = GaussianSmooth(sigma=sigma)(image)
    return image.make_similar(array)


def random_blur(image, mean, std):
    """
    mean: mean fwhm in millimeters.
    std: standard deviation of fwhm in millimeters.
    """
    return smooth_img(image, fwhm=torch.abs(torch.normal(mean, std, size=(3,))))


def affine_swap_axis(affine, shape, axis=0):
    """FROM MONAI FLIP"""
    mat = torch.as_tensor(torch.eye(len(affine)), dtype=affine.dtype)
    sp = axis - 1
    mat[sp, sp], mat[sp, -1] = mat[sp, sp] * -1, shape[axis] - 1
    return affine @ mat
