import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
from nilearn.image.resampling import BoundingBoxError
import random
import itertools

from .utils.utils import get_spacing_from_affine, set_affine_spacing
from .utils.nilearn_custom_utils.nilearn_utils import get_background_values


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1., std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None, translation_scale=None):
    if translation_scale is not None:
        image = translate_image(image, translation_scale, copy=False)
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    return image


def augment_data(data, truth, affine, scale_deviation=None, flip=False, noise_factor=None, background_correction=False,
                 translation_deviation=None, interpolation="linear"):
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
        try:
            resampled_image = resample_to_img(source_img=distorted_image, target_img=image, interpolation=interpolation)
        except BoundingBoxError:
            resampled_image = distorted_image
        data_list.append(resampled_image.get_data())
    data = np.asarray(data_list)
    if background_correction:
        data[:] += background
    if noise_factor is not None:
        data = add_noise(data, sigma_factor=noise_factor)
    truth_image = get_image(truth, affine)
    copied_truth_image = copy_image(truth_image)
    distorted_truth = distort_image(copied_truth_image, flip_axis=flip_axis, scale_factor=scale_factor,
                                    translation_scale=translation_scale)
    try:
        resampled_truth = resample_to_img(distorted_truth, truth_image, interpolation="nearest")
    except BoundingBoxError:
        resampled_truth = distorted_truth
    truth_data = resampled_truth.get_data()
    return data, truth_data


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def reverse_permute_data(data, key):
    key = reverse_permutation_key(key)
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    if flip_z:
        data = data[:, :, :, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, ::-1]
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    return data


def reverse_permutation_key(key):
    rotation = tuple([-rotate for rotate in key[0]])
    return rotation, key[1], key[2], key[3], key[4]


def add_noise(data, mean=0., sigma_factor=0.1):
    """
    Adds Gaussian noise.
    :param data: input numpy array
    :param mean: mean of the additive noise
    :param sigma_factor: standard deviation of the image will be multiplied by sigma_factor to obtain the standard
    deviation of the additive noise. Assumes standard deviation is the same for all channels.
    :return:
    """
    sigma = np.std(data) * sigma_factor
    noise = np.random.normal(mean, sigma, data.shape)
    return np.add(data, noise)


def copy_image(image):
    return image.__class__(np.copy(image.get_data()), image.affine)


def get_extent_from_image(image):
    return np.multiply(image.shape, image.header.get_zooms())


def translate_image(image, translation_scales, copy=False):
    """

    :param image: (NiBabel-like image)
    :param translation_scales: (tuple) Contains x, y, and z translations on scales from -1 to 1. 0 is no translation.
    1 is a forward (RAS-wise) translation of the entire image extent for that direction. -1 is a translation in the
    negative direction of the entire image extent. A translation of 1 is impractical for most purposes, though, as it
    moves the image out of the original field of view almost entirely.
    :return:
    """
    if copy:
        image = copy_image(image)
    translation = np.multiply(translation_scales, get_extent_from_image(image))
    image.affine[:3, 3] += translation
    return image


def translate_affine(affine, shape, translation_scales, copy=True):
    if copy:
        affine = np.copy(affine)
    spacing = get_spacing_from_affine(affine)
    extent = np.multiply(shape, spacing)
    translation = np.multiply(translation_scales, extent)
    affine[:3, 3] += translation
    return affine


def scale_affine(affine, shape, scale, copy=True):
    """
    Scales the affine (while keeping the shape the same) and then adjusts the origin so that the center of the image
     remains the same. This will change the spacing of the affine. This function might not work with non-RAS images.
    :param affine: Affine matrix to scale
    :param shape: Shape of the image/region
    :param scale: How to scale the affine
    :param copy: copies the affine matrix before modifying it
    :return: Modified affine matrix
    """
    if copy:
        affine = np.copy(affine)
    spacing = get_spacing_from_affine(affine)
    extent = np.multiply(spacing, shape)
    new_spacing = np.multiply(scale, spacing)
    new_extent = np.multiply(new_spacing, shape)
    direction = np.sign(np.diagonal(affine)[:3])
    affine[:3, 3] += np.subtract(extent, new_extent)/2 * direction
    affine = set_affine_spacing(affine, new_spacing)
    return affine


def rotate_affine(affine, rotation):
    rotation_affine = np.diag(np.ones(4))
    theta_x, theta_y, theta_z = rotation
    affine_x = np.copy(rotation_affine)
    affine_x[1, 1] = np.cos(theta_x)
    affine_x[1, 2] = -np.sin(theta_x)
    affine_x[2, 1] = np.sin(theta_x)
    affine_x[2, 2] = np.sin(theta_x)

    return affine * affine_x
