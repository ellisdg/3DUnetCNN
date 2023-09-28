import sys

import nibabel as nib
import numpy as np
import json

import torch

from monai.transforms import Orientation

from .image import Image


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def dump_json(dataobj, filename):
    with open(filename, 'w') as opened_file:
        json.dump(dataobj, opened_file, indent=4)


def logical_and(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = torch.logical_and(array, other_array)
    return array


def logical_or(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = torch.logical_or(array, other_array)
    return array


def get_index_value(iterable, index):
    if iterable:
        return iterable[index]


def extract_polydata_vertices(polydata):
    return torch.as_tensor([polydata.GetPoint(index) for index in range(polydata.GetNumberOfPoints())])


def copy_image(image):
    return image.detach.clone()


def update_progress(progress, bar_length=30, message=""):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = "\r{0}[{1}] {2:.2f}% {3}".format(message, "#" * block + "-" * (bar_length - block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def combine_images(images, axis=0):
    """
    After testing the MetaTensor from MONAI, it looks like the first affine is kept and the other affines are discarded
    when doing torch.cat. It might be helpful to check that the affines are equal before combining them, just for
    sanity purposes.
    TODO: add optional affine check to make sure all the affines are the same
    """
    return torch.cat(images, axis)


def move_channels_last(data):
    return torch.moveaxis(data, 0, -1)


def move_channels_first(data):
    return torch.moveaxis(data, -1, 0)


def load_image(filename, feature_axis=0, reorder=True, dtype=None, verbose=False, axcodes="RAS"):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    (for now, anything other than 0, probably won't work).
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """
    if type(filename) == list:
        return combine_images([load_single_image(fn, reorder=reorder, dtype=dtype, verbose=verbose, axcodes=axcodes)
                               for fn in filename], axis=feature_axis)
    else:
        return load_single_image(filename, reorder=reorder, dtype=dtype, verbose=verbose, axcodes=axcodes)


def load_single_image(filename, reorder=True, dtype=None, verbose=False, axcodes="RAS"):
    if verbose:
        print("Loading", filename)
    nib_image = nib.load(filename)
    np_data = np.asarray(nib_image.dataobj)
    if np_data.dtype == np.uint16:
        np_data = np.asarray(np_data, dtype=np.int16)
    nib_data = torch.from_numpy(np_data)
    if len(nib_data.shape) > 3:
        # nibabel loads 4d data with channels in last dimension
        # change that to the first dimension
        data = torch.moveaxis(nib_data, -1, 0)
    else:
        data = nib_data[None]  # Set channels shape to 1
    image = Image(x=data, affine=torch.from_numpy(nib_image.affine),
                  meta={"source_filename": filename})
    if verbose:
        print("Finished loading", filename, "Shape:", image.shape)
    if dtype is not None:
        image.to(dtype)
    if reorder:
        return reorder_image(image, axcodes=axcodes)
    return image


def reorder_image(image, axcodes="RAS"):
    return Orientation(axcodes=axcodes)(image)


def extract_sub_volumes(image, sub_volume_indices):
    return image[sub_volume_indices]


def break_down_volume_into_half_size_volumes(data):
    return (data[::2, ::2, ::2],  # original
            data[1::2, ::2, ::2],  # x shifted
            data[1::2, 1::2, ::2],  # x and y shifted
            data[1::2, ::2, 1::2],  # x and z shifted
            data[1::2, 1::2, 1::2],  # x, y, and z shifted
            data[::2, 1::2, ::2],  # y shifted
            data[::2, 1::2, 1::2],  # y and z shifted
            data[::2, ::2, 1::2])  # z shifted


def combine_half_size_volumes(volumes):
    data = np.zeros(tuple(np.asarray(volumes[0].shape[:3]) * 2) + volumes[0].shape[3:], dtype=volumes[0].dtype)
    data[::2, ::2, ::2] = volumes[0]  # original
    data[1::2, ::2, ::2] = volumes[1]  # x shifted
    data[1::2, 1::2, ::2] = volumes[2]  # x and y shifted
    data[1::2, ::2, 1::2] = volumes[3]  # x and z shifted
    data[1::2, 1::2, 1::2] = volumes[4]  # x, y, and z shifted
    data[::2, 1::2, ::2] = volumes[5]  # y shifted
    data[::2, 1::2, 1::2] = volumes[6]  # y and z shifted
    data[::2, ::2, 1::2] = volumes[7]  # z shifted
    return data


def split_left_right(data):
    center_index = data.shape[0] // 2
    left = np.zeros(data.shape, dtype=data.dtype)
    right = np.copy(left)
    left[:center_index] = data[:center_index]
    right[center_index:] = data[center_index:]
    return left, right


def get_class(_dict, _module, _class_key="name"):
    return getattr(_module, _dict[_class_key])


def get_kwargs(_dict, skip_keys=("name",)):
    kwargs = dict()
    for key, value in _dict.items():
        if key not in skip_keys:
            kwargs[key] = value
    return kwargs
