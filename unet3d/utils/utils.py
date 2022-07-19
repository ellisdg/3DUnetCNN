import sys

import nibabel as nib
import numpy as np
import json
from scipy.ndimage import binary_erosion

import torch

from monai.transforms import Orientation

from .image import Image


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def dump_json(dataobj, filename):
    with open(filename, 'w') as opened_file:
        json.dump(dataobj, opened_file)


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


def compile_one_hot_encoding(data, n_labels, labels=None, dtype=torch.uint8, return_4d=True):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :param dtype: output type of the array
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    while len(data.shape) < 5:
        data = data[None]
    assert data.shape[1] == 1
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = torch.zeros(new_shape, dtype=dtype)
    for label_index in range(n_labels):
        if labels is not None:
            if type(labels[label_index]) == list:
                # lists of labels will group together multiple labels from the label map into a single one-hot channel.
                for label in labels[label_index]:
                    # iterate through each of the labels in the sub list
                    y[:, label_index][data[:, 0] == label] = 1
            else:
                y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    if return_4d:
        assert y.shape[0] == 1
        y = y[0]
    return y


def convert_one_hot_to_label_map(one_hot_encoding, labels, axis=0, threshold=0.5, sum_then_threshold=False,
                                 dtype=torch.int16, label_hierarchy=False):
    if label_hierarchy:
        return convert_one_hot_to_label_map_using_hierarchy(one_hot_encoding, labels, axis=axis, threshold=threshold,
                                                            dtype=dtype)
    else:
        if all([type(_labels) == list for _labels in labels]):
            # output the segmentation label maps into multiple volumes
            i = 0
            label_maps = list()
            for _labels in labels:
                _data = one_hot_encoding[..., i:i+len(_labels)]
                label_maps.append(convert_one_hot_to_label_map(_data, labels=_labels, axis=axis, threshold=threshold,
                                                               sum_then_threshold=sum_then_threshold, dtype=dtype))
                i = i + len(_labels)
            label_map = torch.stack(label_maps, dim=axis)
        else:
            label_map = convert_one_hot_to_single_label_map_volume(one_hot_encoding, labels, threshold, axis,
                                                                   sum_then_threshold, dtype)
        return label_map


def convert_one_hot_to_single_label_map_volume(one_hot_encoding, labels, threshold=0.5, axis=0,
                                               sum_then_threshold=False, dtype=torch.int16):
    # output a single label map volume
    segmentation_mask = mask_encoding(one_hot_encoding, len(labels), threshold=threshold, axis=axis,
                                      sum_then_threshold=sum_then_threshold)
    return assign_labels(one_hot_encoding, segmentation_mask, labels=labels, axis=axis,
                         dtype=dtype, label_indices=np.arange(len(labels)))


def mask_encoding(one_hot_encoding, n_labels, threshold=0.5, axis=0, sum_then_threshold=False):
    if sum_then_threshold:
        return torch.sum(one_hot_encoding[:n_labels], dim=axis) > threshold
    else:
        return torch.any(one_hot_encoding[:n_labels] > threshold, dim=axis)


def assign_labels(one_hot_encoding, segmentation_mask, labels, label_indices, axis=0, dtype=torch.int16):
    labelmap_shape = list(one_hot_encoding.shape)
    labelmap_shape.pop(axis)
    max_arg_map = torch.zeros(labelmap_shape, dtype=dtype)
    label_map = max_arg_map.detach().clone()
    max_arg_map[segmentation_mask] = (torch.argmax(one_hot_encoding[label_indices],
                                                   dim=axis) + 1)[segmentation_mask].to(dtype)
    for index, label in enumerate(labels):
        label_map[max_arg_map == (index + 1)] = label
    return label_map


def convert_one_hot_to_label_map_using_hierarchy(one_hot_encoding, labels, threshold=0.5, axis=0, dtype=torch.int16):
    roi = torch.ones(one_hot_encoding.shape[:axis], dtype=torch.bool)
    label_map = torch.zeros(one_hot_encoding.shape[:axis], dtype=dtype)
    for index, label in enumerate(labels):
        roi = torch.logical_and(one_hot_encoding[..., index] > threshold, roi)
        label_map[roi] = label
    return label_map


def _wip_convert_one_hot_to_label_map_with_label_groups(one_hot_encoding, labels, label_hierarchy, threshold=0.5,
                                                        axis=0):
    """
    This might be useful when doing something like brain segmentation where you want to segment the whole brain
    and then divide up that segmentation between white and gray matter, and then divide up the white and gray matter
    between individual labels.
    :param one_hot_encoding:
    :param labels:
    :param label_hierarchy: Hierarchy of the labels. Will assign labels according to their hierarchical categorization.
    :param kwargs:
    :return:
    """
    n_labels = len(labels)
    label_group_maps = list()
    for label_hierarchy_index, label_group in enumerate(label_hierarchy):
        label_group_channel = n_labels + label_hierarchy_index
        label_group_roi = one_hot_encoding[label_group_channel] > threshold
        # The labels within the label group can now be set
        label_group_label_indices = [labels.index(label) for label in label_group]
        label_group_maps.append(assign_labels(one_hot_encoding, label_group_roi, labels=label_group,
                                              label_indices=label_group_label_indices, axis=axis))
    if len(label_group_maps) == 1:
        return label_group_maps[0]
    else:
        return
        # return reconcile_label_group_maps(label_group_maps, label_hierarchy)


def one_hot_image_to_label_map(one_hot_image, labels, axis=0, threshold=0.5, sum_then_threshold=True, dtype=torch.int16,
                               label_hierarchy=None):
    label_map = convert_one_hot_to_label_map(get_nibabel_data(one_hot_image), labels=labels, axis=axis,
                                             threshold=threshold, sum_then_threshold=sum_then_threshold,
                                             dtype=dtype, label_hierarchy=label_hierarchy)
    return one_hot_image.make_similar(label_map)


def copy_image(image):
    return image.copy()


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
    base_image = images[0]
    if len(images) > 1:
        data = [image.get_data() for image in images]
        combined_data = torch.cat(data, axis)
        return base_image.make_similar(combined_data)
    return base_image


def move_channels_last(data):
    return torch.moveaxis(data, 0, -1)


def move_channels_first(data):
    return torch.moveaxis(data, -1, 0)


def load_image(filename, feature_axis=0, reorder=True, dtype=None, verbose=False):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    (for now, anything other than 0, probably won't work).
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """
    if type(filename) == list:
        return combine_images([load_single_image(fn, reorder=reorder, dtype=dtype, verbose=verbose)
                               for fn in filename], axis=feature_axis)
    else:
        return load_single_image(filename, reorder=reorder, dtype=dtype, verbose=verbose)


def load_single_image(filename, reorder=True, dtype=None, verbose=False):
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
    image = Image(data, torch.from_numpy(nib_image.affine))
    if verbose:
        print("Finished loading", filename, "Shape:", image.shape)
    if dtype is not None:
        image.set_dtype(dtype)
    if reorder:
        return reorder_image(image)
    return image


def reorder_image(image, axcodes="RAS"):
    array, _, affine = Orientation(axcodes=axcodes)(image.get_data(),
                                                    image.affine)
    image.update(array, affine)
    return image


def extract_sub_volumes(image, sub_volume_indices):
    data = image.dataobj[..., sub_volume_indices]
    return image.make_similar(data)


def mask(data, threshold=0, dtype=torch.float):
    return torch.as_tensor(data > threshold, dtype=dtype)


def get_nibabel_data(nibabel_image):
    return nibabel_image.get_fdata()


def in_config(string, dictionary, if_not_in_config_return=None):
    return dictionary[string] if string in dictionary else if_not_in_config_return


def estimate_binary_contour(binary):
    return torch.logical_xor(binary, binary_erosion(binary, iterations=1))


def add_one_hot_encoding_contours(one_hot_encoding):
    new_encoding = torch.zeros(one_hot_encoding.shape[:-1] + (one_hot_encoding.shape[-1] * 2,),
                               dtype=one_hot_encoding.dtype)
    new_encoding[..., :one_hot_encoding.shape[-1]] = one_hot_encoding
    for index in range(one_hot_encoding.shape[-1]):
        new_encoding[..., one_hot_encoding.shape[-1] + index] = estimate_binary_contour(
            one_hot_encoding[..., index] > 0)
    return new_encoding


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