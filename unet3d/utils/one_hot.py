import numpy as np
import torch
from scipy.ndimage import binary_erosion
from monai.data import MetaTensor


def compile_one_hot_encoding(data, n_labels, labels=None, dtype=torch.uint8, return_4d=True, round=True):
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
    if round:
        data = torch.round(data, decimals=0)
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = MetaTensor(torch.zeros(new_shape, dtype=dtype), meta=data.meta)
    for label_index in range(n_labels):
        if labels is not None:
            if type(labels[label_index]) == list:
                # lists of labels will group together multiple labels from the label map into a single one-hot channel.
                for label in labels[label_index]:
                    # iterate through each of the labels in the sub list
                    y[:, label_index][isclose(data[:, 0], label)] = 1
            else:
                y[:, label_index][isclose(data[:, 0], labels[label_index])] = 1
        else:
            y[:, label_index][isclose(data[:, 0], (label_index + 1))] = 1
    if return_4d:
        assert y.shape[0] == 1
        y = y[0]
    return y


def isclose(a, b, atol=1e-08, rtol=1e-05):
    return torch.isclose(torch.ones(1) * a,
                         torch.ones(1) * b,
                         atol=atol, rtol=rtol)


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
                _data = one_hot_encoding[i:i+len(_labels)]
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
    labelmap_shape = shape_without_channels(one_hot_encoding, axis)
    max_arg_map = torch.zeros(labelmap_shape, dtype=dtype)
    label_map = max_arg_map.detach().clone()
    max_arg_map[segmentation_mask] = (torch.argmax(one_hot_encoding[label_indices],
                                                   dim=axis) + 1)[segmentation_mask].to(dtype)
    for index, label in enumerate(labels):
        label_map[max_arg_map == (index + 1)] = label
    return label_map


def shape_without_channels(tensor, dim=0):
    tensor_shape = list(tensor.shape)
    tensor_shape.pop(dim)
    return tensor_shape


def convert_one_hot_to_label_map_using_hierarchy(one_hot_encoding, labels, threshold=0.5, axis=0, dtype=torch.int16):
    # each label in a hierarchy is a contained within the roi of the previous label
    # therefore, each label in the hierarchy is defined by positive predictions in the current label
    # and those from the previous label.
    # For example, the BraTS hierarchy is: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
    # The TC is defined as a subset of the WT, and the ET is defined as a subset of the TC.
    # So we first need to define the WT, then the TC, and then the ET.
    # the variable "roi" defines the positive predictions from the previous label in the hierarchy
    # the initial roi is the entire image
    roi = torch.ones(shape_without_channels(one_hot_encoding, axis), dtype=torch.bool)
    # the initial label map is all zeros
    label_map = torch.zeros(roi.shape, dtype=dtype)
    for index, label in enumerate(labels):
        # refine the roi with the current positive predictions
        roi = torch.logical_and(one_hot_encoding[index] > threshold, roi)
        # assign the current label to the current roi
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
    label_map = convert_one_hot_to_label_map(one_hot_image, labels=labels, axis=axis,
                                             threshold=threshold, sum_then_threshold=sum_then_threshold,
                                             dtype=dtype, label_hierarchy=label_hierarchy)
    return one_hot_image.make_similar(label_map)


def estimate_binary_contour(binary):
    return torch.logical_xor(binary, binary_erosion(binary, iterations=1))


def add_one_hot_encoding_contours(one_hot_encoding):
    new_encoding = torch.zeros(one_hot_encoding.shape[:-1] + (one_hot_encoding.shape[-1] * 2,),
                               dtype=one_hot_encoding.dtype)
    new_encoding[:one_hot_encoding.shape[-1]] = one_hot_encoding
    for index in range(one_hot_encoding.shape[-1]):
        new_encoding[one_hot_encoding.shape[-1] + index] = estimate_binary_contour(
            one_hot_encoding[index] > 0)
    return new_encoding
