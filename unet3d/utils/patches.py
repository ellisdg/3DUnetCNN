import numpy as np


def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)


def get_set_of_patch_indices(start, stop, step):
    return np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1], start[2]:stop[2]:step[2]].reshape(3, -1).T


def get_random_patch_index(image_shape, patch_shape):
    """
    Returns a random corner index for a patch. If this is used during training, the middle pixels will be seen by
    the model way more often than the edge pixels (which is probably a bad thing).
    :param image_shape: Shape of the image
    :param patch_shape: Shape of the patch
    :return: a tuple containing the corner index which can be used to get a patch from an image
    """
    return get_random_nd_index(np.subtract(image_shape, patch_shape))


def get_random_nd_index(index_max):
    return tuple([np.random.choice(index_max[index] + 1) for index in range(len(index_max))])


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.negative((patch_index < 0) * patch_index)
    pad_after = ((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape)
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = np.vstack([[[0, 0] * (len(data.shape) - pad_args.shape[0])], pad_args])
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index
