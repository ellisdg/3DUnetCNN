import os

import nibabel as nib
import numpy as np
import tables

from .normalize import find_downsized_info, normalize_data_storage, resize
from .utils import crop_img_to


def create_data_file(out_file, nb_channels, nb_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, nb_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=nb_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=nb_samples)
    return hdf5_file, data_storage, truth_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, crop=None,
                             truth_dtype=np.uint8):
    for set_of_files in image_files:
        subject_data = read_image_files(set_of_files, image_shape, crop=crop)
        data_storage.append(subject_data[:n_channels][np.newaxis])
        truth_storage.append(np.asarray(subject_data[n_channels][np.newaxis][np.newaxis], dtype=truth_dtype))
    return data_storage, truth_storage


def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0])

    try:
        hdf5_file, data_storage, truth_storage = create_data_file(out_file, nb_channels=n_channels,
                                                                  nb_samples=n_samples, image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    crop_slices, affine, header = find_downsized_info(training_data_files, image_shape)
    hdf5_file.create_array(hdf5_file.root, "affine", affine)
    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape, crop=crop_slices,
                             truth_dtype=truth_dtype, n_channels=n_channels)
    normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def get_affine(in_file):
    return nib.load(in_file).affine


def read_image_files(image_files, image_shape, crop=None):
    data_list = list()
    for image_file in image_files:
        data_list.append(read_image(image_file, image_shape=image_shape, crop=crop).get_data())
    return np.stack(data_list)


def read_image(in_file, image_shape, interpolation='continuous', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(in_file)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    return resize(image, new_shape=image_shape, interpolation=interpolation)