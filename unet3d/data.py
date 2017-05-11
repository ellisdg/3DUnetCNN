import glob
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


def write_folders_to_file(subject_folders, data_storage, truth_storage, image_shape, n_channels, training_modalities,
                          crop=None, truth_dtype=np.uint8):
    for subject_folder in subject_folders:
        subject_data = read_subject_folder(subject_folder, image_shape, training_modalities=training_modalities,
                                           crop=crop)
        data_storage.append(subject_data[:n_channels][np.newaxis])
        truth_storage.append(np.asarray(subject_data[n_channels][np.newaxis][np.newaxis], dtype=truth_dtype))
    return data_storage, truth_storage


def write_data_to_file(data_folder, out_file, image_shape, n_channels=3, truth_dtype=np.uint8):
    subject_folders = get_subject_folders(data_folder)
    nb_samples = len(subject_folders)
    hdf5_file, data_storage, truth_storage = create_data_file(out_file, nb_channels=n_channels, nb_samples=nb_samples,
                                                              image_shape=image_shape)
    crop_slices, affine, header = find_downsized_info(subject_folders, image_shape)
    hdf5_file.create_array(hdf5_file.root, "affine", affine)
    write_folders_to_file(subject_folders, data_storage, truth_storage, image_shape, crop=crop_slices,
                          truth_dtype=truth_dtype, n_channels=n_channels)
    normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def get_subject_folders(data_dir):
    return glob.glob(os.path.join(data_dir, "*", "*"))


def get_affine(in_file):
    return nib.load(in_file).affine


def read_subject_folder(folder, image_shape, training_modalities, crop=None):
    data_list = list()
    for modality in training_modalities:
        data_list.append(read_image(os.path.join(folder, modality + ".nii.gz"), image_shape=image_shape,
                                    crop=crop).get_data())
    data_list.append(read_image(os.path.join(folder, "truth.nii.gz"), image_shape=image_shape, interpolation="nearest",
                                crop=crop).get_data())
    return np.stack(data_list)


def read_image(in_file, image_shape, interpolation='continuous', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(in_file)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    return resize(image, new_shape=image_shape, interpolation=interpolation)