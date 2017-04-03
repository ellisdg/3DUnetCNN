import glob
import os

import tables
import numpy as np
from nilearn.image import resample_img, reorder_img
import nibabel as nib


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(means).std(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


def create_data_file(out_file, nb_channels, nb_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, nb_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.createEArray(hdf5_file.root, 'data',
                                          tables.Float32Atom(),
                                          shape=data_shape,
                                          filters=filters,
                                          expectedrows=nb_samples)
    truth_storage = hdf5_file.createEArray(hdf5_file.root, 'truth',
                                           tables.UInt8Atom(),
                                           shape=truth_shape,
                                           filters=filters,
                                           expectedrows=nb_samples)
    return hdf5_file, data_storage, truth_storage


def write_folders_to_file(subject_folders, data_storage, truth_storage, image_shape, truth_dtype=np.uint8):
    for subject_folder in subject_folders:
        subject_data = read_subject_folder(subject_folder, image_shape)
        data_storage.append(subject_data[:3][np.newaxis])
        truth_storage.append(np.asarray(subject_data[3][np.newaxis][np.newaxis], dtype=truth_dtype))
    return data_storage, truth_storage


def write_data_to_file(data_folder, out_file, image_shape, truth_dtype=np.uint8, nb_channels=3):
    subject_folders = get_subject_folders(data_folder)
    nb_samples = len(subject_folders)
    hdf5_file, data_storage, truth_storage = create_data_file(out_file, nb_channels=nb_channels, nb_samples=nb_samples,
                                                              image_shape=image_shape)
    write_folders_to_file(subject_folders, data_storage, truth_storage, image_shape, truth_dtype=truth_dtype)
    normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def get_subject_folders(data_dir):
    return glob.glob(os.path.join(data_dir, "*", "*"))


def read_subject_folder(folder, image_shape):
    flair_image = read_image(os.path.join(folder, "Flair.nii.gz"), image_shape=image_shape)
    t1_image = read_image(os.path.join(folder, "T1.nii.gz"), image_shape=image_shape)
    t1c_image = read_image(os.path.join(folder, "T1c.nii.gz"), image_shape=image_shape)
    truth_image = read_image(os.path.join(folder, "truth.nii.gz"), image_shape=image_shape,
                             interpolation="nearest")
    return np.asarray([t1_image.get_data(), t1c_image.get_data(), flair_image.get_data(), truth_image.get_data()])


def read_image(in_file, image_shape, interpolation='continuous'):
    print("Reading: {0}".format(in_file))
    image = nib.load(in_file)
    return resize(image, new_shape=image_shape, interpolation=interpolation)


def resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


def get_truth(batch, truth_channel=3):
    truth = np.array(batch)[:, truth_channel]
    batch_list = []
    for sample_number in range(truth.shape[0]):
        array = np.zeros(truth[sample_number].shape)
        array[truth[sample_number] > 0] = 1
        batch_list.append([array])
    return np.array(batch_list)
