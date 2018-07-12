import os

import numpy as np
import tables
from tables.exceptions import NoSuchNodeError, NodeError
import nibabel as nib

from .normalize import normalize_data_storage, reslice_image_set
from .utils.utils import resample, is_iterable


def byte_to_string(array):
    return [item.decode('utf-8') for item in array]


class DataFile(object):
    def __init__(self, filename, image_class=nib.Nifti1Image, mode='a'):
        self.filename = filename
        self._data_file = tables.open_file(filename, mode=mode)
        try:
            self._data_group = self._data_file.root.data
            self._parameters_group = self._data_file.root.parameters
        except AttributeError:
            self._data_group = self._data_file.create_group(self._data_file.root, 'data')
            self._parameters_group = self._data_file.create_group(self._data_file.root, 'parameters')
        self._image_class = image_class

    def add_data(self, features, targets, name, **kwargs):
        group = self._data_file.create_group(self._data_group, name)
        self.add_array(features, 'features', group)
        self.add_array(targets, 'targets', group)
        self.add_supplemental_data(name, **kwargs)

    def add_array(self, array, array_name, group):
        return self._data_file.create_array(group, array_name, array)

    def add_nibabel_images(self, features_image, targets_image, name, check_affine=True, **kwargs):
        if is_iterable(features_image):
            features_image = combine_images(features_image)
        if is_iterable(targets_image):
            targets_image = combine_images(targets_image)
        features = features_image.get_data()
        if check_affine:
            np.testing.assert_array_equal(features_image.affine, targets_image.affine)
        targets = targets_image.get_data()
        self.add_data(features, targets, name, affine=features_image.affine, **kwargs)

    def add_supplemental_data(self, name, **kwargs):
        for key in kwargs:
            try:
                self._data_file.create_array(self[name], key, kwargs[key])
            except NodeError:
                self.overwrite_array(name, kwargs[key], key)

    def get_data(self, name, supplemental_feature_names=None, concatenation_axis=0):
        features, targets = self[name].features, self[name].targets
        if supplemental_feature_names:
            for key in supplemental_feature_names:
                supplemental_data = self.get_supplemental_data(name, key)
                if len(supplemental_data.shape) < len(features.shape):
                    supplemental_data = np.expand_dims(supplemental_data, axis=concatenation_axis)
                features = np.concatenate((features, supplemental_data), axis=concatenation_axis)
        return features, targets

    def get_nibabel_images(self, name, channels_last=False, supplemental_feature_names=None):
        features, targets = self.get_data(name, supplemental_feature_names=supplemental_feature_names)
        if channels_last:
            features = move_4d_channels_last(features)
            targets = move_4d_channels_last(targets)
        affine = self[name].affine
        return self._image_class(features, affine), self._image_class(targets, affine)

    def get_roi(self, name):
        return self.get_roi_affine(name), self.get_roi_shape(name)

    def get_roi_affine(self, name):
        return self[name].roi_affine

    def get_roi_shape(self, name):
        return self[name].roi_shape

    def get_roi_data(self, name, features_interpolation='linear', targets_interpolation='nearest', roi_affine=None,
                     roi_shape=None, supplemental_feature_names=None):
        try:
            features_image, targets_image = self.get_nibabel_images(
                name, channels_last=True, supplemental_feature_names=supplemental_feature_names)
        except NoSuchNodeError:
            return self.get_data(name, supplemental_feature_names=supplemental_feature_names)
        if roi_affine is None:
            try:
                roi_affine = self.get_roi_affine(name)
            except NoSuchNodeError:
                roi_affine = features_image.affine
        if roi_shape is None:
            try:
                roi_shape = self.get_roi_shape(name)
            except NoSuchNodeError:
                roi_shape = features_image.shape[:3]
        roi_features_image = resample(image=features_image, target_affine=roi_affine, target_shape=roi_shape,
                                      interpolation=features_interpolation, pad=False)
        roi_targets_image = resample(image=targets_image, target_affine=roi_affine, target_shape=roi_shape,
                                     interpolation=targets_interpolation, pad=False)
        return (move_4d_channels_first(roi_features_image.get_data()),
                move_4d_channels_first(roi_targets_image.get_data()))

    def get_supplemental_data(self, name, key):
        return self._data_file.get_node(self.__getitem__(name), key)

    def get_supplemental_image(self, name, key):
        data = self.get_supplemental_data(name, key)
        affine = self.get_supplemental_data(name, 'affine')
        return self._image_class(data, affine)

    def set_array(self, array, name, group):
        try:
            node = self.get_node(group, name)
            self._data_file.remove_node(node)
        except NoSuchNodeError:
            pass
        return self.add_array(array, name, group)

    def set_training_groups(self, training_groups):
        return self.set_array(training_groups, 'training', self._parameters_group)

    def set_validation_groups(self, validation_groups):
        return self.set_array(validation_groups, 'validation', self._parameters_group)

    def get_data_groups(self):
        return list(self._data_group._v_children.keys())

    def get_training_groups(self):
        return byte_to_string(self.__getitem__('training', 'parameters'))

    def get_validation_groups(self):
        return byte_to_string(self.__getitem__('validation', 'parameters'))

    def overwrite_array(self, name, array, key):
        self.delete_array(name, key)
        return self.add_array(array, key, self.get_data_node(name))

    def delete_array(self, name, key):
        array_node = self.get_node(self.get_data_node(name), key)
        self._data_file.remove_node(array_node)

    def get_node(self, where, name):
        return self._data_file.get_node(where, name)

    def get_data_node(self, name):
        return self.get_node(self._data_group, name)

    def close(self):
        self._data_file.close()

    def __getitem__(self, key, group="data"):
        return self._data_file.get_node(self._data_file.get_node(self._data_file.root, group), key)

    def __del__(self):
        if self._data_file.isopen:
            self.close()


def combine_images(images, axis=0):
    base_image = images[0]
    data = list()
    max_dim = len(base_image.shape)
    for image in images:
        np.testing.assert_array_equal(image.affine, base_image.affine)
        image_data = image.get_data()
        dim = len(image.shape)
        if dim < max_dim:
            image_data = np.expand_dims(image_data, axis=axis)
        elif dim > max_dim:
            max_dim = max(max_dim, dim)
            data = [np.expand_dims(x, axis=axis) for x in data]
        data.append(image_data)
    if len(data[0].shape) > 3:
        array = np.concatenate(data, axis=axis)
    else:
        array = np.stack(data, axis=axis)
    return base_image.__class__(array, base_image.affine)


def move_4d_channels_last(data):
    if len(data.shape) > 3:
        return move_channels_last(data)
    return data


def move_4d_channels_first(data):
    if len(data.shape) > 3:
        return move_channels_first(data)
    return data


def move_channels_last(data):
    return np.moveaxis(data, 0, -1)


def move_channels_first(data):
    return np.moveaxis(data, -1, 0)


def move_image_channels(image, axis0, axis1):
    data = image.get_data()
    new_data = np.moveaxis(data, axis0, axis1)
    return image.__class__(new_data, image.affine)


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
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
    n_channels = len(training_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage
