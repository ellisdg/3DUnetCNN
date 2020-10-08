import os
from functools import partial
import numpy as np
import nibabel as nib
from keras.utils import Sequence
from nilearn.image import new_img_like, resample_to_img, reorder_img
import random
import warnings

from .nilearn_custom_utils.nilearn_utils import crop_img
from .radiomic_utils import binary_classification, multilabel_classification, fetch_data, fetch_data_for_point
from .hcp import (extract_gifti_surface_vertices, get_vertices_from_scalar, get_metric_data,
                  extract_cifti_volumetric_data)
from .utils import (copy_image, extract_sub_volumes, mask,
                    compile_one_hot_encoding,
                    nib_load_files, load_image, load_single_image,
                    get_nibabel_data, add_one_hot_encoding_contours)
from .normalize import zero_mean_normalize_image_data, foreground_zero_mean_normalize_image_data, \
    zero_floor_normalize_image_data, zero_one_window
from . import normalize
from .resample import resample
from .augment import scale_affine, add_noise, affine_swap_axis, translate_affine, random_blur, random_permutation_x_y
from .affine import resize_affine


def normalization_name_to_function(normalization_name):
    if type(normalization_name) == list:
        return partial(normalize_data_with_multiple_functions, normalization_names=normalization_name)
    elif normalization_name == "zero_mean":
        return zero_mean_normalize_image_data
    elif normalization_name == "foreground_zero_mean":
        return foreground_zero_mean_normalize_image_data
    elif normalization_name == "zero_floor":
        return zero_floor_normalize_image_data
    elif normalization_name == "zero_one_window":
        return zero_one_window
    elif normalization_name == "mask":
        return mask
    elif normalization_name is not None:
        try:
            return getattr(normalize, normalization_name)
        except AttributeError:
            raise NotImplementedError(normalization_name + " normalization is not available.")
    else:
        return lambda x, **kwargs: x


def normalize_image_with_function(image, function, volume_indices=None, **kwargs):
    data = get_nibabel_data(image)
    if volume_indices is not None:
        data[..., volume_indices] = function(data[..., volume_indices], **kwargs)
    else:
        data = function(data, **kwargs)
    return new_img_like(image, data=data, affine=image.affine)


def normalize_data_with_multiple_functions(data, normalization_names, channels_axis=3, **kwargs):
    """

    :param data:
    :param normalization_names:
    :param channels_axis:
    :param kwargs: sets the normalization parameters, but should have multiple sets of parameters for the individual
    normalization functions.
    :return:
    """
    normalized_data = list()
    for name in normalization_names:
        func = normalization_name_to_function(name)
        _kwargs = dict(kwargs[name]) if name in kwargs else None
        if _kwargs and "volume_indices" in _kwargs and _kwargs["volume_indices"] is not None:
            volume_indices = _kwargs.pop("volume_indices")
            _data = data[..., volume_indices]
        else:
            _data = data
        normalized_data.append(func(_data, **_kwargs))
    return np.concatenate(normalized_data, axis=channels_axis)


def augment_affine(affine, shape, augment_scale_std=None, augment_scale_probability=1,
                   flip_left_right_probability=0, augment_translation_std=None, augment_translation_probability=1,
                   flip_front_back_probability=0):
    if augment_scale_std and decision(augment_scale_probability):
        scale = np.random.normal(1, augment_scale_std, 3)
        affine = scale_affine(affine, shape, scale)
    if decision(flip_left_right_probability):  # flips the left and right sides of the image randomly
        affine = affine_swap_axis(affine, shape=shape, axis=0)
    if decision(flip_front_back_probability):
        affine = affine_swap_axis(affine, shape=shape, axis=1)
    if augment_translation_std and decision(augment_translation_probability):
        affine = translate_affine(affine, shape,
                                  translation_scales=np.random.normal(loc=0, scale=augment_translation_std, size=3))
    return affine


def augment_image(image, augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                  additive_noise_std=None, additive_noise_probability=1):
    if not (augment_blur_mean is None or augment_blur_std is None) and decision(augment_blur_probability):
        image = random_blur(image, mean=augment_blur_mean, std=augment_blur_std)
    if additive_noise_std and decision(additive_noise_probability):
        image.dataobj[:] = add_noise(image.dataobj, sigma_factor=additive_noise_std)
    return image


def format_feature_image(feature_image, window, crop=False, cropping_kwargs=None, augment_scale_std=None,
                         augment_scale_probability=1, additive_noise_std=None, additive_noise_probability=0,
                         flip_left_right_probability=0, augment_translation_std=None,
                         augment_translation_probability=0, augment_blur_mean=None, augment_blur_std=None,
                         augment_blur_probability=0, flip_front_back_probability=0, reorder=False,
                         interpolation="linear"):
    if reorder:
        feature_image = reorder_img(feature_image, resample=interpolation)
    if crop:
        if cropping_kwargs is None:
            cropping_kwargs = dict()
        affine, shape = crop_img(feature_image, return_affine=True, **cropping_kwargs)
    else:
        affine = feature_image.affine.copy()
        shape = feature_image.shape
    affine = augment_affine(affine, shape,
                            augment_scale_std=augment_scale_std,
                            augment_scale_probability=augment_scale_probability,
                            augment_translation_std=augment_translation_std,
                            augment_translation_probability=augment_translation_probability,
                            flip_left_right_probability=flip_left_right_probability,
                            flip_front_back_probability=flip_front_back_probability)
    feature_image = augment_image(feature_image,
                                  augment_blur_mean=augment_blur_mean,
                                  augment_blur_std=augment_blur_std,
                                  augment_blur_probability=augment_blur_probability,
                                  additive_noise_std=additive_noise_std,
                                  additive_noise_probability=additive_noise_probability)
    affine = resize_affine(affine, shape, window)
    return feature_image, affine


def decision(probability):
    if not probability or probability <= 0:
        return False
    elif probability >= 1:
        return True
    else:
        return random.random() < probability


class BaseSequence(Sequence):
    def __init__(self, filenames, batch_size, target_labels, window, spacing, classification='binary', shuffle=True,
                 points_per_subject=1, flip=False, reorder=False, iterations_per_epoch=1, deformation_augmentation=None,
                 base_directory=None, subject_ids=None, inputs_per_epoch=None, channel_axis=3):
        self.deformation_augmentation = deformation_augmentation
        self.base_directory = base_directory
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = filenames
        self.inputs_per_epoch = inputs_per_epoch
        if self.inputs_per_epoch is not None:
            if not type(self.filenames) == dict:
                raise ValueError("'inputs_per_epoch' is not None, but 'filenames' is not a dictionary.")
            self.filenames_dict = self.filenames
        else:
            self.filenames_dict = None
        self.target_labels = target_labels
        self.window = window
        self.points_per_subject = points_per_subject
        self.flip = flip
        self.reorder = reorder
        self.spacing = spacing
        self.iterations_per_epoch = iterations_per_epoch
        self.subjects_per_batch = int(np.floor(self.batch_size / self.points_per_subject))
        assert self.subjects_per_batch > 0
        if classification == 'binary':
            self._classify = binary_classification
        elif classification == 'multilabel':
            self._classify = multilabel_classification
        else:
            self._classify = classification
        self.channel_axis = channel_axis
        self.on_epoch_end()

    def get_number_of_subjects_per_epoch(self):
        return self.get_number_of_subjects() * self.iterations_per_epoch

    def get_number_of_subjects(self):
        return len(self.filenames)

    def generate_epoch_filenames(self):
        if self.inputs_per_epoch is not None:
            self.sample_filenames()
        _filenames = list(self.filenames)
        epoch_filenames = list()
        for i in range(self.iterations_per_epoch):
            if self.shuffle:
                np.random.shuffle(_filenames)
            epoch_filenames.extend(_filenames)
        self.epoch_filenames = list(epoch_filenames)

    def switch_to_augmented_filename(self, subject_id, filename):
        augmented_filename = self.deformation_augmentation.format(base_directory=self.base_directory,
                                                                  random_subject_id=np.random.choice(self.subject_ids),
                                                                  subject_id=subject_id,
                                                                  basename=os.path.basename(filename))
        if not os.path.exists(augmented_filename):
            raise RuntimeWarning("Augmented filename {} does not exists!".format(augmented_filename))
        else:
            filename = augmented_filename
        return filename

    def __len__(self):
        return self.get_number_of_batches_per_epoch()

    def get_number_of_batches_per_epoch(self):
        return int(np.floor(np.divide(self.get_number_of_subjects_per_epoch(),
                                      self.subjects_per_batch)))

    def __getitem__(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for feature_filename, target_filename in batch_filenames:
            for x, y in zip(*fetch_data(feature_filename,
                                        target_filename,
                                        self.target_labels,
                                        self.window,
                                        n_points=self.points_per_subject,
                                        flip=self.flip,
                                        reorder=self.reorder,
                                        spacing=self.spacing,
                                        classify=self._classify)):
                batch_x.append(x)
                batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        self.generate_epoch_filenames()

    def sample_filenames(self):
        """
        Sample the filenames.
        """
        filenames = list()
        for key in self.filenames_dict:
            if self.inputs_per_epoch[key] == "all":
                filenames.extend(self.filenames_dict[key])
            else:
                _filenames = list(self.filenames_dict[key])
                np.random.shuffle(_filenames)
                filenames.extend(_filenames[:self.inputs_per_epoch[key]])
        self.filenames = filenames


class HCPParent(object):
    def __init__(self, surface_names, window, flip, reorder, spacing):
        self.surface_names = surface_names
        self.reorder = reorder
        self.flip = flip
        self.spacing = spacing
        self.window = window

    def extract_vertices(self, surface_filenames, metrics):
        # extract the vertices
        surfaces = nib_load_files(surface_filenames)
        vertices = list()
        for surface, surface_name in zip(surfaces, self.surface_names):
            vertices_index = get_vertices_from_scalar(metrics[0], brain_structure_name=surface_name)
            surface_vertices = extract_gifti_surface_vertices(surface, primary_anatomical_structure=surface_name)
            vertices.extend(surface_vertices[vertices_index])
        return np.asarray(vertices)


class HCPRegressionSequence(BaseSequence, HCPParent):
    def __init__(self, filenames, batch_size, window, spacing, metric_names, classification=None,
                 surface_names=('CortexLeft', 'CortexRight'), normalization="zero_mean", normalization_args=None,
                 **kwargs):
        super().__init__(filenames=filenames, batch_size=batch_size, target_labels=tuple(), window=window,
                         spacing=spacing, classification=classification, **kwargs)
        self.metric_names = metric_names
        self.surface_names = surface_names
        self.normalize = normalization is not None
        self.normalization_func = normalization_name_to_function(normalization)
        if normalization_args is not None:
            self.normalization_kwargs = normalization_args
        else:
            self.normalization_kwargs = dict()

    def __getitem__(self, idx):
        return self.fetch_hcp_regression_batch(idx)

    def fetch_hcp_regression_batch(self, idx):
        batch_filenames = self.epoch_filenames[idx * self.subjects_per_batch:(idx + 1) * self.subjects_per_batch]
        batch_x = list()
        batch_y = list()
        for args in batch_filenames:
            _x, _y = self.fetch_hcp_subject_batch(*args)
            batch_x.extend(_x)
            batch_y.extend(_y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def load_metric_data(self, metric_filenames, subject_id):
        metrics = nib_load_files(metric_filenames)
        all_metric_data = get_metric_data(metrics, self.metric_names, self.surface_names, subject_id)
        return metrics, all_metric_data

    def load_feature_data(self, feature_filename, vertices, target_values):
        batch_x = list()
        batch_y = list()
        feature_image = load_single_image(feature_filename, reorder=self.reorder)
        for vertex, y in zip(vertices, target_values):
            x = fetch_data_for_point(vertex, feature_image, window=self.window, flip=self.flip,
                                     spacing=self.spacing)
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def fetch_hcp_subject_batch(self, feature_filename, surface_filenames, metric_filenames, subject_id):
        metrics, all_metric_data = self.load_metric_data(metric_filenames, subject_id)
        vertices = self.extract_vertices(surface_filenames, metrics)
        random_vertices, random_target_values = self.select_random_vertices_and_targets(vertices, all_metric_data)
        return self.load_feature_data(feature_filename, random_vertices, random_target_values)

    def select_random_vertices(self, vertices):
        indices = np.random.choice(np.arange(vertices.shape[0]), size=self.points_per_subject, replace=False)
        random_vertices = vertices[indices]
        return random_vertices, indices

    def select_random_vertices_and_targets(self, vertices, all_metric_data):
        # randomly select the target vertices and corresponding values
        random_vertices, indices = self.select_random_vertices(vertices)
        random_target_values = all_metric_data[indices]
        return random_vertices, random_target_values

    def normalize_image(self, image):
        if self.normalize:
            return normalize_image_with_function(image, self.normalization_func, **self.normalization_kwargs)
        return image


class ParcelBasedSequence(HCPRegressionSequence):
    def __init__(self, *args, target_parcel, parcellation_template, parcellation_name, **kwargs):
        self.target_parcel = target_parcel
        self.parcellation_template = parcellation_template
        self.parcellation_name = parcellation_name
        super().__init__(*args, **kwargs)

    def fetch_hcp_subject_batch(self, feature_filename, surface_filenames, metric_filenames, subject_id):
        metrics, all_metric_data = self.load_metric_data(metric_filenames, subject_id)
        vertices = self.extract_vertices(surface_filenames, metrics)
        parcellation = self.load_parcellation(subject_id)
        parcellation_mask = np.in1d(parcellation, self.target_parcel)
        random_vertices, random_target_values = self.select_random_vertices_and_targets(
            vertices[parcellation_mask], all_metric_data[parcellation_mask])
        return self.load_feature_data(feature_filename, random_vertices, random_target_values)

    def load_parcellation(self, subject_id):
        parcellation_filename = self.parcellation_template.format(subject_id)
        parcellation = np.squeeze(get_metric_data(metrics=nib_load_files([parcellation_filename]),
                                                  metric_names=[[self.parcellation_name]],
                                                  surface_names=self.surface_names,
                                                  subject_id=subject_id))
        return parcellation


class SubjectPredictionSequence(HCPParent, Sequence):
    def __init__(self, feature_filename, surface_filenames, surface_names, reference_metric_filename,
                 batch_size=50, window=(64, 64, 64), flip=False, spacing=(1, 1, 1), reorder=False):
        super().__init__(surface_names=surface_names, window=window, flip=flip, reorder=reorder, spacing=spacing)
        self.feature_image = load_single_image(feature_filename, reorder=self.reorder)
        self.reference_metric = nib.load(reference_metric_filename)
        self.vertices = self.extract_vertices(surface_filenames=surface_filenames, metrics=[self.reference_metric])
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(np.divide(len(self.vertices), self.batch_size)))

    def __getitem__(self, idx):
        batch_vertices = self.vertices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [fetch_data_for_point(vertex,
                                      self.feature_image,
                                      window=self.window,
                                      flip=self.flip,
                                      spacing=self.spacing) for vertex in batch_vertices]
        return np.asarray(batch)


class WholeVolumeToSurfaceSequence(HCPRegressionSequence):
    def __init__(self, interpolation='linear', crop=True, cropping_kwargs=None,
                 augment_scale_std=0, augment_scale_probability=1, additive_noise_std=0, additive_noise_probability=1,
                 augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                 augment_translation_std=None, augment_translation_probability=1, flip_left_right_probability=0,
                 flip_front_back_probability=0, resample=None, **kwargs):
        """

        :param interpolation: interpolation to use when formatting the feature image.
        :param crop: if true, images will be cropped to remove the background from the feature image.
        :param cropping_pad_width: width of the padding around the foreground after cropping.
        :param augment_scale_std: randomly augment the scale with this standard deviation (mean of 1). If None, 0 or
        False, no augmentation will be done.
        :param augment_scale_probability: If the scale augmentation is set, randomly pick when to implement.
        :param additive_noise_std:
        :param additive_noise_probability:
        :param augment_blur_mean:
        :param augment_blur_std:
        :param augment_blur_probability:
        :param augment_translation_std:
        :param augment_translation_probability:
        :param flip_left_right_probability:
        :param resample: deprecated
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.interpolation = interpolation
        if resample is not None:
            warnings.warn("'resample' argument is deprecated. Use 'interpolation'.", DeprecationWarning)
        self.crop = crop
        self.augment_scale_std = augment_scale_std
        self.augment_scale_probability = augment_scale_probability
        self.additive_noise_std = additive_noise_std
        self.additive_noise_probability = additive_noise_probability
        self.cropping_kwargs = cropping_kwargs
        self.augment_blur_mean = augment_blur_mean
        self.augment_blur_std = augment_blur_std
        self.augment_blur_probability = augment_blur_probability
        self.augment_translation_std = augment_translation_std
        self.augment_translation_probability = augment_translation_probability
        self.flip_left_right_probability = flip_left_right_probability
        self.flip_front_back_probability = flip_front_back_probability

    def __len__(self):
        return int(np.ceil(np.divide(len(self.epoch_filenames), self.batch_size)))

    def __getitem__(self, idx):
        x = list()
        y = list()
        batch_filenames = self.epoch_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        for feature_filename, surface_filenames, metric_filenames, subject_id in batch_filenames:
            if self.deformation_augmentation:
                feature_filename = self.switch_to_augmented_filename(subject_id=subject_id, filename=feature_filename)
            metrics = nib_load_files(metric_filenames)
            x.append(self.resample_input(feature_filename))
            y.append(get_metric_data(metrics, self.metric_names, self.surface_names, subject_id).T.ravel())
        return np.asarray(x), np.asarray(y)

    def resample_input(self, feature_filename):
        feature_image = load_image(feature_filename, reorder=False)
        feature_image, affine = format_feature_image(feature_image=feature_image, window=self.window, crop=self.crop,
                                                     cropping_kwargs=self.cropping_kwargs,
                                                     augment_scale_std=self.augment_scale_std,
                                                     augment_scale_probability=self.augment_scale_probability,
                                                     additive_noise_std=self.additive_noise_std,
                                                     additive_noise_probability=self.additive_noise_probability,
                                                     augment_blur_mean=self.augment_blur_mean,
                                                     augment_blur_std=self.augment_blur_std,
                                                     augment_blur_probability=self.augment_blur_probability,
                                                     flip_left_right_probability=self.flip_left_right_probability,
                                                     flip_front_back_probability=self.flip_front_back_probability,
                                                     augment_translation_std=self.augment_translation_std,
                                                     augment_translation_probability=self.augment_translation_probability,
                                                     reorder=self.reorder,
                                                     interpolation=self.interpolation)
        input_img = resample(feature_image, affine, self.window, interpolation=self.interpolation)
        return get_nibabel_data(self.normalize_image(input_img))


class WholeVolumeAutoEncoderSequence(WholeVolumeToSurfaceSequence):
    def __init__(self, *args, target_interpolation=None, target_index=None, feature_index=0, extract_sub_volumes=False,
                 feature_sub_volumes_index=1, target_sub_volumes_index=3, random_permutation_probability=0, **kwargs):
        """

        :param args:
        :param target_interpolation:
        :param target_index:
        :param feature_index:
        :param extract_sub_volumes: if True, the sequence will expect a set of indices that will be used to extract
        specific volumes out of the volumes being read. (default=False)
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if target_interpolation is None:
            self.target_interpolation = self.interpolation
        else:
            self.target_interpolation = target_interpolation
        self.target_index = target_index
        self.feature_index = feature_index
        self.extract_sub_volumes = extract_sub_volumes
        self.feature_sub_volumes_index = feature_sub_volumes_index
        self.target_sub_volumes_index = target_sub_volumes_index
        self.random_permutation_probability = random_permutation_probability

    def __getitem__(self, idx):
        x_batch = list()
        y_batch = list()
        batch_filenames = self.epoch_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        for item in batch_filenames:
            x, y = self.resample_input(item)
            x_batch.append(x)
            y_batch.append(y)
        return np.asarray(x_batch), np.asarray(y_batch)

    def resample_input(self, input_filenames):
        input_image, target_image = self.resample_image(input_filenames)
        x, y = get_nibabel_data(input_image), get_nibabel_data(target_image)
        return self.permute_inputs(x, y)

    def permute_inputs(self, x, y):
        if decision(self.random_permutation_probability):
            x, y = random_permutation_x_y(x, y, channel_axis=self.channel_axis)
        return x, y

    def resample_image(self, input_filenames):
        feature_image = self.format_feature_image(input_filenames=input_filenames)
        target_image = self.resample_target(self.load_target_image(feature_image, input_filenames),
                                            feature_image)
        feature_image = augment_image(feature_image,
                                      additive_noise_std=self.additive_noise_std,
                                      additive_noise_probability=self.additive_noise_probability,
                                      augment_blur_mean=self.augment_blur_mean,
                                      augment_blur_std=self.augment_blur_std,
                                      augment_blur_probability=self.augment_blur_probability)
        return feature_image, target_image

    def load_image(self, filenames, index, force_4d=True, interpolation="linear", sub_volume_indices=None):
        filename = filenames[index]
        # Reordering is done when the image is formatted
        image = load_image(filename, force_4d=force_4d, reorder=False, interpolation=interpolation)
        if sub_volume_indices:
            image = extract_sub_volumes(image, sub_volume_indices)
        return image

    def load_feature_image(self, input_filenames):
        if self.extract_sub_volumes:
            sub_volume_indices = input_filenames[self.feature_sub_volumes_index]
        else:
            sub_volume_indices = None
        return self.load_image(input_filenames, self.feature_index, force_4d=True, interpolation=self.interpolation,
                               sub_volume_indices=sub_volume_indices)

    def format_feature_image(self, input_filenames, return_unmodified=False):
        unmodified_image = self.load_feature_image(input_filenames)
        image, affine = format_feature_image(feature_image=self.normalize_image(unmodified_image),
                                             crop=self.crop,
                                             cropping_kwargs=self.cropping_kwargs,
                                             augment_scale_std=self.augment_scale_std,
                                             augment_scale_probability=self.augment_scale_probability,
                                             window=self.window,
                                             additive_noise_std=None,  # augmented later
                                             augment_blur_mean=None,  # augmented later
                                             augment_blur_std=None,  # augmented later
                                             flip_left_right_probability=self.flip_left_right_probability,
                                             flip_front_back_probability=self.flip_front_back_probability,
                                             augment_translation_std=self.augment_translation_std,
                                             augment_translation_probability=self.augment_translation_probability,
                                             reorder=self.reorder,
                                             interpolation=self.interpolation)
        resampled = resample(image, affine, self.window, interpolation=self.interpolation)
        if return_unmodified:
            return resampled, unmodified_image
        else:
            return resampled

    def load_target_image(self, feature_image, input_filenames):
        if self.target_index is None:
            target_image = copy_image(feature_image)
        else:
            if self.extract_sub_volumes:
                sub_volume_indices = input_filenames[self.target_sub_volumes_index]
            else:
                sub_volume_indices = None
            target_image = self.load_image(input_filenames, self.target_index, force_4d=True,
                                           sub_volume_indices=sub_volume_indices,
                                           interpolation=self.target_interpolation)
        return target_image

    def resample_target(self, target_image, feature_image):
        target_image = resample_to_img(target_image, feature_image, interpolation=self.target_interpolation)
        return target_image

    def get_image(self, idx):
        input_image, target_image = self.resample_image(self.epoch_filenames[idx])
        return input_image, target_image

    def get_feature_image(self, idx, return_unmodified=False):
        return self.format_feature_image(self.epoch_filenames[idx], return_unmodified=return_unmodified)


class WholeVolumeSegmentationSequence(WholeVolumeAutoEncoderSequence):
    def __init__(self, *args, target_interpolation="nearest", target_index=2, labels=None, add_contours=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.target_interpolation = target_interpolation
        self.target_index = target_index
        self.labels = labels
        self.add_contours = add_contours

    def resample_input(self, input_filenames):
        input_image, target_image = self.resample_image(input_filenames)
        target_data = get_nibabel_data(target_image)
        if self.labels is None:
            self.labels = np.unique(target_data)
        assert len(target_data.shape) == 4
        if target_data.shape[3] == 1:
            target_data = np.moveaxis(
                compile_one_hot_encoding(np.moveaxis(target_data, 3, 0),
                                         n_labels=len(self.labels),
                                         labels=self.labels,
                                         return_4d=True), 0, 3)
        else:
            _target_data = list()
            for channel, labels in zip(range(target_data.shape[self.channel_axis]), self.labels):
                _target_data.append(np.moveaxis(
                    compile_one_hot_encoding(np.moveaxis(target_data[..., channel, None], self.channel_axis, 0),
                                             n_labels=len(labels),
                                             labels=labels,
                                             return_4d=True), 0, self.channel_axis))
            target_data = np.concatenate(_target_data, axis=self.channel_axis)
        if self.add_contours:
            target_data = add_one_hot_encoding_contours(target_data)
        return self.permute_inputs(get_nibabel_data(input_image), target_data)


class WindowedAutoEncoderSequence(HCPRegressionSequence):
    def __init__(self, *args, resample="linear", additive_noise_std=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.additive_noise_std = additive_noise_std
        self.resample = resample

    def fetch_hcp_subject_batch(self, feature_filename, surface_filenames, metric_filenames, subject_id):
        surfaces = nib_load_files(surface_filenames)
        vertices = list()
        for surface, surface_name in zip(surfaces, self.surface_names):
            vertices.extend(extract_gifti_surface_vertices(surface, primary_anatomical_structure=surface_name))
        random_vertices, _ = self.select_random_vertices(np.asarray(vertices))
        return self.load_feature_data_without_metrics(feature_filename, random_vertices)

    def load_feature_data_without_metrics(self, feature_filename, random_vertices):
        batch_x = list()
        batch_y = list()
        feature_image = load_single_image(feature_filename, resample=self.resample, reorder=self.reorder)
        normalized_image = self.normalize_image(feature_image)
        for vertex in random_vertices:
            x = fetch_data_for_point(vertex, normalized_image, window=self.window, flip=self.flip, spacing=self.spacing,
                                     normalization_func=None)
            batch_x.append(self.augment(x))
            batch_y.append(x)
        return batch_x, batch_y

    def augment(self, x):
        if self.additive_noise_std > 0:
            x = add_noise(x, sigma_factor=self.additive_noise_std)
        return x


class WholeVolumeSupervisedRegressionSequence(WholeVolumeAutoEncoderSequence):
    def __init__(self, *args, target_normalization=None, target_interpolation=None, target_index=2, **kwargs):
        super().__init__(*args, target_index=target_index, target_interpolation=target_interpolation, **kwargs)
        self.normalize_target = target_normalization is not None
        self.target_normalization_func = normalization_name_to_function(target_normalization)

    def load_target_image(self, feature_image, input_filenames, resample=False):
        target_image = super().load_target_image(feature_image, input_filenames)
        if self.normalize_target:
            image_data = self.target_normalization_func(target_image.get_fdata())
            return new_img_like(ref_niimg=feature_image, data=image_data,
                                affine=target_image.header.affine)
        else:
            return target_image


class WholeVolumeCiftiSupervisedRegressionSequence(WholeVolumeSupervisedRegressionSequence):
    def __init__(self, *args, target_normalization=None, target_interpolation=None, target_index=2,
                 subject_id_index=3, **kwargs):
        super().__init__(*args, target_index=target_index, target_interpolation=target_interpolation,
                         target_normalization=target_normalization, **kwargs)
        self.subject_id_index = subject_id_index

    def load_target_image(self, feature_image, input_filenames, reorder=False):
        target_image_filename = input_filenames[self.target_index]
        cifti_target_image = nib.load(target_image_filename)
        image_data = extract_cifti_volumetric_data(cifti_image=cifti_target_image,
                                                   map_names=self.metric_names,
                                                   subject_id=input_filenames[self.subject_id_index])
        image_data = self.target_normalization_func(image_data)
        return new_img_like(ref_niimg=feature_image, data=image_data,
                            affine=cifti_target_image.header.get_axis(1).affine)
