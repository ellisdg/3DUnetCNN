import os
import glob
import nibabel as nib
import numpy as np

from unet3d.data import DataFile, combine_images, move_image_channels
from unet3d.normalize import (compute_region_of_interest_affine, compute_region_of_interest_affine_from_foreground,
                              normalize_data)
from unet3d.generator import get_generators_from_data_file, split_list, label_map_to_4d_labels
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model, set_model_learning_rate
from unet3d.utils.utils import update_progress, resize_affine
from unet3d.utils.nilearn_custom_utils.nilearn_utils import reorder_affine
from nilearn.image import resample_to_img

import json


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def get_model(model_file, overwrite=False, n_labels=1, batch_size=1, **kwargs):
    print(os.path.abspath(model_file))
    if not os.path.exists(model_file) or overwrite:
        # set the input shape to be the number of channels plus the image shape
        input_shape = [kwargs["n_channels"], *kwargs["image_shape"]]
        if batch_size > 1:
            normalization = 'batch'
        else:
            normalization = 'instance'
        # instantiate new model
        model = isensee2017_model(input_shape=input_shape, n_labels=n_labels,
                                  initial_learning_rate=kwargs["initial_learning_rate"],
                                  n_base_filters=kwargs["n_filters"],
                                  normalization=normalization)
    else:
        model = load_old_model(model_file)
        set_model_learning_rate(model, kwargs['initial_learning_rate'])
    return model


def set_roi(data_file, level, image_shape, crop=True, preload_validation_data=False, threshold=0.5):
    subject_ids = data_file.get_data_groups()
    validation_ids = data_file.get_validation_groups()
    training_ids = data_file.get_training_groups()
    n_subjects = len(subject_ids)
    for index, subject_id in enumerate(subject_ids):
        if crop:
            if level == 0:
                # set ROI for each image in the data file to be the cropped brain
                images = data_file.get_nibabel_images(subject_id)
                image = combine_images(images, axis=0)
                image = move_image_channels(image, axis0=0, axis1=-1)
                roi_affine = compute_region_of_interest_affine([image], image_shape)
            else:
                # set ROI for each image to be the crop from the previous levels prediction/target(validation/training)
                if subject_id in training_ids:
                    # roi is based on ground truth from the previous level
                    _, truth_image = data_file.get_nibabel_images(subject_id)
                    if len(truth_image.shape) > 3:
                        truth_image = truth_image.__class__(truth_image.get_data()[0], truth_image.affine)
                    assert len(truth_image.shape) == 3
                    mask_image = truth_image
                else:
                    # roi is based on previous prediction
                    image = data_file.get_supplemental_image(subject_id, 'level{}_prediction'.format(level - 1))
                    assert len(image.shape) == 3
                    mask_data = np.zeros(image.shape)
                    mask_data[image.get_data() > threshold] = 1
                    mask_image = image.__class__(mask_data, image.affine)
                if np.any(mask_image.get_data() > 0):
                    roi_affine = compute_region_of_interest_affine_from_foreground(mask_image, image_shape)
                else:
                    roi_affine, roi_shape = data_file.get_roi(subject_id)
                    roi_affine = resize_affine(affine=roi_affine, shape=roi_shape, target_shape=image_shape)
        else:
            roi_affine = data_file.get_roi_affine(subject_id)
        roi_affine = reorder_affine(roi_affine, image_shape)
        kwargs = {'level{}_affine'.format(level): roi_affine,
                  'level{}_shape'.format(level): image_shape}
        data_file.add_supplemental_data(subject_id, **kwargs)
        data_file.add_supplemental_data(subject_id, roi_affine=roi_affine)
        data_file.add_supplemental_data(subject_id, roi_shape=image_shape)
        if preload_validation_data and subject_id in validation_ids:
            roi_features, roi_targets = data_file.get_roi_data(subject_id, roi_affine=roi_affine, roi_shape=image_shape)
            data_file.add_supplemental_data(subject_id, roi_features=roi_features, roi_targets=roi_targets)
        update_progress(float(index + 1) / n_subjects)


def create_data_file(folder_path, filename):
    data_file = DataFile(filename)
    for subject_folder in glob.glob(os.path.join(folder_path, "*", "*")):
        subject_id = os.path.basename(subject_folder)
        if not os.path.isdir(subject_folder) or subject_id in data_file.get_data_groups():
            continue
        print("Loading subject: {}".format(subject_id))
        features = list()
        targets = None
        for modality in ["t1", "t1ce", "flair", "t2", "seg"]:
            image_file = glob.glob(os.path.join(subject_folder, "*" + modality + ".nii.gz"))[0]
            print("Reading: {}".format(image_file))
            if modality is "seg":
                targets = nib.load(image_file)
            else:
                features.append(nib.load(image_file))
        data_file.add_nibabel_images(features, targets, subject_id)
        data_file.add_supplemental_data(subject_id, label_map=targets.get_data())
    return data_file


def randomly_set_training_subjects(data_file, split=0.8):
    subject_ids = data_file.get_data_groups()
    training_ids, validation_ids = split_list(subject_ids, split=split)
    data_file.set_training_groups(training_ids)
    data_file.set_validation_groups(validation_ids)


def set_targets(data_file, labels):
    n_subjects = len(data_file.get_data_groups())
    for i, subject_id in enumerate(data_file.get_data_groups()):
        target_data = np.asarray(data_file.get_supplemental_data(subject_id, 'label_map'))
        new_data = np.zeros(target_data.shape, target_data.dtype)
        for label in labels:
            index = target_data == label
            new_data[index] = 1
        data_file.overwrite_array(subject_id, np.asarray([new_data]), 'targets')
        update_progress(float(i + 1)/n_subjects)


def make_predictions(model, data_file, name, normalize_features=True):
    for subject_id in data_file.get_data_groups():
        features_image, targets_image = data_file.get_nibabel_images(subject_id)
        targets_image = move_image_channels(targets_image, axis0=0, axis1=-1)
        assert np.all(np.greater(targets_image.shape[:3], 1))
        features, targets = data_file.get_roi_data(subject_id)
        if normalize_features:
            features = normalize_data(features)
        prediction = model.predict(np.asarray([features]))[0][0]
        p_image = nib.Nifti1Image(prediction, data_file.get_roi_affine(subject_id))
        p_image = resample_to_img(p_image, targets_image, interpolation='linear')
        prediction = p_image.get_data()
        data_file.add_supplemental_data(subject_id, **{name: prediction})


def main(config):
    data_file = create_data_file(os.path.join(os.path.dirname(__file__), "data", 'BRATS2018'), config['data_file'])
    try:
        _ = data_file.get_training_groups(), data_file.get_validation_groups()
    except KeyError:
        print("Splitting training/validation data")
        randomly_set_training_subjects(data_file, split=config['validation_split'])
    for level, (labels, image_shape, model_file, n_filters, batch_size) in enumerate(zip(config["labels"],
                                                                                         config["image_shape"],
                                                                                         config["model_file"],
                                                                                         config["n_base_filters"],
                                                                                         config["batch_sizes"])):
        skip = config["skip_levels"] and level in config["skip_levels"]
        # set targets
        print("Setting the targets to labels {}".format(labels))
        set_targets(data_file, labels)
        print("Setting regions of interest")
        set_roi(data_file, level, image_shape, crop=config['crop'],
                preload_validation_data=config['generator_parameters']['preload_validation_data'])
        # get training and testing generators
        training_ids = data_file.get_training_groups()
        validation_ids = data_file.get_validation_groups()
        data_file.close()

        train_generator, validation_generator = get_generators_from_data_file(data_file.filename,
                                                                              training_ids=training_ids,
                                                                              validation_ids=validation_ids,
                                                                              batch_size=batch_size,
                                                                              **config["generator_parameters"])
        print("Creating/loading model")
        model = get_model(model_file, overwrite=config["overwrite"], image_shape=image_shape, batch_size=batch_size,
                          n_channels=config["n_channels"], n_filters=n_filters,
                          initial_learning_rate=config["training_parameters"]["initial_learning_rate"])
        if config['test_generators']:
            test_generators(train_generator, validation_generator)

        if not skip:
            # run training
            train_model(model=model,
                        model_file=model_file,
                        training_generator=train_generator,
                        validation_generator=validation_generator,
                        steps_per_epoch=len(train_generator),
                        validation_steps=len(validation_generator),
                        **config["training_parameters"])

        data_file = DataFile(data_file.filename, mode='a')
        # make predictions on validation data
        print("Making predictions")
        make_predictions(model, data_file, 'level{}_prediction'.format(level),
                         normalize_features=config['generator_parameters']['normalize'])

    labels = [config['labels'][-1][0]]  # last item should only contain 1 label
    for label_set in config['labels'][:2][::-1]:
        labels.insert(0, set(label_set).difference(set(labels)).pop())

    train_final_model(model_file='final_model.h5',
                      labels=labels,
                      data_file=data_file,
                      training_ids=data_file.get_training_groups(),
                      validation_ids=data_file.get_validation_groups(),
                      image_shape=config['image_shape'][0],
                      n_channels=config['n_channels'] + len(config['model_file']),
                      n_filters=config['n_base_filters'][0],
                      initial_learning_rate=config['training_parameters']['initial_learning_rate'],
                      generator_parameters=config['generator_parameters'],
                      training_parameters=config['training_parameters'])


def train_final_model(data_file, training_ids, validation_ids, labels, model_file='final_model.h5',
                      image_shape=(128, 128, 128), n_channels=7, n_filters=16, initial_learning_rate=1e-4, batch_size=1,
                      generator_parameters=None, roi_level=0, n_levels=3, training_parameters=None):

    # Set ROI and targets
    for subject_id in data_file.get_data_groups():
        roi_affine = np.asarray(data_file.get_supplemental_data(subject_id, 'level{}_affine'.format(roi_level)))
        data_file.add_supplemental_data(subject_id, roi_affine=roi_affine)
        data_file.add_supplemental_data(subject_id, roi_shape=image_shape)

        label_map = np.asarray(data_file.get_supplemental_data(subject_id, 'label_map'))
        target_data = label_map_to_4d_labels(label_map, labels=labels)
        data_file.add_supplemental_data(name=subject_id, targets=target_data)

    data_file.close()
    supplemental_feature_names = ['level{}_prediction'.format(level) for level in range(n_levels)]

    train_generator, validation_generator = get_generators_from_data_file(
        data_file.filename, training_ids=training_ids, validation_ids=validation_ids, batch_size=batch_size,
        supplemental_feature_names=supplemental_feature_names, **generator_parameters)

    test_generators(train_generator, validation_generator)

    final_model = get_model(model_file,
                            n_labels=len(labels),
                            image_shape=image_shape,
                            n_channels=n_channels,
                            n_filters=n_filters,
                            initial_learning_rate=initial_learning_rate,
                            batch_size=batch_size)

    train_model(model=final_model,
                model_file=model_file,
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=len(train_generator),
                validation_steps=len(validation_generator),
                **training_parameters)


def test_generators(train_generator, validation_generator):
    affine = np.diag(np.ones(4))
    for generator, name1 in zip((train_generator, validation_generator), ("training", "validation")):
        for i in range(10):
            for data, name2 in zip(generator.get_batch_from_file(i), ("feature", "target")):
                for j in range(data.shape[1]):
                    image = nib.Nifti1Image(data[0, j], affine)
                    image.to_filename("{}_{}_{}_{}.nii".format(name1, name2, i, j))


if __name__ == "__main__":
    main(load_json(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "cascade_config.json")))
