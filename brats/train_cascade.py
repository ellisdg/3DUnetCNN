import os
import glob
import nibabel as nib
import numpy as np

from tables.exceptions import NoSuchNodeError

from unet3d.data import DataFile, combine_images, move_image_channels
from unet3d.normalize import (compute_region_of_interest_affine, compute_region_of_interest_affine_from_foreground,
                              normalize_data)
from unet3d.generator import get_generators_from_data_file, split_list
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
from unet3d.utils.utils import update_progress
from nilearn.image import resample_to_img

import json


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def get_model(model_file, overwrite=False, **kwargs):
    if not os.path.exists(model_file) or overwrite:
        # set the input shape to be the number of channels plus the image shape
        input_shape = [kwargs["n_channels"], *kwargs["image_shape"]]
        # instantiate new model
        model = isensee2017_model(input_shape=input_shape, n_labels=1,
                                  initial_learning_rate=kwargs["initial_learning_rate"],
                                  n_base_filters=kwargs["n_filters"],
                                  instance_normalization=True)
    else:
        model = load_old_model(model_file)
    return model


def set_roi(data_file, level, image_shape, crop=True):
    if level == 0 and crop:
        # set ROI for each image in the data file to be the cropped brain
        subject_ids = data_file.get_data_groups()
        n_subjects = len(subject_ids)
        for index, subject_id in enumerate(subject_ids):
            images = data_file.get_nibabel_images(subject_id)
            image = combine_images(images, axis=0)
            image = move_image_channels(image, axis0=0, axis1=-1)
            roi_affine = compute_region_of_interest_affine([image], image_shape)
            kwargs = {'level{}_affine'.format(level): roi_affine,
                      'level{}_shape'.format(level): image_shape}
            data_file.add_supplemental_data(subject_id, roi_affine=roi_affine, roi_shape=image_shape, **kwargs)
            update_progress(float(index + 1)/n_subjects)
    elif level > 0:
        # set ROI for each image to be the crop from the previous levels prediction/target(validation/training)
        print("Setting ROI for training subjects")
        for subject_id in data_file.get_training_groups():
            print("Subject: {}".format(subject_id))
            # roi is based on ground truth from the previous level
            _, truth_image = data_file.get_nibabel_images(subject_id)
            roi_affine = compute_region_of_interest_affine_from_foreground(truth_image, image_shape)
            kwargs = {'level{}_affine'.format(level): roi_affine,
                      'level{}_shape'.format(level): image_shape}
            data_file.add_supplemental_data(subject_id, **kwargs)
            data_file.overwrite_array(subject_id, roi_affine, 'roi_affine')
            data_file.overwrite_array(subject_id, image_shape, 'roi_shape')
        print("Setting ROI for validation subjects")
        for subject_id in data_file.get_validation_groups():
            print("Subject: {}".format(subject_id))
            # roi is based on previous prediction
            image = data_file.get_supplemental_image(subject_id, 'level{}_prediction'.format(level - 1))
            roi_affine = compute_region_of_interest_affine([image], image_shape)
            kwargs = {'level{}_affine'.format(level): roi_affine,
                      'level{}_shape'.format(level): image_shape}
            data_file.add_supplemental_data(subject_id, **kwargs)
            data_file.overwrite_array(subject_id, roi_affine, 'roi_affine')
            data_file.overwrite_array(subject_id, image_shape, 'roi_shape')


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


def predict_validation_data(model, data_file, name, normalize_features=True, threshold=0.5):
    for subject_id in data_file.get_validation_groups():
        features_image, targets_image = data_file.get_nibabel_images(subject_id)
        assert len(targets_image.shape) == 3
        features, targets = data_file.get_roi_data(subject_id)
        if normalize_features:
            features = normalize_data(features)
        prediction = model.predict([features])[0][0]
        p_image = nib.Nifti1Image(prediction, data_file.get_roi_affine(subject_id))
        p_image = resample_to_img(p_image, targets_image, interpolation='linear')
        prediction = np.zeros(targets_image.shape)
        index = p_image.get_data() >= threshold
        prediction[index] = 1
        data_file.add_supplemental_data(subject_id, {name: prediction})


def main(config):
    data_file = create_data_file(os.path.join(os.path.dirname(__file__), "data", 'BRATS2018'), config['data_file'])
    try:
        _ = data_file.get_training_groups(), data_file.get_validation_groups()
    except NoSuchNodeError:
        print("Splitting training/validation data")
        randomly_set_training_subjects(data_file, split=config['validation_split'])
    for level, (labels, image_shape, model_file, n_filters) in enumerate(zip(config["labels"], config["image_shape"],
                                                                             config["model_file"],
                                                                             config["n_base_filters"])):
        # set targets
        print("Setting the targets to labels {}".format(labels))
        set_targets(data_file, labels)
        print("Setting regions of interest")
        set_roi(data_file, level, image_shape, crop=config['crop'])
        print("Creating model")
        model = get_model(model_file, overwrite=config["overwrite"], image_shape=image_shape,
                          n_channels=config["n_channels"], n_filters=n_filters,
                          initial_learning_rate=config["training_parameters"]["initial_learning_rate"])
        # get training and testing generators

        train_generator, validation_generator = get_generators_from_data_file(data_file,
                                                                              **config["generator_parameters"])

        # run training
        train_model(model=model,
                    model_file=config["model_file"],
                    training_generator=train_generator,
                    validation_generator=validation_generator,
                    steps_per_epoch=len(data_file.get_training_groups()),
                    validation_steps=len(data_file.get_validation_groups()),
                    **config["training_parameters"])

        # make predictions on validation data
        print("Making predictions on validation data")
        predict_validation_data(model, data_file, 'level{}_prediction'.format(level),
                                normalize_features=config['training_parameters']['normalize'])


if __name__ == "__main__":
    main(load_json(os.path.join(os.path.dirname(__file__), "cascade_config.json")))
