import os
import glob

import tables

from unet3d.data import write_data_to_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model


config = dict()
config["pool_size"] = (2, 2, 2)
config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (64, 64, 64)
config["labels"] = (1, 2, 3, 4)
config["n_labels"] = len(config["labels"])
config["modalities"] = ["T1", "T1c", "Flair", "T2"]
config["training_modalities"] = ["T1", "T1c", "Flair"]  # set this to the modalities that you want the model to use
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True

config["batch_size"] = 6
config["validation_batch_size"] = 12
config["n_epochs"] = 500
config["patience"] = 10
config["early_stop"] = 50
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5
config["validation_split"] = 0.8
config["flip"] = True
config["distort"] = None  # switch to None if you want no distortion
config["validation_patch_overlap"] = 0
config["training_patch_start_offset"] = (16, 16, 16)

config["hdf5_file"] = os.path.abspath("brats_data.hdf5")
config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")


def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["hdf5_file"]):
        training_files = fetch_training_data_files()

        write_data_to_file(training_files, config["hdf5_file"], image_shape=config["image_shape"])
    hdf5_file_opened = tables.open_file(config["hdf5_file"], "r")

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"])

    # get training and testing generators
    train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
        hdf5_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=nb_train_samples,
                validation_steps=nb_test_samples,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    hdf5_file_opened.close()


if __name__ == "__main__":
    main(overwrite=False)
