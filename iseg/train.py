import glob
import os

import tables

from unet3d.data import write_data_to_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model

config = dict()
config["pool_size"] = (2, 2, 2)
config["image_shape"] = (128, 144, 112)  # This determines what shape the images will be cropped/resampled to.
# image shape must be in multiples of 16
config["labels"] = (10, 150, 250)
config["n_labels"] = len(config["labels"])
config["batch_size"] = 1
config["n_epochs"] = 3000
config["decay_learning_rate_every_x_epochs"] = 500
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5
config["validation_split"] = 0.8
config["hdf5_file"] = "./data.hdf5"
config["model_file"] = "./3d_unet_model.h5"
config["training_file"] = "./training_ids.pkl"
config["validation_file"] = "./testing_ids.pkl"
config["n_channels"] = 2
config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))
config["augment"] = True
config["augment_flip"] = True
config["augment_distortion"] = None
config["deconvolution"] = False
config["metrics"] = []


def main():
    if not os.path.exists(config["hdf5_file"]):
        training_files = list()
        for label_file in glob.glob("/home/aizenberg/dgellis/3DUnetCNN/iseg/data/training/subject-*-label.hdr"):
            training_files.append((label_file.replace("label", "T1"), label_file.replace("label", "T2"), label_file))

        write_data_to_file(training_files, config["hdf5_file"], image_shape=config["image_shape"])

    hdf5_file_opened = tables.open_file(config["hdf5_file"], "r")

    if os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"], n_labels=config["n_labels"],
                              deconvolution=config["deconvolution"],
                              metrics=config["metrics"])

    # get training and testing generators
    train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
        hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"],
        validation_keys_file=config["validation_file"], training_keys_file=config["training_file"],
        n_labels=config["n_labels"], labels=config["labels"], augment=config["augment"],
        augment_flip=config["augment_flip"], augment_distortion_factor=config["augment_distortion"])

    # run training
    train_model(model=model, model_file=config["model_file"], training_generator=train_generator,
                validation_generator=validation_generator, steps_per_epoch=nb_train_samples,
                validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])
    hdf5_file_opened.close()

if __name__ == "__main__":
    main()
