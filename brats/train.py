import os

import tables

from brats.config import config
from unet3d.data import write_data_to_file
from unet3d.generator import get_training_and_testing_generators
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["hdf5_file"]):
        write_data_to_file(config["data_dir"],
                           config["hdf5_file"],
                           image_shape=config["image_shape"],
                           n_channels=config["nb_channels"])
    hdf5_file_opened = tables.open_file(config["hdf5_file"], "r")

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d()

    # get training and testing generators
    train_generator, test_generator, nb_train_samples, nb_test_samples = get_training_and_testing_generators(
        hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"], overwrite=overwrite)

    # run training
    train_model(model, config["model_file"], train_generator, test_generator, nb_train_samples, nb_test_samples)
    hdf5_file_opened.close()


if __name__ == "__main__":
    main(overwrite=False)