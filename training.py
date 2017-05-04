import math
import os
from functools import partial

import tables
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.models import load_model

from generator import get_training_and_testing_generators, pickle_dump
from config import config
from model import unet_model_3d, dice_coef, dice_coef_loss
from data import write_data_to_file

K.set_image_dim_ordering('th')


# learning rate schedule
def step_decay(epoch, initial_lrate=config["initial_learning_rate"], drop=config["learning_rate_drop"],
               epochs_drop=config["decay_learning_rate_every_x_epochs"]):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


def get_callbacks(model_file):
    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    logger = CSVLogger("training.log")
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay,
                                              initial_lrate=config["initial_learning_rate"],
                                              drop=config["learning_rate_drop"],
                                              epochs_drop=config["decay_learning_rate_every_x_epochs"]))
    return [model_checkpoint, logger, history, scheduler]


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["hdf5_file"]):
        write_data_to_file(config["data_dir"],
                           config["hdf5_file"],
                           image_shape=config["image_shape"],
                           nb_channels=config["nb_channels"])
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


def load_old_model(model_file):
    print("Loading pre-trained model")
    return load_model(model_file,
                      custom_objects={'dice_coef_loss': dice_coef_loss,
                                      'dice_coef': dice_coef})


def train_model(model, model_file, training_generator, testing_generator, steps_per_epoch, validation_steps):
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        nb_epoch=config["n_epochs"],
                        validation_data=testing_generator,
                        validation_steps=validation_steps,
                        pickle_safe=True,
                        callbacks=get_callbacks(model_file))
    model.save(model_file)


if __name__ == "__main__":
    main(overwrite=False)
