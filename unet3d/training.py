import os
import math
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.models import load_model

from .generator import pickle_dump
from .model import dice_coef, dice_coef_loss

K.set_image_dim_ordering('th')


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):
    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    logger = CSVLogger(os.path.join(logging_dir, "training.log"))
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay,
                                              initial_lrate=initial_learning_rate,
                                              drop=learning_rate_drop,
                                              epochs_drop=learning_rate_epochs))
    return [model_checkpoint, logger, history, scheduler]


def load_old_model(model_file):
    print("Loading pre-trained model")
    return load_model(model_file,
                      custom_objects={'dice_coef_loss': dice_coef_loss,
                                      'dice_coef': dice_coef})


def train_model(model, model_file, training_generator, testing_generator, steps_per_epoch, validation_steps,
                initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=testing_generator,
                        validation_steps=validation_steps,
                        pickle_safe=True,
                        callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs))
    model.save(model_file)
