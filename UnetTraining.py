import os
import math
from functools import partial

import numpy as np
from keras import backend as K
from keras.layers import (Conv3D, MaxPooling3D, Activation, UpSampling3D, merge, Input, Reshape)
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler

from DataGenerator import get_training_and_testing_generators, pickle_dump

pool_size = (2, 2, 2)
image_shape = (240, 240, 144)
n_channels = 3
input_shape = tuple([n_channels] + list(image_shape))
n_labels = 1  # not including background
batch_size = 1
n_epochs = 50
data_dir = "/home/neuro-user/PycharmProjects/BRATS/data"
truth_channel = 3
background_channel = 4
decay_learning_rate_every_x_epochs = 1
initial_learning_rate = 0.1
learning_rate_drop = 0.5
validation_split = 0.8


# learning rate schedule
def step_decay(epoch, initial_lrate=initial_learning_rate, drop=learning_rate_drop,
               epochs_drop=decay_learning_rate_every_x_epochs):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


K.set_image_dim_ordering('th')
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model():
    inputs = Input(input_shape)
    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(conv4)

    conv5 = Conv3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling3D(size=pool_size)(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling3D(size=pool_size)(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling3D(size=pool_size)(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling3D(size=pool_size)(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv3D(n_labels, 1, 1, 1)(conv9)
    act = Activation('sigmoid')(conv10)
    model = Model(input=inputs, output=act)

    model.compile(optimizer=SGD(lr=initial_learning_rate, decay=0.0, momentum=0.9), loss=dice_coef_loss,
                  metrics=[dice_coef])

    return model


def get_training_weights(training_generator, nb_training_samples, n_classes=2):
    counts = np.zeros((n_labels, n_classes))
    i = 0
    while i < nb_training_samples:
        _, y_train = training_generator.next()
        if n_labels > 1:
            for label in range(n_labels):
                counts[label, :] += np.bincount(np.array(y_train[:, label].ravel(), dtype=np.uint8),
                                                minlength=n_classes)
        else:
            counts += np.bincount(np.array(y_train.ravel(), dtype=np.uint8), minlength=n_classes)
        i += 1
    print("Class Counts: {0}".format(counts))
    return counts_to_weights(counts)


def counts_to_weights(array):
    weights_list = []
    for label in range(array.shape[0]):
        weights = float(array[label, :].max())/array[label, :]
        weights_list.append({0: weights[0], 1: weights[1]})
    if len(weights_list) == 1:
        return weights_list[0]
    else:
        return weights_list


def get_training_weights_from_data(y_train):
    weights = []
    for label in range(y_train.shape[1]):
        background, foreground = get_class_weights(y_train[:, label], n_classes=2)
        weights.append({0: background, 1: foreground})
    print("Training weights: {0}".format(weights))
    if len(weights) == 1:
        return weights[0]
    else:
        return weights


def get_class_weights(labels_array, n_classes):
    counts = np.bincount(np.array(labels_array.ravel(), dtype=np.uint8), minlength=n_classes)
    weights = np.zeros(n_classes)
    weights[counts > 0] = float(counts.max())/counts[counts > 0]
    return weights


class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


def get_callbacks(model_file):
    epoch_model_file = model_file.replace(".h", "_latest.h")
    model_checkpoint = ModelCheckpoint(epoch_model_file)
    logger = CSVLogger("training.log")
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate, drop=learning_rate_drop,
                                              epochs_drop=decay_learning_rate_every_x_epochs))
    return [model_checkpoint, logger, history, scheduler]


def main(overwrite=False):
    model_file = os.path.abspath("3d_unet_model.h5")
    if not overwrite and os.path.exists(model_file):
        print("Loading pre-trained model")
        model = load_model(model_file, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model()
    train_model(model, model_file)


def train_model(model, model_file):
    training_generator, testing_generator, nb_training_samples, nb_testing_samples = get_training_and_testing_generators(
        data_dir=data_dir, batch_size=batch_size, nb_channels=n_channels, input_shape=image_shape,
        validation_split=validation_split)

    model.fit_generator(generator=training_generator,
                        samples_per_epoch=nb_training_samples,
                        nb_epoch=n_epochs,
                        validation_data=testing_generator,
                        nb_val_samples=nb_testing_samples,
                        pickle_safe=True,
                        callbacks=get_callbacks(model_file))
    model.save(model_file)


if __name__ == "__main__":
    main(overwrite=False)
