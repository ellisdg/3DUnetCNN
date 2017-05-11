import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

from unet3d.config import config

if config["deconvolution"]:
    from keras_contrib.layers import Deconvolution3D


def unet_model_3d():
    inputs = Input(config["input_shape"])
    conv1 = Conv3D(int(32/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=config["pool_size"])(conv1)

    conv2 = Conv3D(int(64/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=config["pool_size"])(conv2)

    conv3 = Conv3D(int(128/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=config["pool_size"])(conv3)

    conv4 = Conv3D(int(256/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv4)

    up5 = get_upconv(depth=2, nb_filters=int(512/config["downsize_nb_filters_factor"]))(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(int(256/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv5)

    up6 = get_upconv(depth=1, nb_filters=int(256/config["downsize_nb_filters_factor"]))(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(128/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv6)

    up7 = get_upconv(depth=0, nb_filters=int(128/config["downsize_nb_filters_factor"]))(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(64/config["downsize_nb_filters_factor"]), (3, 3, 3), activation='relu',
                   padding='same')(conv7)

    conv8 = Conv3D(config["n_labels"], (1, 1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def compute_level_output_shape(filters, depth):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(config["image_shape"], np.multiply(config["pool_size"], depth)).tolist()
    else:
        output_image_shape = config["image_shape"]
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
    if config["deconvolution"]:
        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1))
    else:
        return UpSampling3D(size=config["pool_size"])
