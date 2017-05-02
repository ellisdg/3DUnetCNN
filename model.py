from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam

try:
    from keras.engine import merge
except ImportError:
    from keras.layers import merge

from config import config


def unet_model_3d():
    inputs = Input(config["input_shape"])
    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=config["pool_size"])(conv1)

    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=config["pool_size"])(conv2)

    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=config["pool_size"])(conv3)

    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    up4 = merge([UpSampling3D(size=config["pool_size"])(conv4), conv3], mode='concat', concat_axis=1)

    conv5 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(up4)
    conv5 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv5)
    up5 = merge([UpSampling3D(size=config["pool_size"])(conv5), conv2], mode='concat', concat_axis=1)

    conv6 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(up5)
    conv6 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv6)
    up6 = merge([UpSampling3D(size=config["pool_size"])(conv6), conv1], mode='concat', concat_axis=1)

    conv7 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(up6)
    conv7 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    conv8 = Conv3D(config["n_labels"], 1, 1, 1)(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(input=inputs, output=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
