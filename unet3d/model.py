import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam

from functools import partial

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32):
    """
    Builds the 3D UNet Keras model.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_number in range(depth):
        layer1 = Conv3D(n_base_filters*(2**layer_number), (3, 3, 3), activation='relu', padding='same')(current_layer)
        layer2 = Conv3D(n_base_filters*(2**layer_number)*2, (3, 3, 3), activation='relu', padding='same')(layer1)
        current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
        levels.append([layer1, layer2, current_layer])

    # add levels with up-convolution or up-sampling
    for layer_number in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, depth=layer_number+1,
                                            n_filters=current_layer._keras_shape[1],
                                            image_shape=input_shape[-3:])(current_layer)
        concat = concatenate([up_convolution, levels[layer_number][-1]], axis=1)
        current_layer = Conv3D(levels[layer_number][1]._keras_shape[1], (3, 3, 3), activation='relu',
                               padding='same')(concat)
        current_layer = Conv3D(levels[layer_number][1]._keras_shape[1], (3, 3, 3), activation='relu',
                               padding='same')(current_layer)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation('sigmoid')(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef] + [partial(label_wise_dice_coefficient,
                                                 label_index=index) for index in range(n_labels)])

    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:, label_index], y_pred[:, label_index])


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(depth, n_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")

        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(n_filters=n_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(n_filters=n_filters,
                                                                                       depth=depth,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)
