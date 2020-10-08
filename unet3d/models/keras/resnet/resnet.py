from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv3D,
    MaxPooling3D,
    AveragePooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.metrics import mean_absolute_error
import keras


def sensitivity(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.backend.epsilon())


def j_stat(y_true, y_pred):
    return sensitivity(y_true, y_pred) + specificity(y_true, y_pred) - 1


def compare_scores(y_true, y_pred, comparison=0, metric=mean_absolute_error):
    keras_comparison = keras.backend.variable(comparison)
    return metric(y_true, y_pred) - metric(y_true, keras_comparison)


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=4)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    stride_depth = int(round(input_shape[3] / residual_shape[3]))
    equal_channels = input_shape[4] == residual_shape[4]

    shortcut = input
    # 1 X 1 conv if reduced_shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual_shape[4],
                          kernel_size=(1, 1, 1),
                          strides=(stride_width, stride_height, stride_depth),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer encoder.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        adjacency_matrix final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, activation="softmax", kernel_initializer='he_normal',
              n_dense_layers=1):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input reduced_shape in the form (nb_rows, nb_cols, nb_z_cols, nb_channels)
            num_outputs: The number of outputs at final activation layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
            activation: default is 'softmax'
            kernel_initializer: default is 'he_normal'

        Returns:
            The keras `Model`.
        """

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2))(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling3D(pool_size=(block_shape[1], block_shape[2], block_shape[3]),
                                 strides=(1, 1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense_input = flatten1
        for i in range(n_dense_layers):
            if (i + 1) < n_dense_layers:
                layer_activation = None
            else:
                layer_activation = activation
            dense_input = Dense(units=num_outputs,
                                kernel_initializer=kernel_initializer,
                                activation=layer_activation)(dense_input)

        model = Model(inputs=input, outputs=dense_input)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, **kwargs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], **kwargs)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, **kwargs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3], **kwargs)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, **kwargs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3], **kwargs)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, **kwargs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3], **kwargs)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, **kwargs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3], **kwargs)
