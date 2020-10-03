'''
Squeeze-and-Excitation ResNets

References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - []() # added when paper is published on Arxiv
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import GlobalMaxPooling3D
from keras.layers import Conv3D
from keras.layers import add
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs
from keras import backend as K
from .se import squeeze_excite_block

WEIGHTS_PATH = ""
WEIGHTS_PATH_NO_TOP = ""


def SEResNet(input_shape=None,
             num_outputs=1000,
             initial_conv_filters=64,
             depth=(3, 4, 6, 3),
             filters=(64, 128, 256, 512),
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             **kwargs):
    """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filter: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            width: width multiplier for the network (for Wide ResNets)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 3D tensor.
                - `max` means that global max pooling will
                    be applied.
            num_outputs: optional number of num_outputs to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            adjacency_matrix Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and num_outputs != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_se_resnet(num_outputs, img_input, include_top, initial_conv_filters,
                          filters, depth, width, bottleneck, weight_decay, pooling, **kwargs)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

    # load weights

    return model


def SEResNet18(input_shape=None,
               num_outputs=1000,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               **kwargs):
    return SEResNet(input_shape,
                    depth=[2, 2, 2, 2],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    num_outputs=num_outputs,
                    **kwargs)


def SEResNet34(input_shape=None,
               num_outputs=1000,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               **kwargs):
    return SEResNet(input_shape,
                    depth=[3, 4, 6, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    num_outputs=num_outputs,
                    **kwargs)


def SEResNet50(input_shape=None,
               num_outputs=1000,
               width=1,
               bottleneck=True,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               **kwargs):
    return SEResNet(input_shape,
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    num_outputs=num_outputs,
                    **kwargs)


def SEResNet101(input_shape=None,
                num_outputs=1000,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                **kwargs):
    return SEResNet(input_shape,
                    depth=[3, 6, 23, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    num_outputs=num_outputs,
                    **kwargs)


def SEResNet154(input_shape=None,
                num_outputs=1000,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                **kwargs):
    return SEResNet(input_shape,
                    depth=[3, 8, 36, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    num_outputs=num_outputs,
                    **kwargs)


def _resnet_block(input, filters, k=1, strides=(1, 1, 1)):
    ''' Adds a pre-activation encoder block without bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv3D(filters * k, (1, 1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv3D(filters * k, (3, 3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv3D(filters * k, (3, 3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation encoder block with bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1, 1) or init._keras_shape[channel_axis] != bottleneck_expand * filters * k:
        init = Conv3D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv3D(filters * k, (1, 1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv3D(filters * k, (3, 3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv3D(bottleneck_expand * filters * k, (1, 1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _create_se_resnet(num_outputs, img_input, include_top, initial_conv_filters, filters,
                      depth, width, bottleneck, weight_decay, pooling, activation="softmax",
                      padding="same"):
    """Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last fc layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 3D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv3D(initial_conv_filters, (7, 7, 7), padding=padding, use_bias=False, strides=(2, 2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding=padding)(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling3D()(x)
        x = Dense(num_outputs, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation=activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)

    return x


class SEResnetBuilder(object):
    @staticmethod
    def build_se_resnet_18(*args, **kwargs):
        return SEResNet18(*args, **kwargs)

    @staticmethod
    def build_se_resnet_34(*args, **kwargs):
        return SEResNet34(*args, **kwargs)

    @staticmethod
    def build_se_resnet_50(*args, **kwargs):
        return SEResNet50(*args, **kwargs)

    @staticmethod
    def build_se_resnet_101(*args, **kwargs):
        return SEResNet101(*args, **kwargs)

    @staticmethod
    def build_se_resnet_154(*args, **kwargs):
        return SEResNet154(*args, **kwargs)
