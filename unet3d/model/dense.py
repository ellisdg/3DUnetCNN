from keras import backend as K
from keras.engine import Input, Model
from keras.layers import (Conv3D, MaxPooling3D, Activation, BatchNormalization, SpatialDropout3D, Conv3DTranspose)
from keras.optimizers import Adam

from unet3d.metrics import weighted_dice_coefficient_loss

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def dense_unet(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, depth=4, n_base_filters=32,
               loss=weighted_dice_coefficient_loss, activation_name="sigmoid", layer_activation='relu',
               layer_kernel_size=(3, 3, 3), data_format="channels_first", dropout_rate=0.2,
               normalization=BatchNormalization):

    K.set_image_data_format(data_format=data_format)
    if data_format == "channels_first":
        modalities_axis = 1
    else:
        modalities_axis = -1

    inputs = Input(input_shape)
    input_convolution = Conv3D(n_base_filters, kernel_size=layer_kernel_size, padding='same')(inputs)

    output_level = create_levels(input_convolution, n_levels=depth, n_filters=n_base_filters*2, n_layers=depth,
                                 concatenation_axis=modalities_axis, normalization=normalization,
                                 data_format=data_format, dropout_rate=dropout_rate, activation=layer_activation,
                                 pool_size=pool_size, normalization_axis=modalities_axis, kernel_size=layer_kernel_size)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(output_level)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss)
    return model


def create_levels(input_node, n_levels, n_filters, n_layers, concatenation_axis=1, normalization=BatchNormalization,
                  data_format="channels_first", dropout_rate=0.2, activation='relu', pool_size=(2, 2, 2),
                  normalization_axis=1, kernel_size=(3, 3, 3)):
    down_dense_block = create_dense_block(input_node, n_layers, n_filters, concatenation_axis=concatenation_axis)
    if n_levels > 1:
        down_concatenation_block = concatenate([input_node, down_dense_block], axis=concatenation_axis)
        down_transition_block = create_transition_down(down_concatenation_block, n_filters, normalization=normalization,
                                                       data_format=data_format, dropout_rate=dropout_rate,
                                                       activation=activation, pool_size=pool_size,
                                                       normalization_axis=normalization_axis)
        lower_level_output_node = create_levels(down_transition_block, n_levels-1, n_filters*2, n_layers+1,
                                                concatenation_axis=concatenation_axis)
        up_transition_block = create_transition_up(lower_level_output_node, n_filters, kernel_size=kernel_size)
        up_concatenation_block = concatenate([down_concatenation_block, up_transition_block], axis=concatenation_axis)
        up_dense_block = create_dense_block(up_concatenation_block, n_layers=n_layers, n_filters=n_filters,
                                            concatenation_axis=concatenation_axis)
        return up_dense_block
    else:
        return down_dense_block


def create_layer(input_node, n_filters, kernel=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 normalization=BatchNormalization, dropout_rate=0.2, data_format="channels_first",
                 normalization_axis=1):
    norm = normalization(axis=normalization_axis)(input_node)
    act = Activation('relu')(norm)
    conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(act)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(conv)
    return dropout


def create_dense_block(input_node, n_layers, n_filters, concatenation_axis=1):
    layers = [input_node]
    current_node = input_node
    for layer_index in range(n_layers):
        layers.append(create_layer(current_node, n_filters))
        if layer_index < (n_layers - 1):
            current_node = concatenate(layers, axis=concatenation_axis)
        else:
            current_node = concatenate(layers[1:], axis=concatenation_axis)
    return current_node


def create_transition_down(input_node, n_filters, normalization=BatchNormalization, activation='relu', dropout_rate=0.2,
                           pool_size=(2, 2, 2), normalization_axis=1, data_format="channels_first"):
    norm = normalization(axis=normalization_axis)(input_node)
    act = Activation(activation)(norm)
    conv = Conv3D(n_filters, (1, 1, 1))(act)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(conv)
    pool = MaxPooling3D(pool_size=pool_size)(dropout)
    return pool


def create_transition_up(input_node, n_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2)):
    return Conv3DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(input_node)
