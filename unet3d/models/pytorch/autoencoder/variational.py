from functools import partial

import numpy as np
import torch.nn as nn
import torch

from unet3d.models.pytorch.classification.decoder import MyronenkoDecoder, MirroredDecoder
from unet3d.models.pytorch.classification.myronenko import MyronenkoEncoder, MyronenkoConvolutionBlock
from unet3d.models.pytorch.classification.resnet import conv1x1x1


class VariationalBlock(nn.Module):
    def __init__(self, in_size, n_features, out_size, return_parameters=False):
        super(VariationalBlock, self).__init__()
        self.n_features = n_features
        self.return_parameters = return_parameters
        self.dense1 = nn.Linear(in_size, out_features=n_features*2)
        self.dense2 = nn.Linear(self.n_features, out_size)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.dense1(x)
        mu, logvar = torch.split(x, self.n_features, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.dense2(z)
        if self.return_parameters:
            return out, mu, logvar, z
        else:
            return out, mu, logvar


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                 encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                 decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.base_width = base_width
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder_class(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size)
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                                decoder_blocks)
        self.decoder = decoder_class(base_width=base_width, layer_blocks=decoder_blocks,
                                     upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)
        self.set_final_convolution(n_features)
        self.set_activation(activation=activation)

    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            if decoder_class is None:
                decoder_class = MirroredDecoder
        elif decoder_blocks is None:
            decoder_blocks = [1] * len(encoder_blocks)
            if decoder_class is None:
                decoder_class = MyronenkoDecoder
        return decoder_class, decoder_blocks

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MyronenkoVariationalLayer(nn.Module):
    def __init__(self, in_features, input_shape, reduced_features=16, latent_features=128,
                 conv_block=MyronenkoConvolutionBlock, conv_stride=2, upsampling_mode="trilinear",
                 align_corners_upsampling=False):
        super(MyronenkoVariationalLayer, self).__init__()
        self.in_conv = conv_block(in_planes=in_features, planes=reduced_features, stride=conv_stride)
        self.reduced_shape = tuple(np.asarray((reduced_features, *np.divide(input_shape, conv_stride)), dtype=np.int))
        self.in_size = np.prod(self.reduced_shape, dtype=np.int)
        self.var_block = VariationalBlock(in_size=self.in_size, out_size=self.in_size, n_features=latent_features)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = conv1x1x1(in_planes=reduced_features, out_planes=in_features, stride=1)
        self.upsample = partial(nn.functional.interpolate, scale_factor=conv_stride, mode=upsampling_mode,
                                align_corners=align_corners_upsampling)

    def forward(self, x):
        x = self.in_conv(x).flatten(start_dim=1)
        x, mu, logvar = self.var_block(x)
        x = self.relu(x).view(-1, *self.reduced_shape)
        x = self.out_conv(x)
        x = self.upsample(x)
        return x, mu, logvar


class VariationalAutoEncoder(ConvolutionalAutoEncoder):
    def __init__(self, n_reduced_latent_feature_maps=16, vae_features=128, variational_layer=MyronenkoVariationalLayer,
                 input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear", encoder_class=MyronenkoEncoder,
                 decoder_class=MyronenkoDecoder, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False,
                 activation=None, use_transposed_convolutions=False, var_layer_stride=2):
        super(VariationalAutoEncoder, self).__init__(input_shape=input_shape, n_features=n_features,
                                                     base_width=base_width, encoder_blocks=encoder_blocks,
                                                     decoder_blocks=decoder_blocks, feature_dilation=feature_dilation,
                                                     downsampling_stride=downsampling_stride,
                                                     interpolation_mode=interpolation_mode, encoder_class=encoder_class,
                                                     decoder_class=decoder_class, n_outputs=n_outputs, layer_widths=layer_widths,
                                                     decoder_mirrors_encoder=decoder_mirrors_encoder,
                                                     activation=activation,
                                                     use_transposed_convolutions=use_transposed_convolutions)
        if vae_features is not None:
            depth = len(encoder_blocks) - 1
            n_latent_feature_maps = base_width * (feature_dilation ** depth)
            latent_image_shape = np.divide(input_shape, downsampling_stride ** depth)
            self.var_layer = variational_layer(in_features=n_latent_feature_maps,
                                               input_shape=latent_image_shape,
                                               reduced_features=n_reduced_latent_feature_maps,
                                               latent_features=vae_features,
                                               upsampling_mode=interpolation_mode,
                                               conv_stride=var_layer_stride)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar

    def test(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(mu)
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mu, logvar


class LabeledVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self, *args, n_outputs=None, base_width=32, **kwargs):
        super().__init__(*args, n_outputs=n_outputs, base_width=base_width, **kwargs)
        self.final_convolution = conv1x1x1(in_planes=base_width, out_planes=n_outputs, stride=1)
