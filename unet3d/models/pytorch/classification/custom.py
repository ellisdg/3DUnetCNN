from torch import nn

from unet3d.models.pytorch.autoencoder.variational import VariationalAutoEncoder
from .decoder import BasicDecoder, Decoder1D
from .resnet import conv1x1x1, ResNet, BasicBlock


class RegularizedResNet(VariationalAutoEncoder):
    def __init__(self, n_outputs, *args, **kwargs):
        super(RegularizedResNet, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.var_layer.in_size, n_outputs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.var_layer.in_conv(x).flatten(start_dim=1)
        output = self.fc(x)
        x, mu, logvar = self.var_layer.var_block(x)
        x = self.var_layer.relu(x).view(-1, *self.var_layer.reduced_shape)
        x = self.var_layer.out_conv(x)
        x = self.var_layer.upsample(x)
        x = self.decoder(x)
        vae_output = self.final_convolution(x)
        return output, vae_output, mu, logvar


class RegularizedBasicResNet(nn.Module):
    def __init__(self, n_features, upsampling_mode="trilinear", upsampling_scale=2, plane_dilation=2,
                 decoding_layers=None, latent_planes=512, layer_block=BasicBlock, **encoder_kwargs):
        super(RegularizedBasicResNet, self).__init__()
        if decoding_layers is None:
            decoding_layers = [1, 1, 1, 1, 1, 1, 1]
        self.encoder = _ResNetLatent(block=layer_block, n_features=n_features, **encoder_kwargs)
        self.decoder = BasicDecoder(upsampling_scale=upsampling_scale, upsampling_mode=upsampling_mode,
                                    plane_dilation=plane_dilation, layers=decoding_layers, in_planes=latent_planes,
                                    block=layer_block)
        out_decoder_planes = int(latent_planes/(plane_dilation**len(decoding_layers)))
        self.final_convolution = conv1x1x1(in_planes=out_decoder_planes, out_planes=n_features, stride=1)

    def forward(self, x):
        out, x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return out, x


class _ResNetLatent(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latent = x
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, latent


class ResNetWithDecoder1D(nn.Module):
    def __init__(self, n_fc_outputs, n_outputs, initial_upsample=1024, blocks_per_layer=1, channel_decay=2,
                 upsample_factor=2, resnet_block=BasicBlock, interpolation_mode="linear",
                 interpolation_align_corners=True, **kwargs):
        super(ResNetWithDecoder1D, self).__init__()
        self.encoder = ResNet(n_outputs=n_fc_outputs, block=resnet_block, **kwargs)
        self.initial_upsample = initial_upsample
        _size = initial_upsample
        _channels = n_fc_outputs
        layer_blocks = list()
        layer_channels = list()
        while _size < n_outputs:
            _size = int(_size * upsample_factor)
            _channels = int(_channels/channel_decay)
            layer_blocks.append(blocks_per_layer)
            layer_channels.append(_channels)
        self.decoder = Decoder1D(input_features=n_fc_outputs, output_features=n_outputs, layer_blocks=layer_blocks,
                                 layer_channels=layer_channels, upsample_factor=upsample_factor,
                                 interpolation_mode=interpolation_mode,
                                 interpolation_align_corners=interpolation_align_corners)
        self.out_conv = nn.Conv1d(in_channels=layer_channels[-1], out_channels=1, kernel_size=3, bias=False)
        self.output_features = n_outputs
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x.flatten(start_dim=1)[..., None], size=(self.initial_upsample,))
        x = self.decoder(x)
        x = self.out_conv(x)
        return nn.functional.interpolate(x,
                                         size=(self.output_features,),
                                         mode=self.interpolation_mode,
                                         align_corners=self.interpolation_align_corners)
