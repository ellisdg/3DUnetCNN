from functools import partial

from torch import nn

from .myronenko import MyronenkoLayer, MyronenkoResidualBlock
from . import resnet


class BasicDecoder(nn.Module):
    def __init__(self, in_planes, layers, block=resnet.BasicBlock, plane_dilation=2, upsampling_mode="trilinear",
                 upsampling_scale=2):
        super(BasicDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.upsampling_mode = upsampling_mode
        self.upsampling_scale = upsampling_scale
        layer_planes = in_planes
        for n_blocks in layers:
            self.conv1s.append(resnet.conv1x1x1(in_planes=layer_planes,
                                                out_planes=int(layer_planes/plane_dilation)))
            layer = nn.ModuleList()
            layer_planes = int(layer_planes/plane_dilation)
            for i_block in range(n_blocks):
                layer.append(block(layer_planes, layer_planes))
            self.layers.append(layer)

    def forward(self, x):
        for conv1, layer in zip(self.conv1s, self.layers):
            x = conv1(x)
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_scale, mode=self.upsampling_mode)
            for block in layer:
                x = block(x)
        return x


class MyronenkoDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernal_size=3):
        super(MyronenkoDecoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 1, 1]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            if layer_widths is not None:
                out_width = layer_widths[depth]
                in_width = layer_widths[depth + 1]
            else:
                out_width = base_width * (feature_reduction_scale ** depth)
                in_width = out_width * feature_reduction_scale
            if use_transposed_convolutions:
                self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))
            else:
                self.pre_upsampling_blocks.append(nn.Sequential())
                self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernal_size,
                                                                 stride=upsampling_scale, padding=1))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=out_width, planes=out_width,
                                     kernal_size=kernal_size))

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers):
            x = pre(x)
            x = up(x)
            x = lay(x)
        return x


class MirroredDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernel_size=3):
        super(MirroredDecoder, self).__init__()
        self.use_transposed_convolutions = use_transposed_convolutions
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth != 0:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=in_width,
                                         kernel_size=kernel_size))
                if self.use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=kernel_size,
                                                                     stride=upsampling_scale, padding=1))
                else:
                    self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                          mode=upsampling_mode, align_corners=align_corners))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        if self.layer_widths is not None:
            out_width = self.layer_widths[depth]
            in_width = self.layer_widths[depth + 1]
        else:
            if depth > 0:
                out_width = int(self.base_width * (self.feature_reduction_scale ** (depth - 1)))
                in_width = out_width * self.feature_reduction_scale
            else:
                out_width = self.base_width
                in_width = self.base_width
        return in_width, out_width

    def forward(self, x):
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1]):
            x = lay(x)
            x = pre(x)
            x = up(x)
        x = self.layers[-1](x)
        return x


class Decoder1D(nn.Module):
    def __init__(self, input_features, output_features, layer_blocks, layer_channels, block=resnet.BasicBlock1D,
                 kernel_size=3, upsample_factor=2, interpolation_mode="linear", interpolation_align_corners=True):
        super(Decoder1D, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.output_features = output_features
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.upsample_factor = upsample_factor
        in_channels = input_features
        for n_blocks, out_channels in zip(layer_blocks, layer_channels):
            layer = nn.ModuleList()
            self.conv1s.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                         stride=1, bias=False))
            for i_block in range(n_blocks):
                layer.append(block(in_channels=out_channels, channels=out_channels, kernel_size=kernel_size, stride=1))
            in_channels = out_channels
            self.layers.append(layer)

    def forward(self, x):
        for (layer, conv1) in zip(self.layers, self.conv1s):
            x = nn.functional.interpolate(x,
                                          size=(x.shape[-1] * self.upsample_factor),
                                          mode=self.interpolation_mode,
                                          align_corners=self.interpolation_align_corners)
            x = conv1(x)
            for block in layer:
                x = block(x)
        return x
