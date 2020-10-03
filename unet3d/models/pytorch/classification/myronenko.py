from torch import nn as nn
from .resnet import conv3x3x3, conv1x1x1


class MyronenkoConvolutionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoConvolutionBlock, self).__init__()
        self.norm_groups = norm_groups
        if norm_layer is None:
            self.norm_layer = nn.GroupNorm
        else:
            self.norm_layer = norm_layer
        self.norm1 = self.create_norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3x3(in_planes, planes, stride, kernel_size=kernel_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

    def create_norm_layer(self, planes, error_on_non_divisible_norm_groups=False):
        if planes < self.norm_groups:
            return self.norm_layer(planes, planes)
        elif not error_on_non_divisible_norm_groups and (planes % self.norm_groups) > 0:
            # This will just make a the number of norm groups equal to the number of planes
            print("Setting number of norm groups to {} for this convolution block.".format(planes))
            return self.norm_layer(planes, planes)
        else:
            return self.norm_layer(self.norm_groups, planes)


class MyronenkoResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8, kernel_size=3):
        super(MyronenkoResidualBlock, self).__init__()
        self.conv1 = MyronenkoConvolutionBlock(in_planes=in_planes, planes=planes, stride=stride,
                                               norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        self.conv2 = MyronenkoConvolutionBlock(in_planes=planes, planes=planes, stride=stride, norm_layer=norm_layer,
                                               norm_groups=norm_groups, kernel_size=kernel_size)
        if in_planes != planes:
            self.sample = conv1x1x1(in_planes, planes)
        else:
            self.sample = None

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x


class MyronenkoLayer(nn.Module):
    def __init__(self, n_blocks, block, in_planes, planes, *args, dropout=None, kernel_size=3, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(block(in_planes, planes, *args, kernel_size=kernel_size, **kwargs))
            in_planes = planes
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x


class MyronenkoEncoder(nn.Module):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_width = layer_widths[i]
            else:
                out_width = base_width * (feature_dilation ** i)
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x):
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            x = downsampling(x)
        x = self.layers[-1](x)
        return x
