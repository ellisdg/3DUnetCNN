from unittest import TestCase

from unet3d.model import unet_model_3d


class TestModel(TestCase):
    def test_batch_normalization(self):
        model = unet_model_3d(input_shape=(1, 16, 16, 16), depth=2, deconvolution=True, metrics=[], n_labels=1,
                              batch_normalization=True)

        layer_names = [layer.name for layer in model.layers]

        for name in layer_names[:-3]:  # exclude the last convolution layer
            if 'conv3d' in name and 'transpose' not in name:
                self.assertIn(name.replace('conv3d', 'batch_normalization'), layer_names)
