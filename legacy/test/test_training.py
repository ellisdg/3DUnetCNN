from unittest import TestCase

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from unet3d.training import get_callbacks


class TestCallbakcs(TestCase):
    def test_reduce_on_plateau(self):
        _, _, scheduler = get_callbacks(model_file='model.h5', learning_rate_patience=50, learning_rate_drop=0.5)
        self.assertIsInstance(scheduler, ReduceLROnPlateau)

    def test_early_stopping(self):
        _, _, _, stopper = get_callbacks(model_file='model.h5', early_stopping_patience=100)
        self.assertIsInstance(stopper, EarlyStopping)

