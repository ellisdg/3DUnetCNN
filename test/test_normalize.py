import numpy as np

from unittest import TestCase

from unet3d.normalize import normalize_data


class TestNormalize(TestCase):
    def test_normalize_data(self):
        shape = (3, 4, 4, 4)
        data = np.zeros(shape)
        data[0] = -1
        data[0, :2] = -3
        data[1:, :2] = 2
        self.assertEqual(data.mean(), 0)
        mean = data.mean(axis=(1, 2, 3))
        std = data.std(axis=(1, 2, 3))
        normalized = normalize_data(data, mean, std)
        self.assertTrue(np.all(np.abs(normalized[1:]) == 1))
        np.testing.assert_array_equal(normalized, normalize_data(data))
        np.testing.assert_array_equal(normalized, normalize_data([data])[0])
