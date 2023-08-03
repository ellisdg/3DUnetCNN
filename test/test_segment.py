from unittest import TestCase
import numpy as np

from unet3d.utils.utils import split_left_right
from unet3d.utils.one_hot import convert_one_hot_to_label_map


class TestSegment(TestCase):
    def test_segment_left_right(self):
        data = np.zeros((10, 10, 10), dtype=np.int16)
        labels = [3, 17]
        data[4] = labels[0]
        data[5] = labels[1]
        grouped_data = np.copy(data)
        grouped_data[data > 0] = 1
        assert np.all(grouped_data <= 1)
        left_right = np.stack(split_left_right(grouped_data), axis=-1)
        label_map = convert_one_hot_to_label_map(left_right, labels=labels)
        np.testing.assert_equal(label_map, data)
