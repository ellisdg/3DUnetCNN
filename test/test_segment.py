from unittest import TestCase
import torch

from unet3d.utils.one_hot import convert_one_hot_to_label_map


class TestSegment(TestCase):
    def test_segment_left_right(self):
        data = torch.zeros((10, 10, 10), dtype=torch.int16)
        labels = [3, 17]
        data[4] = labels[0]
        data[5] = labels[1]
        grouped_data = data.clone()
        grouped_data[grouped_data > 0] = 1
        assert torch.all(grouped_data <= 1)
        left, right = split_left_right(grouped_data)
        left_right = torch.stack((left, right), dim=0)
        label_map = convert_one_hot_to_label_map(left_right, labels=labels)
        torch.testing.assert_close(label_map, data)


def split_left_right(data: torch.Tensor):
    center_index = data.shape[0] // 2
    left = torch.zeros_like(data)
    right = torch.zeros_like(data)
    left[:center_index] = data[:center_index]
    right[center_index:] = data[center_index:]
    return left, right
