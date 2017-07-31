import numpy as np

from unittest import TestCase

from unet3d.generator import get_multi_class_labels


class TestDataGenerator(TestCase):

    def test_multi_class_labels(self):
        n_labels = 5
        labels = np.arange(1, n_labels+1)
        x_dim = 3
        label_map = np.asarray([[[np.arange(n_labels+1)] * x_dim]])
        binary_labels = get_multi_class_labels(label_map, n_labels, labels)

        for label in labels:
            self.assertTrue(np.all(binary_labels[:, label - 1][label_map[:, 0] == label] == 1))
