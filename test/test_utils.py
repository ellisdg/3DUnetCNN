from unittest import TestCase

import nibabel as nib
import numpy as np

from unet3d.utils.utils import resize


class TestUtils(TestCase):
    def _resize_image_test(self, image, target_shape):
        original_image_shape = image.shape
        new_image = resize(image, target_shape)
        self.assertEqual(new_image.shape, target_shape)
        new_image = resize(new_image, original_image_shape)
        self.assertEqual(new_image.shape, original_image_shape)
        new_data = np.asarray(np.round(new_image.get_data()), np.int)
        self.assertEqual(np.count_nonzero(new_data == 0), 1)

    def _create_image(self, image_shape):
        data = np.arange(np.prod(image_shape)).reshape(image_shape)
        affine = np.zeros((4, 4))
        np.fill_diagonal(affine, 1)
        return nib.Nifti1Image(data, affine)

    def test_resize_image1(self):
        image_shape = (4, 4, 4)
        image = self._create_image(image_shape)
        new_size = (2, 2, 2)
        self._resize_image_test(image, new_size)

    def test_resize_image2(self):
        self._resize_image_test(self._create_image((10, 10, 8)), (8, 8, 8))
