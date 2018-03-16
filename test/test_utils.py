from unittest import TestCase

import nibabel as nib
import numpy as np

from unet3d.utils.utils import resize
from unet3d.utils.sitk_utils import resample_to_spacing


class TestUtils(TestCase):
    def _resize_image_test(self, image, target_shape):
        original_image_shape = image.shape
        new_image = resize(image, target_shape)
        self.assertEqual(new_image.shape, target_shape)
        new_image = resize(new_image, original_image_shape, interpolation="linear")
        self.assertEqual(new_image.shape, original_image_shape)

    def _create_image(self, image_shape):
        data = np.asarray(np.arange(np.prod(image_shape)).reshape(image_shape), dtype=np.float)
        affine = np.zeros((4, 4))
        np.fill_diagonal(affine, 1)
        return nib.Nifti1Image(data, affine)

    def test_resize_image_1(self):
        image_shape = (4, 4, 4)
        image = self._create_image(image_shape)
        new_size = (2, 2, 2)
        self._resize_image_test(image, new_size)

    def test_resize_image_2(self):
        self._resize_image_test(self._create_image((12, 10, 8)), (8, 8, 8))

    def test_resize_image_2d(self):
        data = np.arange(1, 5).reshape((2, 2))
        new_data = resample_to_spacing(data, (2, 2), (1, 1), interpolation="nearest")
        self.assertTrue(np.all(new_data == np.asarray([[1, 1, 2, 2],
                                                       [1, 1, 2, 2],
                                                       [3, 3, 4, 4],
                                                       [3, 3, 4, 4]])))
        orig_data = resample_to_spacing(new_data, (1, 1), (2, 2), interpolation="linear")
        self.assertTrue(np.all(data == orig_data))

    def test_resize_image_3(self):
        self._resize_image_test(self._create_image((2, 5, 3)), (7, 5, 11))

    def test_resize_image_3d(self):
        data = np.arange(1, 9).reshape((2, 2, 2))
        new_data = resample_to_spacing(data, (2, 2, 2), (1, 1, 1), interpolation="nearest")
        self.assertTrue(np.all(new_data[0] == np.asarray([[1, 1, 2, 2],
                                                          [1, 1, 2, 2],
                                                          [3, 3, 4, 4],
                                                          [3, 3, 4, 4]])))
        orig_data = resample_to_spacing(new_data, (1, 1, 1), (2, 2, 2), interpolation="linear")
        self.assertTrue(np.all(data == orig_data))

    def test_images_align(self):
        data = np.arange(1, 9).reshape((2, 2, 2))
        affine = np.diag(np.ones(4) * 2)
        affine[3, 3] = 1
        image_nib = nib.Nifti1Image(data, affine=affine)
        new_image_nib = resize(image_nib, (4, 4, 4), interpolation="nearest")
        self.assertTrue(np.all(new_image_nib.get_data()[0] == np.asarray([[1, 1, 2, 2],
                                                                          [1, 1, 2, 2],
                                                                          [3, 3, 4, 4],
                                                                          [3, 3, 4, 4]])))
        self.assertTrue(np.all(new_image_nib.affine == np.asarray([[1., 0., 0., -0.5],
                                                                   [0., 1., 0., -0.5],
                                                                   [0., 0., 1., -0.5],
                                                                   [0., 0., 0., 1.]])))
