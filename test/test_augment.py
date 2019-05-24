from unittest import TestCase

import numpy as np
import nibabel as nib

from unet3d.utils.utils import resample

from unet3d.augment import scale_affine, rotate_affine


class TestAugmentation(TestCase):
    def setUp(self):
        self.shape = (4, 4, 4)
        self.affine = np.diag(np.ones(4))
        self.data = np.arange(np.prod(self.shape), dtype=np.float).reshape(self.shape)
        self.image = nib.Nifti1Image(self.data, self.affine)

    def test_scale_affine(self):
        scale = (0.5, 0.5, 0.5)
        new_affine = scale_affine(self.affine, self.shape, scale)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image.get_data()
        self.assertEqual(np.sum(new_data[:1]), 0)
        self.assertEqual(np.sum(new_data[-1:]), 0)
        self.assertEqual(np.sum(new_data[:, :1]), 0)
        self.assertEqual(np.sum(new_data[:, -1:]), 0)
        self.assertEqual(np.sum(new_data[..., :1]), 0)
        self.assertEqual(np.sum(new_data[..., -1:]), 0)

        self.affine[0, 0] *= -1
        self.image = nib.Nifti1Image(self.data, self.affine)
        new_affine = scale_affine(self.affine, self.shape, scale)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image.get_data()
        print(new_data)
        self.assertEqual(np.sum(new_data[:1]), 0)
        self.assertEqual(np.sum(new_data[-1:]), 0)
        self.assertEqual(np.sum(new_data[:, :1]), 0)
        self.assertEqual(np.sum(new_data[:, -1:]), 0)
        self.assertEqual(np.sum(new_data[..., :1]), 0)
        self.assertEqual(np.sum(new_data[..., -1:]), 0)

    def _rotate_affine(self):
        rotation = np.ones(3) * np.pi * 2
        new_affine = rotate_affine(self.affine, rotation)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image.get_data()
        np.testing.assert_almost_equal(self.data, new_data)

        rotation = (np.pi/2, 0, 0)
        new_affine = rotate_affine(self.affine, rotation)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image.get_data()
        print(self.data)
        print(np.rot90(self.data))
        np.testing.assert_almost_equal(np.rot90(self.data), new_data)
