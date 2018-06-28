from unittest import TestCase

import numpy as np
import nibabel as nib

from unet3d.utils.utils import resample

from unet3d.augment import scale_affine


class TestAugmentation(TestCase):
    def test_scale_affine(self):
        shape = (4, 4, 4)
        affine = np.diag(np.ones(4))
        data = np.arange(np.prod(shape), dtype=np.float).reshape(shape)
        image = nib.Nifti1Image(data, affine)
        scale = (2, 2, 2)
        new_affine = scale_affine(affine, shape, scale)
        new_image = resample(image, target_affine=new_affine, target_shape=shape)
        new_data = new_image.get_data()
        self.assertEqual(np.sum(new_data[:1]), 0)
        self.assertEqual(np.sum(new_data[-1:]), 0)
        self.assertEqual(np.sum(new_data[:, :1]), 0)
        self.assertEqual(np.sum(new_data[:, -1:]), 0)
        self.assertEqual(np.sum(new_data[..., :1]), 0)
        self.assertEqual(np.sum(new_data[..., -1:]), 0)
