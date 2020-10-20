from unittest import TestCase

import numpy as np
import nibabel as nib

from unet3d.utils.resample import resample

from unet3d.utils.augment import scale_affine, generate_permutation_keys, permute_data


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

    def test_permutations(self):
        permutation_keys = generate_permutation_keys()
        assert len(permutation_keys) == 48
        permutations = list()
        for key in permutation_keys:
            data = permute_data(self.data[None], key)
            if any([np.array_equal(data, other) for other in permutations]):
                raise ValueError("Key {} generates a permuted data array that is not unique.".format(key))
            permutations.append(data)
