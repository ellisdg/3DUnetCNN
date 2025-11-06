from unittest import TestCase

import torch

from unet3d.utils.resample import resample

from unet3d.utils.augment import scale_affine, generate_permutation_keys, permute_data
from unet3d.utils.image import Image


class TestAugmentation(TestCase):
    def setUp(self):
        self.shape = (4, 4, 4)
        self.affine = torch.diag(torch.ones(4))
        self.data = torch.arange(
            self.shape[0] * self.shape[1] * self.shape[2], dtype=torch.float32
        ).reshape(self.shape)
        # channels-first tensor
        self.image = Image(x=self.data.unsqueeze(0), affine=self.affine)

    def test_scale_affine(self):
        scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        new_affine = scale_affine(self.affine, self.shape, scale)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image[0]
        self.assertEqual(new_data[:1].sum().item(), 0)
        self.assertEqual(new_data[-1:].sum().item(), 0)
        self.assertEqual(new_data[:, :1].sum().item(), 0)
        self.assertEqual(new_data[:, -1:].sum().item(), 0)
        self.assertEqual(new_data[..., :1].sum().item(), 0)
        self.assertEqual(new_data[..., -1:].sum().item(), 0)

        self.affine = self.affine.clone()
        self.affine[0, 0] *= -1
        self.image = Image(x=self.data.unsqueeze(0), affine=self.affine)
        new_affine = scale_affine(self.affine, self.shape, scale)
        new_image = resample(self.image, target_affine=new_affine, target_shape=self.shape)
        new_data = new_image[0]
        self.assertEqual(new_data[:1].sum().item(), 0)
        self.assertEqual(new_data[-1:].sum().item(), 0)
        self.assertEqual(new_data[:, :1].sum().item(), 0)
        self.assertEqual(new_data[:, -1:].sum().item(), 0)
        self.assertEqual(new_data[..., :1].sum().item(), 0)
        self.assertEqual(new_data[..., -1:].sum().item(), 0)

    def test_permutations(self):
        permutation_keys = generate_permutation_keys()
        self.assertEqual(48, len(permutation_keys))
        permutations = list()
        for key in permutation_keys:
            data = permute_data(self.image.clone(), key)[0]  # use channel 0 for uniqueness
            if any([torch.equal(data, other) for other in permutations]):
                raise ValueError("Key {} generates a permuted data tensor that is not unique.".format(key))
            permutations.append(data)
