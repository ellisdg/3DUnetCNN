import nibabel as nib
import numpy as np

from unittest import TestCase

from unet3d.utils.patches import compute_patch_indices, get_patch_from_3d_data, reconstruct_from_patches


class TestPrediction(TestCase):
    def setUp(self):
        image_shape = (120, 144, 90)
        data = np.arange(0, image_shape[0]*image_shape[1]*image_shape[2]).reshape(image_shape)
        spacing = (1, 0.5, 1.5)
        affine = np.diag(np.ones(4))
        affine[:3, -1] = spacing
        self.image = nib.Nifti1Image(data, affine)

    def test_reconstruct_from_patches(self):
        patch_shape = (32, 32, 32)
        patch_overlap = 0
        patch_indices = compute_patch_indices(self.image.shape, patch_shape, patch_overlap)
        patches = [get_patch_from_3d_data(self.image.get_data(), patch_shape, index) for index in patch_indices]
        reconstruced_data = reconstruct_from_patches(patches, patch_indices, self.image.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(self.image.get_data() == reconstruced_data))

    def test_reconstruct_with_overlapping_patches(self):
        patch_overlap = 0
        patch_shape = (32, 32, 32)
        patch_indices = compute_patch_indices(self.image.shape, patch_shape, patch_overlap)
        patches = [get_patch_from_3d_data(self.image.get_data(), patch_shape, index) for index in patch_indices]
        # extend patches with modified patches that are 2 lower than the original patches
        patches.extend([patch - 2 for patch in patches])
        patch_indices = np.concatenate([patch_indices, patch_indices], axis=0)
        reconstruced_data = reconstruct_from_patches(patches, patch_indices, self.image.shape)
        # The reconstructed data should be 1 lower than the original data as 2 was subtracted from half the patches.
        # The resulting reconstruction should be the average.
        # noinspection PyTypeChecker
        self.assertTrue(np.all((self.image.get_data() - 1) == reconstruced_data))
