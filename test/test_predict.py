import nibabel as nib
import numpy as np

from unittest import TestCase

from unet3d.utils.patches import compute_patch_indices, get_patch_from_3d_data, reconstruct_from_patches


class TestPrediction(TestCase):
    def setUp(self):
        image_shape = (120, 144, 90)
        data = np.arange(0, image_shape[0]*image_shape[1]*image_shape[2]).reshape(image_shape)
        affine = np.diag(np.ones(4))
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

    def test_reconstruct_with_overlapping_patches2(self):
        image_shape = (144, 144, 144)
        data = np.arange(0, image_shape[0]*image_shape[1]*image_shape[2]).reshape(image_shape)
        patch_overlap = 16
        patch_shape = (64, 64, 64)
        patch_indices = compute_patch_indices(data.shape, patch_shape, patch_overlap)
        patches = [get_patch_from_3d_data(data, patch_shape, index) for index in patch_indices]

        no_overlap_indices = compute_patch_indices(data.shape, patch_shape, 32)
        patch_indices = np.concatenate([patch_indices, no_overlap_indices])
        patches.extend([get_patch_from_3d_data(data, patch_shape, index) for index in no_overlap_indices])
        reconstruced_data = reconstruct_from_patches(patches, patch_indices, data.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(data == reconstruced_data))

    def test_reconstruct_with_multiple_channels(self):
        image_shape = (144, 144, 144)
        n_channels = 4
        data = np.arange(0, image_shape[0]*image_shape[1]*image_shape[2]*n_channels).reshape(
            [n_channels] + list(image_shape))
        patch_overlap = 16
        patch_shape = (64, 64, 64)
        patch_indices = compute_patch_indices(image_shape, patch_shape, patch_overlap)
        patches = [get_patch_from_3d_data(data, patch_shape, index) for index in patch_indices]
        self.assertEqual(patches[0].shape, tuple([4] + list(patch_shape)))

        reconstruced_data = reconstruct_from_patches(patches, patch_indices, data.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(data == reconstruced_data))

