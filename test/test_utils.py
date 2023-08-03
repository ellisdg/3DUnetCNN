from unittest import TestCase

import nibabel as nib
import numpy as np

from unet3d.utils.affine import resize_affine, get_spacing_from_affine
from unet3d.utils.resample import resample, resample_image_to_spacing
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img, reorder_affine
from unet3d.utils.utils import (break_down_volume_into_half_size_volumes,
                                combine_half_size_volumes)
from unet3d.utils.one_hot import compile_one_hot_encoding


class TestUtils(TestCase):
    def _create_image(self, image_shape):
        data = np.asarray(np.arange(np.prod(image_shape)).reshape(image_shape), dtype=np.float)
        affine = np.zeros((4, 4))
        np.fill_diagonal(affine, 1)
        return nib.Nifti1Image(data, affine)

    def test_affine_crop(self):
        shape = (9, 9, 9)
        data = np.zeros(shape)
        data[3:6, 3:6, 3:6] = 1
        affine = np.diag(np.ones(len(shape) + 1))
        image = nib.Nifti1Image(data, affine)
        cropped_affine, cropped_shape = crop_img(image, return_affine=True, pad=False)
        expected_affine = np.copy(affine)
        expected_affine[:3, 3] = 3
        self.assertTrue(np.all(cropped_affine == expected_affine))

    def test_adjust_affine_spacing(self):
        old_shape = (128, 128, 128)
        new_shape = (64, 64, 64)
        old_affine = np.diag(np.ones(4))
        new_affine = resize_affine(old_affine, old_shape, new_shape)
        expected_affine = np.diag(np.ones(4) * 2)
        expected_affine[3, 3] = 1
        expected_affine[:3, 3] = 0.5
        np.testing.assert_array_equal(new_affine, expected_affine)

    def test_edge_resample(self):
        shape = (9, 9, 9)
        target_shape = shape
        data = np.ones(shape)
        data[-3:, -3:, -3:] = 2
        affine = np.diag(np.ones(4))
        image = nib.Nifti1Image(data, affine)
        cropped_affine, cropped_shape = crop_img(image, return_affine=True, percentile=50, pad=False)
        np.testing.assert_array_equal(cropped_shape, (3, 3, 3))
        np.testing.assert_array_almost_equal(np.ones(3)*6, cropped_affine[:3, 3])
        resized_affine = resize_affine(cropped_affine, cropped_shape, target_shape=target_shape)
        final_image = resample(image, resized_affine, target_shape, pad=True)
        np.testing.assert_array_equal(final_image.shape, target_shape)
        self.assertGreater(final_image.get_data().min(), 1)
        np.testing.assert_array_almost_equal(final_image.header.get_zooms(), np.ones(3)/(np.ones(3)*3))

    def test_crop_4d(self):
        shape = (9, 9, 9, 4)
        data = np.zeros(shape)
        data[3:6, 3:6, 3:6] = 1
        affine = np.diag(np.ones(4))
        image = nib.Nifti1Image(data, affine)
        cropped_image = crop_img(image, pad=False)
        expected_affine = np.copy(affine)
        expected_affine[:3, 3] = 3
        np.testing.assert_array_equal(cropped_image.affine, expected_affine)
        self.assertTrue(np.all(cropped_image.get_data() == 1))
        cropped_affine, cropped_shape = crop_img(image, pad=False, return_affine=True)
        np.testing.assert_array_equal(cropped_affine, expected_affine)

    def test_reorder_affine(self):
        affine = np.diag([-1, -3, 2, 1])
        affine[:3, 3] = [4, 6, 2]
        shape = (4, 4, 4)
        data = np.ones(shape)
        image = nib.Nifti1Image(data, affine)
        cropped_image = crop_img(image, pad=False)
        np.testing.assert_array_equal(cropped_image.affine, affine)
        np.testing.assert_array_equal(cropped_image.get_data(), data)
        new_affine = reorder_affine(affine, shape)
        np.testing.assert_array_equal(np.diagonal(new_affine), np.abs(np.diagonal(new_affine)))
        new_image = resample(image, new_affine, shape)
        np.testing.assert_array_equal(new_image.get_fdata(), image.get_fdata())

    def test_adjust_image_spacing(self):
        affine = np.asarray([[-1, 2.6e-4, -2.6e-4, -220],
                             [-2.6e-4, 3.4e-8, 1, 98],
                             [-2.6e-4, -1, -3.4e-8, 149],
                             [0, 0, 0, 1]])
        spacing = get_spacing_from_affine(affine)
        np.testing.assert_almost_equal(spacing, [1, 1, 1], decimal=7)
        data = np.asarray([np.diag(np.ones(10))] * 10)
        image = nib.Nifti1Image(data, affine)
        new_spacing = [2, 2, 2]
        new_image = resample_image_to_spacing(image, new_spacing, interpolation='linear')
        np.testing.assert_almost_equal(new_image.get_data().diagonal().diagonal(),
                                       (image.get_data().diagonal().diagonal()/2)[:5],
                                       decimal=7)

    def test_compile_one_hot_encoding(self):
        # test single label
        data = np.zeros((100, 100, 100))
        data[10:3000] = 5
        data[4000:90000] = 22
        _target = np.zeros(data.shape)
        _target[data == 5] = 1
        _target = _target[None, None]
        np.testing.assert_array_equal(compile_one_hot_encoding(data[None, None], 1, labels=[5]),
                                      _target)
        # test multiple labels
        _target2 = np.zeros(data.shape)
        _target2[data == 22] = 1
        _target = np.concatenate([_target, _target2[None, None]], axis=1)
        np.testing.assert_array_equal(compile_one_hot_encoding(data[None, None], 2, labels=[5, 22]),
                                      _target)

        # test multiple labels as the same map
        _target2[data == 5] = 1
        np.testing.assert_array_equal(compile_one_hot_encoding(data[None, None], 1, labels=[[5, 22]]),
                                      _target2[None, None])

    def test_super_resolution(self):
        shape = np.asarray((4, 4, 4))
        image = self._create_image(shape)
        new_shape = shape * 2
        data = image.get_fdata()
        new_data = np.ones(new_shape) * -1
        new_data[::2, ::2, ::2] = data  # original
        new_data[1::2, ::2, ::2] = data  # x shifted
        new_data[1::2, 1::2, ::2] = data  # x and y shifted
        new_data[1::2, ::2, 1::2] = data  # x and z shifted
        new_data[1::2, 1::2, 1::2] = data  # x, y, and z shifted
        new_data[::2, 1::2, ::2] = data  # y shifted
        new_data[::2, 1::2, 1::2] = data  # y and z shifted
        new_data[::2, ::2, 1::2] = data  # z shifted
        assert np.all(new_data != -1)  # test that all the values have been assigned

        volumes = break_down_volume_into_half_size_volumes(data)
        np.testing.assert_array_equal(data, combine_half_size_volumes(volumes))
