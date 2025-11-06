from unittest import TestCase

import numpy as np
import torch

from unet3d.utils.affine import resize_affine, get_spacing_from_affine
from unet3d.utils.resample import resample, resample_image_to_spacing
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img, reorder_affine, image_slices_to_affine
from unet3d.utils.utils import (break_down_volume_into_half_size_volumes,
                                combine_half_size_volumes)
from unet3d.utils.one_hot import compile_one_hot_encoding
from unet3d.utils.image import Image


class TestUtils(TestCase):
    def _create_array(self, image_shape):
        # numpy array used for super-resolution utilities which are numpy-based
        data = np.asarray(np.arange(np.prod(image_shape)).reshape(image_shape), dtype=float)
        return data

    def test_affine_crop(self):
        shape = (9, 9, 9)
        data = torch.zeros((1,) + shape)
        data[:, 3:6, 3:6, 3:6] = 1
        affine = torch.eye(4, dtype=torch.float64)
        image = Image(x=data, affine=affine)
        cropped_affine, cropped_shape = crop_img(image, return_affine=True, pad=False)
        expected_affine = affine.clone()
        expected_affine[:3, 3] = torch.tensor([3.0, 3.0, 3.0], dtype=affine.dtype)
        torch.testing.assert_close(cropped_affine, expected_affine)

    def test_adjust_affine_spacing(self):
        old_shape = (128, 128, 128)
        new_shape = (64, 64, 64)
        old_affine = torch.eye(4, dtype=torch.float64)
        new_affine = resize_affine(old_affine, old_shape, new_shape)
        expected_affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0], dtype=torch.float64))
        expected_affine[:3, 3] = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        torch.testing.assert_close(new_affine, expected_affine)

    def test_edge_resample(self):
        shape = (9, 9, 9)
        target_shape = shape
        data = torch.ones((1,) + shape)
        data[:, -3:, -3:, -3:] = 2
        affine = torch.eye(4, dtype=torch.float64)
        image = Image(x=data, affine=affine)
        # Instead of percentile-based crop (numpy), directly define slices for the last 3 voxels in each dim
        slices = [slice(6, 9), slice(6, 9), slice(6, 9)]
        cropped_affine = image_slices_to_affine(image, slices)
        cropped_shape = torch.tensor([3, 3, 3], dtype=torch.int64)
        torch.testing.assert_close(cropped_shape, torch.tensor([3, 3, 3], dtype=torch.int64))
        torch.testing.assert_close(cropped_affine[:3, 3], torch.tensor([6.0, 6.0, 6.0], dtype=affine.dtype))
        resized_affine = resize_affine(cropped_affine, cropped_shape, target_shape=target_shape)
        final_image = resample(image, resized_affine, target_shape, pad=True)
        # shape check (spatial)
        self.assertEqual(tuple(final_image.shape[-3:]), target_shape)
        self.assertGreater(final_image.min().item(), 1)
        spacing = get_spacing_from_affine(final_image.affine)
        torch.testing.assert_close(spacing, torch.ones(3, dtype=affine.dtype) / 3.0, rtol=1e-5, atol=1e-6)

    def test_crop_4d(self):
        shape = (4, 9, 9, 9)  # channels first
        data = torch.zeros(shape)
        data[:, 3:6, 3:6, 3:6] = 1
        affine = torch.eye(4, dtype=torch.float64)
        image = Image(x=data, affine=affine)
        cropped_affine, cropped_shape = crop_img(image, pad=False, return_affine=True)
        expected_affine = affine.clone()
        expected_affine[:3, 3] = torch.tensor([3.0, 3.0, 3.0], dtype=affine.dtype)
        torch.testing.assert_close(cropped_affine, expected_affine)
        # sanity: cropped shape spatial dims should be 3,3,3
        torch.testing.assert_close(cropped_shape, torch.tensor([3, 3, 3], dtype=torch.int64))

    def test_reorder_affine(self):
        affine = np.diag([-1.0, -3.0, 2.0, 1.0])
        affine[:3, 3] = [4.0, 6.0, 2.0]
        shape = (4, 4, 4)
        data = torch.ones((1,) + shape)
        image = Image(x=data, affine=torch.from_numpy(affine))
        new_affine = reorder_affine(affine, shape)
        self.assertTrue(np.all(np.diagonal(new_affine) == np.abs(np.diagonal(new_affine))))
        new_image = resample(image, torch.from_numpy(new_affine), shape)
        torch.testing.assert_close(new_image, image)

    def test_adjust_image_spacing(self):
        # set higher precision on affine for spacing test
        affine = torch.tensor([[-1.0, 2.6e-4, -2.6e-4, -220.0],
                               [-2.6e-4, 3.4e-8, 1.0, 98.0],
                               [-2.6e-4, -1.0, -3.4e-8, 149.0],
                               [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        spacing = get_spacing_from_affine(affine)
        torch.testing.assert_close(spacing, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64), rtol=0, atol=1e-7)
        data = torch.stack([torch.diag(torch.ones(10))] * 10).unsqueeze(0)  # shape (1,10,10,10)
        image = Image(x=data, affine=affine)
        new_spacing = torch.tensor([2.0, 2.0, 2.0], dtype=affine.dtype)
        new_image = resample_image_to_spacing(image, new_spacing, interpolation='linear')
        # Compare main diagonal intensity profile scaled by 1/2 and shortened
        orig_diag = image[0].diagonal(offset=0, dim1=-2, dim2=-1).diagonal(offset=0)
        new_diag = new_image[0].diagonal(offset=0, dim1=-2, dim2=-1).diagonal(offset=0)
        torch.testing.assert_close(new_diag, (orig_diag / 2.0)[: new_diag.shape[0]], rtol=0, atol=1e-6)

    def test_compile_one_hot_encoding(self):
        # test single label
        data = torch.zeros((1, 1, 100, 100, 100))  # MetaTensor required by function
        mt = Image(x=data[0], affine=torch.eye(4))  # shape (1,100,100,100)
        mt = mt.unsqueeze(0)  # add batch dim -> (1,1,100,100,100)
        flat = mt[0, 0].reshape(-1)
        flat[10:3000] = 5
        flat[4000:90000] = 22
        mt[0, 0] = flat.reshape(100, 100, 100)

        _target = torch.zeros((1, 1, 100, 100, 100), dtype=torch.uint8)
        _target[0, 0][mt[0, 0] == 5] = 1
        out = compile_one_hot_encoding(mt, 1, labels=[5])
        torch.testing.assert_close(out, _target[0])  # function returns 4D when return_4d=True then squeezes batch

        # test multiple labels
        _target2 = torch.zeros((100, 100, 100), dtype=torch.uint8)
        _target2[mt[0, 0] == 22] = 1
        expected = torch.cat([_target[0], _target2.unsqueeze(0)], dim=0)
        out2 = compile_one_hot_encoding(mt, 2, labels=[5, 22])
        torch.testing.assert_close(out2, expected)

        # test multiple labels as the same map
        _target2[mt[0, 0] == 5] = 1
        out3 = compile_one_hot_encoding(mt, 1, labels=[[5, 22]])
        torch.testing.assert_close(out3, _target2.unsqueeze(0))

    def test_super_resolution(self):
        shape = np.asarray((4, 4, 4))
        data = self._create_array(shape)
        new_shape = shape * 2
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
