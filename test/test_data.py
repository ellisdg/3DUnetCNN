from unittest import TestCase
import os

from unet3d.data import DataFile
from unet3d.utils.utils import resize_affine
import numpy as np
import nibabel as nib


class TestDataFile(TestCase):
    def setUp(self):
        self.filename = os.path.abspath('test.h5')
        self.data_file = DataFile(self.filename)

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.filename))

    def test_add_data(self):
        features = np.zeros((9, 9, 9))
        targets = np.ones(features.shape)
        affine = np.diag(np.ones(4))
        subject_id = 'mydata'
        self.data_file.add_data(features, targets, subject_id)
        x, y = self.data_file.get_data(subject_id)
        np.testing.assert_array_equal(features, x)
        np.testing.assert_array_equal(targets, y)

        subject_id = 'yourdata'
        features = features.copy()
        features[3:6, 3:6, 3:6] = 5
        targets = np.zeros(features.shape)
        targets[3:6, 3:6, 3:6] = 1
        self.data_file.add_data(features, targets, subject_id, affine=affine)
        x_image, y_image = self.data_file.get_nibabel_images(subject_id)
        np.testing.assert_array_equal(x_image.get_data(), features)

        roi_affine = affine.copy()
        roi_affine[:3, 3] = 3
        roi_shape = (3, 3, 3)
        self.data_file.add_supplemental_data(subject_id, roi_affine=roi_affine, roi_shape=roi_shape)

        _affine, _shape = self.data_file.get_roi(subject_id)
        np.testing.assert_array_equal(_affine, roi_affine)
        np.testing.assert_array_equal(_shape, roi_shape)

        roi_features, roi_targets = self.data_file.get_roi_data(subject_id)
        self.assertEqual(roi_features.min(), 5)
        self.assertEqual(roi_features.max(), 5)
        np.testing.assert_array_equal(roi_features.shape, roi_shape)
        np.testing.assert_array_equal(roi_targets.shape, roi_features.shape)
        self.assertEqual(roi_targets.min(), 1)
        self.assertEqual(roi_targets.max(), 1)

        target_shape = (4, 4, 4)
        _affine = resize_affine(roi_affine, shape=roi_shape, target_shape=target_shape)
        roi_features, roi_targets = self.data_file.get_roi_data(subject_id, roi_affine=_affine, roi_shape=target_shape)
        self.assertEqual(roi_targets.min(), 1)
        self.assertEqual(roi_targets.max(), 1)
        np.testing.assert_array_equal(roi_features.shape, target_shape)

    def test_set_parameters(self):
        training_subject_ids = ["train1", "train2", "train3"]
        self.data_file.set_training_groups(training_subject_ids)
        np.testing.assert_array_equal(self.data_file.get_training_groups(), training_subject_ids)
        shape = (3, 3, 3)
        for index, subject_id in enumerate(training_subject_ids):
            x = np.ones(shape) * index * 2
            y = x + 1
            self.data_file.add_data(x, y, subject_id)

        for index, subject_id in enumerate(self.data_file.get_training_groups()):
            x, y = self.data_file.get_data(subject_id)
            np.testing.assert_array_equal(x, np.ones(shape) * index * 2)
            np.testing.assert_array_equal(y, np.ones(shape) * index * 2 + 1)

        validation_subject_ids = ["val1", "val2"]
        self.data_file.set_validation_groups(validation_subject_ids)
        np.testing.assert_array_equal(validation_subject_ids, self.data_file.get_validation_groups())

    def test_multi_channel_data(self):
        shape = (4, 5, 6)
        affine = np.diag(np.ones(4) * 0.5)
        affine[3, 3] = 1

        subject_id = 'subject1'
        data_list = list()
        for i in range(3):
            data = np.ones(shape) * i
            data_list.append(data)
        self.data_file.add_data(data_list, data_list, subject_id, affine=affine, roi_shape=(2, 3, 4), roi_affine=affine)
        roi_features, roi_targets = self.data_file.get_roi_data(subject_id)
        np.testing.assert_array_equal(roi_features.shape, (3, 2, 3, 4))
        for i in range(3):
            self.assertTrue(np.all(roi_features[i] == i))

        subject_id = 'subject2'
        self.data_file.add_nibabel_images([nib.Nifti1Image(data, affine) for data in data_list], nib.Nifti1Image(data, affine),
                                          subject_id, roi_shape=(2, 3, 4), roi_affine=affine)
        roi_features, roi_targets = self.data_file.get_roi_data(subject_id)
        np.testing.assert_array_equal(roi_features.shape, (3, 2, 3, 4))
        for i in range(3):
            self.assertTrue(np.all(roi_features[i] == i))

        subject_id = 'subject3'
        self.data_file.add_nibabel_images([nib.Nifti1Image(data, affine) for data in data_list],
                                          [nib.Nifti1Image(data, affine) for data in data_list],
                                          subject_id, roi_shape=(2, 3, 4), roi_affine=affine)
        roi_features, roi_targets = self.data_file.get_roi_data(subject_id)
        np.testing.assert_array_equal(roi_targets.shape, (3, 2, 3, 4))
        for i in range(3):
            self.assertTrue(np.all(roi_targets[i] == i))

        subject_id = 'subject4'
        targets_affine = affine.copy()
        targets_affine[1, 2] = 1
        with self.assertRaises(AssertionError):
            self.data_file.add_nibabel_images([nib.Nifti1Image(data, affine) for data in data_list],
                                              [nib.Nifti1Image(data, targets_affine) for data in data_list],
                                              subject_id, roi_shape=(2, 3, 4), roi_affine=affine)

        subject_id = 'subject5'
        images = list()
        for index, data in enumerate(data_list):
            _affine = affine.copy()
            _affine[index, index] = 2
            images.append(nib.Nifti1Image(data, _affine))

        with self.assertRaises(AssertionError):
            self.data_file.add_nibabel_images(images[0], images, subject_id)

        with self.assertRaises(AssertionError):
            self.data_file.add_nibabel_images(images[0], images, subject_id)

    def test_add_images(self):
        shape = (4, 5, 6)
        affine = np.diag(np.ones(4) * 0.5)
        affine[3, 3] = 1
        subject_id = "subject1"
        data = np.ones(shape)
        image = nib.Nifti1Image(data, affine)
        targets = data.copy()
        targets[-3:] = 0
        targets_image = nib.Nifti1Image(targets, affine)
        self.data_file.add_nibabel_images(image, targets_image, subject_id)
        _features, _targets = self.data_file.get_data(subject_id)
        np.testing.assert_array_equal(_features, data)
        np.testing.assert_array_equal(_targets, targets)

    def tearDown(self):
        self.data_file.close()
        os.remove(self.filename)
