import os
from unittest import TestCase

import numpy as np

from unet3d.generator import get_multi_class_labels, get_training_and_validation_generators
from unet3d.data import add_data_to_storage, create_data_file


class TestDataGenerator(TestCase):

    def test_multi_class_labels(self):
        n_labels = 5
        labels = np.arange(1, n_labels+1)
        x_dim = 3
        label_map = np.asarray([[[np.arange(n_labels+1)] * x_dim]])
        binary_labels = get_multi_class_labels(label_map, n_labels, labels)

        for label in labels:
            self.assertTrue(np.all(binary_labels[:, label - 1][label_map[:, 0] == label] == 1))

    def test_get_training_and_validation_generators(self):
        data_file_path = "./temporary_data_test_file.h5"
        training_keys_file = "./temporary_training_keys_file.pkl"
        validation_keys_file = "./temporary_validation_keys_file.pkl"
        tmp_files = [data_file_path, training_keys_file, validation_keys_file]

        def rm_tmp_files():
            for tmp_file in tmp_files:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        rm_tmp_files()

        n_samples = 20
        validation_split = 0.8
        n_channels = 1
        len_x = 5
        len_y = 5
        len_z = 10
        batch_size = 3
        n_labels = 1
        image_shape = (len_x, len_y, len_z)
        data_size = n_samples * n_channels * len_x * len_y * len_z
        data = np.asarray(np.arange(data_size).reshape((n_samples, n_channels, len_x, len_y, len_z)), dtype=np.int16)
        self.assertEqual(data.shape[-3:], image_shape)
        truth = (data == 3).astype(np.int8)
        affine = np.diag(np.ones(4))
        affine[:, -1] = 1
        data_file, data_storage, truth_storage, affine_storage = create_data_file(data_file_path, n_channels, n_samples,
                                                                                  image_shape)

        for index in range(n_samples):
            add_data_to_storage(data_storage, truth_storage, affine_storage,
                                np.stack([data[index], truth[index]], axis=1)[0], affine=affine, n_channels=n_channels,
                                truth_dtype=np.int16)
            self.assertTrue(np.all(data_storage[index] == data[index]))
            self.assertTrue(np.all(truth_storage[index] == truth[index]))

        generators = get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file,
                                                            validation_keys_file, data_split=validation_split)
        training_generator, validation_generator, n_training_steps, n_validation_steps = generators

        # check that the training covers all the samples
        n_training_samples = 0
        training_samples = list()
        for i in range(n_training_steps):
            x, y = next(training_generator)
            hash_x = hash(str(x))
            self.assertNotIn(hash_x, training_samples)
            training_samples.append(hash_x)
            n_training_samples += x.shape[0]
        self.assertEqual(n_training_samples, n_samples * validation_split)

        # check that the validation covers all the samples
        n_validation_samples = 0
        validation_samples = list()
        for i in range(n_validation_steps):
            x, y = next(validation_generator)
            hash_x = hash(str(x))
            print(hash_x)
            self.assertNotIn(hash_x, validation_samples)
            validation_samples.append(hash_x)
            n_validation_samples += x.shape[0]
        self.assertEqual(n_training_samples, n_samples * validation_split)

        rm_tmp_files()
