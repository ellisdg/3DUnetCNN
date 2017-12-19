from unittest import TestCase

import numpy as np
import keras.backend as K


from unet3d.metrics import weighted_dice_coefficient


class TestWeightedDice(TestCase):
    def test_weighted_dice_coefficient(self):
        data = np.zeros((5**3) * 3).reshape(3, 5, 5, 5)
        data[0, 0:1] = 1
        data[1, 0:2] = 1
        data[2, 1:4] = 1

        max_dice = K.eval(weighted_dice_coefficient(K.variable(data), K.variable(data)))
        for index in range(data.shape[0]):
            temp_data = np.copy(data)
            temp_data[index] = 0
            dice = K.eval(weighted_dice_coefficient(K.variable(data), K.variable(temp_data)))
            self.assertAlmostEqual(dice, (2 * max_dice)/3, delta=0.00001)

    def test_blank_dice_coefficient(self):
        data = np.zeros((5**3) * 3).reshape(3, 5, 5, 5)
        blank = np.copy(data)
        data[0, 0:1] = 1
        data[1, 0:2] = 1
        data[2, 1:4] = 1

        self.assertAlmostEqual(K.eval(weighted_dice_coefficient(K.variable(data), K.variable(blank))), 0, delta=0.00001)

    def test_empty_label(self):
        data = np.zeros((5**3) * 3).reshape(3, 5, 5, 5)
        data[1, 0:2] = 1
        data[2, 1:4] = 1

        self.assertEqual(K.eval(weighted_dice_coefficient(K.variable(data), K.variable(data))), 1)
