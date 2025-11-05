import unittest
import torch

from unet3d.utils.augment import translate_image, find_image_center, scale_image, smooth_img


class TestAugmentValidation(unittest.TestCase):
    """Test that augmentation functions provide helpful error messages when inputs lack affine attribute"""
    
    def test_translate_image_with_plain_tensor(self):
        """Test translate_image raises helpful error with plain Tensor"""
        plain_tensor = torch.randn(1, 10, 10, 10)
        translation_scales = torch.tensor([0.1, 0.1, 0.1])
        
        with self.assertRaises(TypeError) as context:
            translate_image(plain_tensor, translation_scales)
        
        error_msg = str(context.exception)
        self.assertIn("does not have 'affine' attribute", error_msg)
        self.assertIn("MetaTensor", error_msg)
    
    def test_find_image_center_with_plain_tensor(self):
        """Test find_image_center raises helpful error with plain Tensor"""
        plain_tensor = torch.randn(1, 10, 10, 10)
        
        with self.assertRaises(TypeError) as context:
            find_image_center(plain_tensor)
        
        error_msg = str(context.exception)
        self.assertIn("does not have 'affine' attribute", error_msg)
        self.assertIn("MetaTensor", error_msg)
    
    def test_scale_image_with_plain_tensor(self):
        """Test scale_image raises helpful error with plain Tensor"""
        plain_tensor = torch.randn(1, 10, 10, 10)
        scale = torch.tensor([1.2, 1.2, 1.2])
        
        with self.assertRaises(TypeError) as context:
            scale_image(plain_tensor, scale)
        
        error_msg = str(context.exception)
        self.assertIn("does not have 'affine' attribute", error_msg)
        self.assertIn("MetaTensor", error_msg)
    
    def test_smooth_img_with_plain_tensor(self):
        """Test smooth_img raises helpful error with plain Tensor"""
        plain_tensor = torch.randn(1, 10, 10, 10)
        fwhm = torch.tensor([2.0, 2.0, 2.0])
        
        with self.assertRaises(TypeError) as context:
            smooth_img(plain_tensor, fwhm)
        
        error_msg = str(context.exception)
        self.assertIn("does not have 'affine' attribute", error_msg)
        self.assertIn("MetaTensor", error_msg)


if __name__ == '__main__':
    unittest.main()
