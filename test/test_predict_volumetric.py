import unittest
import tempfile
import shutil
import os
import torch
import numpy as np
import nibabel as nib
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImageD, Compose
from monai.data.meta_tensor import MetaTensor
from unet3d.predict.volumetric import volumetric_predictions


class SimpleDummyModel(torch.nn.Module):
    """Simple model for testing that preserves input shape"""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class TestVolumetricPredictions(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory and test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.filenames = []
        
        # Create 2 temporary nifti files
        for i in range(2):
            data = np.random.randn(10, 10, 10).astype(np.float32)
            img = nib.Nifti1Image(data, np.eye(4))
            filename = os.path.join(self.temp_dir, f"test_{i}.nii.gz")
            img.to_filename(filename)
            self.filenames.append({"image": filename})
        
        self.prediction_dir = os.path.join(self.temp_dir, "predictions")
        os.makedirs(self.prediction_dir, exist_ok=True)
        
        # Create a simple model
        self.model = SimpleDummyModel()
        self.model.eval()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_volumetric_predictions_with_metatensor(self):
        """Test that volumetric_predictions works with proper MetaTensor inputs"""
        # Create dataset with MONAI transforms
        transforms = Compose([
            LoadImageD(keys=("image",), image_only=True, ensure_channel_first=True)
        ])
        
        dataset = Dataset(data=self.filenames, transform=transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # This should work without errors
        output_files = volumetric_predictions(
            model=self.model,
            dataloader=loader,
            prediction_dir=self.prediction_dir,
            activation=None,
            resample=False
        )
        
        # Verify output files were created
        self.assertEqual(len(output_files), 2)
        for output_file in output_files:
            self.assertTrue(os.path.exists(output_file))
    
    def test_volumetric_predictions_missing_meta_attribute(self):
        """Test that volumetric_predictions raises helpful error when input lacks meta attribute"""
        # Create a dataset that returns plain tensors instead of MetaTensors
        class PlainTensorDataset(Dataset):
            def __init__(self, filenames):
                self.filenames = filenames
            
            def __len__(self):
                return len(self.filenames)
            
            def __getitem__(self, idx):
                # Return a plain tensor without metadata
                return {"image": torch.randn(1, 10, 10, 10)}
        
        dataset = PlainTensorDataset(self.filenames)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # This should raise a helpful TypeError
        with self.assertRaises(TypeError) as context:
            volumetric_predictions(
                model=self.model,
                dataloader=loader,
                prediction_dir=self.prediction_dir,
                activation=None,
                resample=False
            )
        
        # Verify the error message is helpful
        error_msg = str(context.exception)
        self.assertIn("does not have 'meta' attribute", error_msg)
        self.assertIn("MetaTensor", error_msg)
        self.assertIn("LoadImageD", error_msg)
    
    def test_volumetric_predictions_missing_filename_in_meta(self):
        """Test that volumetric_predictions raises helpful error when meta lacks filename_or_obj"""
        # Create a dataset that returns MetaTensor without filename_or_obj in meta
        class IncompleteMetaDataset(Dataset):
            def __init__(self, filenames):
                self.filenames = filenames
            
            def __len__(self):
                return len(self.filenames)
            
            def __getitem__(self, idx):
                data = torch.randn(1, 10, 10, 10)
                affine = torch.eye(4)
                # Create MetaTensor with meta but without filename_or_obj
                meta = {"some_key": "some_value"}
                mt = MetaTensor(data, affine=affine, meta=meta)
                return {"image": mt}
        
        dataset = IncompleteMetaDataset(self.filenames)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # This should raise a helpful KeyError
        with self.assertRaises(KeyError) as context:
            volumetric_predictions(
                model=self.model,
                dataloader=loader,
                prediction_dir=self.prediction_dir,
                activation=None,
                resample=False
            )
        
        # Verify the error message is helpful
        error_msg = str(context.exception)
        self.assertIn("filename_or_obj", error_msg)
        self.assertIn("missing", error_msg.lower())
    
    def test_volumetric_predictions_with_resample(self):
        """Test that volumetric_predictions works with resample=True"""
        # Create dataset with MONAI transforms
        transforms = Compose([
            LoadImageD(keys=("image",), image_only=True, ensure_channel_first=True)
        ])
        
        dataset = Dataset(data=self.filenames, transform=transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # This should work without errors
        output_files = volumetric_predictions(
            model=self.model,
            dataloader=loader,
            prediction_dir=self.prediction_dir,
            activation=None,
            resample=True,
            interpolation="trilinear"
        )
        
        # Verify output files were created
        self.assertEqual(len(output_files), 2)
        for output_file in output_files:
            self.assertTrue(os.path.exists(output_file))


if __name__ == '__main__':
    unittest.main()
