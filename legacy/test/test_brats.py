from unittest import TestCase
import os
import shutil
import sys
import nibabel as nib
import numpy as np


def create_blank_image(filename, image_shape=(140, 140, 140)):
    data = np.zeros(image_shape, dtype=np.int16)
    affine = np.diag(np.ones(4))
    image = nib.Nifti1Image(dataobj=data, affine=affine)
    image.to_filename(filename)


class TestPreprocess(TestCase):
    def setUp(self):
        self.temp_brats_dir = os.path.abspath("temp_brats")
        os.makedirs(self.temp_brats_dir)
        sys.path.append('../brats')
        self.preprocessed_dir = os.path.abspath("temp_preprocessed_brats")

    def tearDown(self):
        for directory in (self.temp_brats_dir, self.preprocessed_dir):
            if os.path.exists(directory):
                shutil.rmtree(directory)

    def create_replica_dataset(self, subject_ids, scan_types, directory_name):
        gbm_dir = os.path.join(self.temp_brats_dir, directory_name)
        for subject_id in subject_ids:
            subject_dir = os.path.join(gbm_dir, subject_id)
            os.makedirs(subject_dir)
            for scan_label in scan_types:
                basename = '{}_{}.nii.gz'.format(subject_id, scan_label)
                scan_filename = os.path.join(subject_dir, basename)
                create_blank_image(scan_filename)

    def create_replica_dataset_pre2018(self):
        self.create_replica_dataset(subject_ids=('TCGA-00-000',),
                                    scan_types=('flair',
                                                'GlistrBoost',
                                                'GlistrBoost_ManuallyCorrected',
                                                't1',
                                                't1Gd',
                                                't2'),
                                    directory_name='Pre-operative_TCGA_GBM_NIfTI_and_Segmentations')
        self.create_replica_dataset(subject_ids=('TCGA-01-000',),
                                    scan_types=('flair',
                                                'GlistrBoost',
                                                't1',
                                                't1Gd',
                                                't2'),
                                    directory_name='Pre-operative_TCGA_GBM_NIfTI_and_Segmentations')

    def create_replica_dataset_2018(self):
        self.create_replica_dataset(subject_ids=('Brats18_1900_1_1',),
                                    scan_types=('flair',
                                                't1',
                                                't1ce',
                                                't2',
                                                'seg'),
                                    directory_name='HGG')

    def test_preprocess_pre2018(self):
        from preprocess import convert_brats_data
        self.create_replica_dataset_pre2018()
        convert_brats_data(self.temp_brats_dir, self.preprocessed_dir)

    def test_preprocess_2018(self):
        self.create_replica_dataset_2018()
        from preprocess import convert_brats_data
        convert_brats_data(self.temp_brats_dir, self.preprocessed_dir)

