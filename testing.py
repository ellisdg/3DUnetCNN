import os
import numpy as np
import nibabel as nib
import tables

from utils.utils import pickle_load
from config import config
from training import load_old_model


def get_prediction_labels(prediction, threshold=0.5):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction) > threshold] = 0
        label_arrays.append(label_data)
    return label_arrays


def get_test_indices():
    return pickle_load(config["testing_file"])


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def run_test_case(test_index, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = load_old_model(config["model_file"])

    data_file = tables.open_file(config["hdf5_file"], "r")
    data_index = get_test_indices()[test_index]
    affine = data_file.root.affine
    test_data = np.asarray([data_file.root.data[data_index]])
    for i, modality in enumerate(config["training_modalities"]):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(out_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth.to_filename(os.path.join(out_dir, "truth.nii.gz"))

    prediction = predict_and_get_image(model, test_data, affine)
    prediction.to_filename(os.path.join(out_dir, "prediction.nii.gz"))

    data_file.close()
