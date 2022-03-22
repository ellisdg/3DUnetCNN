import numpy as np

from unet3d.predict import pytorch_predict_batch


def pytorch_predict_batch_array(model, batch, n_gpus=1):
    import torch
    batch_x = torch.tensor(np.moveaxis(np.asarray(batch), -1, 1)).float()
    pred_x = pytorch_predict_batch(batch_x, model, n_gpus)
    return np.moveaxis(pred_x.numpy(), 1, -1)


def get_feature_filename_and_subject_id(dataset, idx, verbose=False):
    epoch_filenames = dataset.epoch_filenames[idx]
    x_filename = epoch_filenames[dataset.feature_index]
    if verbose:
        print("Reading:", x_filename)
    subject_id = epoch_filenames[-1]
    return x_filename, subject_id