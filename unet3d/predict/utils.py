import os


def pytorch_predict_batch_array(model, batch, n_gpus=1):
    import torch
    batch_x = torch.stack(batch)
    pred_x = pytorch_predict_batch(batch_x, model, n_gpus)
    return pred_x


def get_feature_filename_and_subject_id(dataset, idx, verbose=False):
    epoch_filenames = dataset.epoch_filenames[idx]
    x_filename = epoch_filenames[dataset.feature_index]
    if verbose:
        print("Reading:", x_filename)
    if len(epoch_filenames) > 2:
        # the subject id is specified in the list of filenames
        subject_id = epoch_filenames[-1]
    else:
        # infer the subject id from the filename
        if idx == 0:
            ref_filename = dataset.epoch_filenames[-1]
        else:
            ref_filename = dataset.epoch_filenames[0]
        subject_id = infer_subject_id(x_filename, ref_filename[dataset.feature_index])
    return x_filename, subject_id


def infer_subject_id(filename, ref_filename):
    if type(filename) == list:
        filename = filename[0]
    if type(ref_filename) == list:
        ref_filename = ref_filename[0]
    args = set(filename.split("/"))
    ref_args = set(ref_filename.split("/"))
    # return the parts that aren't in the reference filename
    return "_".join(args.difference(ref_args))


def pytorch_predict_batch(batch_x, model, n_gpus):
    if n_gpus > 0:
        batch_x = batch_x.cuda()
    model_dtype = list(model.parameters())[0].dtype
    if batch_x.dtype != model_dtype:
        batch_x = batch_x.to(model_dtype)
    if hasattr(model, "test"):
        pred_x = model.test(batch_x)
    else:
        pred_x = model(batch_x)
    return pred_x.cpu()
