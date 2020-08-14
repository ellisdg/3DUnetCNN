import os
import argparse

import nibabel as nib
from nilearn.image import resample_to_img

from brats.train import config, fetch_brats_2020_files
from unet3d.prediction import run_validation_cases
from unet3d.data import write_data_to_file
from unet3d.utils.utils import pickle_dump


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file")
    parser.add_argument("--data_file", default="./BraTS2020_Validation_data.h5")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--modalities", nargs="*")
    parser.add_argument("--validation_file", default="./BraTS2020_Validation_ids.pkl")
    parser.add_argument("--no_label_map", action="store_true", default=False)
    parser.add_argument("--prediction_dir", default="./BraTS2020_Validation_predictions")
    parser.add_argument("--output_basename", default="{subject}.nii.gz")
    return parser.parse_args()


def main():
    kwargs = vars(parse_args())
    prediction_dir = os.path.abspath(kwargs.pop("prediction_dir"))
    output_label_map = not kwargs.pop("no_label_map")
    for key, value in kwargs.items():
        if value:
            if key == "modalities":
                config["training_modalities"] = value
            else:
                config[key] = value
    filenames, subject_ids = fetch_brats_2020_files(config["training_modalities"], group="Validation",
                                                    include_truth=False, return_subject_ids=True)
    if not os.path.exists(config["data_file"]):
        write_data_to_file(filenames, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids, save_truth=False)
        pickle_dump(list(range(len(subject_ids))), config["validation_file"])

    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=output_label_map,
                         output_dir=prediction_dir,
                         test=False,
                         output_basename=kwargs["output_basename"],
                         permute=config["permute"])
    for filename_list, subject_id in zip(filenames, subject_ids):
        prediction_filename = os.path.join(prediction_dir, kwargs["output_basename"].format(subject=subject_id))
        print("Resampling:", prediction_filename)
        ref = nib.load(filename_list[0])
        pred = nib.load(prediction_filename)
        pred_resampled = resample_to_img(pred, ref, interpolation="nearest")
        pred_resampled.to_filename(prediction_filename)


if __name__ == "__main__":
    main()
