import os

from train import config
from unet3d.prediction import run_validation_case


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_case(test_index=0, out_dir=prediction_dir, model_file=config["model_file"],
                        validation_keys_file=config["validation_file"],
                        training_modalities=config["training_modalities"], output_label_map=True,
                        labels=config["labels"], hdf5_file=config["data_file"])


if __name__ == "__main__":
    main()
