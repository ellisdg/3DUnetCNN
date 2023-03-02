# TODO: delete this script and make the predict.py script simpler
from unet3d.utils.utils import load_json
from unet3d.models.build import build_or_load_model
from unet3d.utils.filenames import load_dataset_class
import argparse
from torch.utils.data import DataLoader
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename")
    parser.add_argument("--config_filename")
    parser.add_argument("--output_directory")
    parser.add_argument("--group")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(args.config_filename)
    model = build_or_load_model(config["model"].pop("name"),
                                args.model_filename, n_gpus=1, **config["model"])
    model.eval()

    dataset_class = load_dataset_class(config["dataset"])

    if "training" in config["dataset"]:
        config["dataset"].pop("training")

    if "validation" in config["dataset"]:
        validation_kwargs = config["dataset"].pop("validation")
    else:
        validation_kwargs = dict()

    validation_dataset = dataset_class(filenames=config['validation_filenames'],
                                       **validation_kwargs,
                                       **config["dataset"])
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=1,
                                   shuffle=False)
    for i, (img, trg) in enumerate(validation_loader):
        pred = model(img[None])
        print(pred.min(), pred.max(), pred.mean())
        torch.save(pred[0], os.path.join(args.output_directory, "{}.pt".format(i)))


if __name__ == "__main__":
    main()
