import argparse
import os
import nibabel as nib
from unet3d.utils.utils import load_image
from unet3d.utils.one_hot import one_hot_image_to_label_map


def parse_args():
    return format_parser(argparse.ArgumentParser(), sub_command=False).parse_args()


def format_parser(parser, sub_command=False):
    if sub_command:
        parser.add_argument("--segment", action="store_true", default=False)
    else:
        parser.add_argument("--filenames", nargs="*", required=True)
        parser.add_argument("--labels", nargs="*", required=True)
        parser.add_argument("--hierarchy", default=False, action="store_true")
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--output_replace", nargs="*")
        parser.add_argument("--output_filenames", nargs="*")
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="If segmentation is set, this is the threshold for segmentation cutoff.")
    parser.add_argument("--sum", default=False, action="store_true",
                        help="Does not sum the predictions before using threshold.")
    parser.add_argument("--use_contours", action="store_true", default=False,
                        help="If the model was trained to predict contours you can use the contours to assist in the "
                             "segmentation. (This has not been shown to improve results.)")
    parser.add_argument("--no_overwrite", action="store_true", default=False,
                        help="Default is to overwrite.")
    return parser


def main():
    namespace = parse_args()
    overwrite = not namespace.no_overwrite
    if namespace.output_filenames:
        output_filenames = namespace.output_filenames
    elif namespace.output_replace:
        output_filenames = list()
        for fn in namespace.filenames:
            ofn = fn
            for i in range(0, len(namespace.output_replace), 2):
                ofn = ofn.replace(namespace.output_replace[i], namespace.output_replace[i + 1])
            output_filenames.append(ofn)
    else:
        raise RuntimeError("Please specify output_filenames or output_replace.")
    for fn, ofn in zip(namespace.filenames, output_filenames):
        if overwrite or not os.path.exists(ofn):
            if namespace.verbose:
                print(fn, "-->", ofn)
            if not os.path.exists(os.path.dirname(ofn)):
                os.makedirs(os.path.dirname(ofn))
            image = load_image(fn, force_4d=True)
            label_map = one_hot_image_to_label_map(image,
                                                   labels=namespace.labels,
                                                   threshold=namespace.threshold,
                                                   sum_then_threshold=namespace.sum,
                                                   label_hierarchy=namespace.hierarchy)
            label_map.to_filename(ofn)


if __name__ == "__main__":
    raise RuntimeError("segment.py is not setup to work with the latest version of the project.")
