import os
import glob
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, resample_to_img
from fcnn.utils.utils import load_json, split_left_right, convert_one_hot_to_single_label_map_volume


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_filename")
    parser.add_argument("--prediction_dir")
    parser.add_argument("--output_dir")
    return parser.parse_args()


def main():
    namespace = parse_args()
    if not os.path.exists(namespace.output_dir):
        os.makedirs(namespace.output_dir)
    config = load_json(namespace.config_filename)
    labels1, labels2 = config["sequence_kwargs"]["labels"]
    for fn in glob.glob(os.path.join(namespace.prediction_dir, "*")):
        print(fn)
        bn = os.path.basename(fn)
        ofn = os.path.join(namespace.output_dir, bn)
        image = nib.load(fn)
        _image = reorder_img(image)
        data = _image.get_fdata()
        data1 = data[..., :len(labels1)]
        data2 = data[..., len(labels1):]
        for i, (l, d) in enumerate(((labels1, data1), (labels2, data2))):
            volumes = list()
            labels = list()
            for ii, label in enumerate(l):
                if type(label) == list and len(label) == 2:
                    volumes.extend(split_left_right(d[..., ii]))
                    labels.extend(label)
                else:
                    volumes.append(d[..., ii])
                    labels.append(label)
            fixed_data = resample_to_img(_image.__class__(dataobj=np.stack(volumes, axis=-1),
                                                          affine=_image.affine), target_img=image).get_fdata()
            label_map = convert_one_hot_to_single_label_map_volume(fixed_data, labels, dtype=np.uint8)
            out_image = image.__class__(dataobj=label_map, affine=image.affine)
            out_image.to_filename(ofn.replace(".", "_pred{}.".format(i + 1), 1))


if __name__ == "__main__":
    main()
