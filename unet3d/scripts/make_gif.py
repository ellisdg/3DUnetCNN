from argparse import ArgumentParser
from unet3d.utils.utils import load_image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import numpy as np
import os

muted_palette = ([72, 120, 208],
                 [238, 133, 74],
                 [106, 204, 100],
                 [214, 95, 95],
                 [149, 108, 180],
                 [140, 97, 60],
                 [220, 126, 192],
                 [121, 121, 121],
                 [213, 187, 103],
                 [130, 198, 226])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--background", required=True,
                        help="Filename for the background image file.")
    parser.add_argument("--labelmap1", required=True,
                        help="Filename for the 'Expert' or ground truth label map.")
    parser.add_argument("--labelmap2", required=True,
                        help="Filename for the 'UNet' label map.")
    parser.add_argument("--output", required=True,
                        help="Output gif filename.")
    parser.add_argument("--labels", nargs=2, default=("Expert", "UNet"),
                        help="Annotation labels to place under the images.")
    parser.add_argument("--text_size", default=30, type=int,
                        help="Size of text for annotating the images.")
    parser.add_argument("--fps", default=6, type=int,
                        help="Frames per second (fps) of the output gif file.")
    parser.add_argument("--pad", default=5, type=int,
                        help="Number of frames above and below the labels to start and end the gif respectively.")
    parser.add_argument("--palette", default="muted", type=str,
                        help="Seaborn palette to use for the label colors. (default='muted'). Requires "
                             "Seaborn to be installed if palette other than 'muted' is to be used.")
    parser.add_argument("--axcodes", default="RAS", type=str,
                        help="Axcodes to use for re-ording the image orientations during loading.")
    parser.add_argument("--include_zero", default=False, action="store_true", type=bool,
                        help="Include zero as a label value.")

    return parser.parse_args()


def get_palette(name="muted"):
    if name == "muted":
        return muted_palette
    else:
        import seaborn
        return np.asarray(np.asarray(seaborn.color_palette(name)) * 255, np.uint8)


def to_255(array):
    _255 = np.asarray(((array - array.min()) / array.max()) * 255, np.uint8)
    return np.stack([_255] * 3, axis=3)


def annotate(img, text, size=30,
             font_file=os.path.join(os.path.abspath(__file__), "..", "..", "misc", "fonts", "OpenSans-Bold.ttf")):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_file, size)
    draw.text((img.size[0] / 2 - len(text) * size / 4, img.size[1] - size * 1.5), text, font=font)


def concat_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def _load_image(fn, axcodes):
    return np.asarray(load_image(fn, axcodes=axcodes).dataobj[0])


def main():
    namespace = parse_args()

    ignore_zero = not namespace.include_zero
    annot_axis = 1

    cp = get_palette(namespace.palette)

    bg = _load_image(namespace.background, axcodes=namespace.axcodes)
    bg255 = to_255(bg)

    lb1 = _load_image(namespace.labelmap1, axcodes=namespace.axcodes)
    lb2 = _load_image(namespace.labelmap2, axcodes=namespace.axcodes)

    bg1 = np.copy(bg255)
    bg2 = np.copy(bg255)

    idx_min = np.asarray(np.where(lb1 == 1)).min(axis=1)
    idx_max = np.asarray(np.where(lb1 == 1)).max(axis=1)

    labels = np.unique(lb1)
    if ignore_zero and 0 in labels:
        labels = labels[labels != 0]

    for i, label in enumerate(labels):
        bg1[lb1 == label] = cp[i]
        bg2[lb2 == label] = cp[i]

    concat_shape = list(bg1.shape)
    concat_shape[annot_axis] = namespace.text_size

    bg1 = np.concatenate([np.zeros(concat_shape, bg1.dtype), bg1], axis=annot_axis)
    bg2 = np.concatenate([np.zeros(concat_shape, bg2.dtype), bg2], axis=annot_axis)

    images = list()

    for idx in range(idx_min[2] - namespace.pad, idx_max[2] + 1 + namespace.pad):
        img1 = Image.fromarray(np.rot90(bg1[:, :, idx]))
        annotate(img1, "Expert", size=namespace.text_size)
        img2 = Image.fromarray(np.rot90(bg2[:, :, idx]))
        annotate(img2, "UNet", size=namespace.text_size)
        img = concat_images(img1, img2)
        images.append(img)

    images[0].save(namespace.output, save_all=True, append_images=images[1:], duration=1000 / namespace.fps, loop=0)


if __name__ == "__main__":
    main()
