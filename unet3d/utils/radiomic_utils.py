import numpy as np
import os
import nibabel as nib

from .utils import update_progress, move_channels_first, move_channels_last, load_single_image
from .normalize import zero_mean as unet3d_normalize
from .resample import resample as unet3d_resample
from .augment import permute_data, random_permutation_key


def compute_affine_from_point(point, window, spacing):
    affine = np.diag(np.ones(4))
    np.fill_diagonal(affine, list(spacing) + [1])
    window_extent = np.multiply(window, spacing)
    offset = window_extent/2
    affine[:3, 3] = point - offset
    return affine
    

def fetch_data_for_point(point, image, window, flip=False, interpolation='linear', spacing=None,
                         normalization_func=unet3d_normalize):
    if spacing is None:
        spacing = np.asarray(image.header.get_zooms())
    affine = compute_affine_from_point(point, window, spacing)
    _image = unet3d_resample(image, affine, window, interpolation)
    image_data = _image.get_data()
    if len(image_data.shape) == 3:
        image_data = image_data[..., None]
    ch_first = move_channels_first(image_data)
    if flip:
        ch_first = permute_data(ch_first, random_permutation_key())
    if normalization_func is not None:
        normalized = normalization_func(ch_first)
    else:
        normalized = ch_first
    image_data[:] = move_channels_last(normalized)
    return image_data


def window_data(data, lower_percentile=None, upper_percentile=99):
    data = np.copy(data)
    if lower_percentile:
        lower = np.percentile(data, lower_percentile)
        data[data < lower] = lower
    upper = np.percentile(data, upper_percentile)
    data[data > upper] = upper
    return data


def normalize(in_file, out_file):
    image = nib.load(in_file)
    data = np.copy(image.get_data())
    mean = np.mean(data)
    std = np.std(data)
    data = np.subtract(data, mean)
    data = np.divide(data, std)
    image.__class__(data, image.affine).to_filename(out_file)


def get_points_from_surgery(surgery, name="labeled_points"):
    return surgery["intraop"]["{}.fcsv".format(name)]


def get_labeled_points(surgery, is_reviewed=True):
    points = get_points_from_surgery(surgery)
    if is_reviewed:
        points.get_metadata()["review_date"]
    return points


def append_negatives(surgery, point_dict):
    try:
        negative_points = get_points_from_surgery(surgery, "negative")
        for point_name, point in negative_points.get_points_dict().items():
            point_dict["NEGATIVE " + point_name] = point
    except KeyError:
        pass

    
def get_complete_point_dict(surgery, include_negative=False, is_reviewed=True):
    points = get_labeled_points(surgery, is_reviewed=is_reviewed)
    point_dict = points.get_points_dict()
    if include_negative:
        append_negatives(surgery, point_dict)
    return point_dict


def load_fs_lut():
    fs_lut_file = "/home/neuro-user/apps/freesurfer-6.0/FreeSurferColorLUT.txt"
    fs_lut = dict()
    with open(fs_lut_file, 'r') as opened_file:
        for row_number, line in enumerate(opened_file):
            line = line.strip()
            if not line or line[0] == "#":
                continue
            while "  " in line:
                line = line.replace("  ", " ")
            row = line.split(" ")
            fs_lut[row[1]] = int(row[0])
    return fs_lut

    
def label_coordinates(image, label=1):
    """Returns the coordinate locations that are equal to the provided label(default 1)."""
    coordinates = []
    for idx in zip(*np.where(image.get_data() == label)):
        coordinates.append(index_to_point(idx, image.affine))
    return coordinates


def get_label_indices(image):
    return np.stack(np.where(image.get_data() > 0), axis=-1)


def load_subject_image(subject_directory, basename, resample=None, reorder=True):
    image_filename = os.path.join(subject_directory, basename)
    return load_single_image(image_filename, resample=resample, reorder=reorder)


def index_to_point(index, affine):
    return np.dot(affine, list(index) + [1])[:3]


def compute_roc_xy(predictions, truth):
    x = list()
    y = list()
    for threshold in np.arange(0, 1.01, 0.001):
        p_pos = predictions > threshold
        p_neg = p_pos == False
        sensitivity = np.sum(p_pos[truth])/np.sum(truth)
        specificity = np.sum(p_neg[truth == False])/np.sum(truth == False)
        y.append(sensitivity)
        x.append(1 - specificity)
    return x, y


def binary_classification(label, target_labels):
    return int(label in target_labels)


def multilabel_classification(label, target_labels):
    array = np.zeros(len(target_labels))
    if label in target_labels:
        target_index = list(target_labels).index(label)
        array[target_index] = 1
    return array


def fetch_data(feature_filename, target_filename, target_labels, input_window, 
               flip=False, interpolation='linear', n_points=1, spacing=(1, 1, 1),
               reorder=True, resample='continuous', skip_blank=True,
               classify=binary_classification):
    image = load_single_image(feature_filename, reorder=reorder, resample=resample)
    target_image = load_single_image(target_filename, reorder=reorder, resample=resample)
    indices = get_label_indices(target_image)
    _candidates = np.copy(indices)
    np.random.shuffle(_candidates)
    _candidates = _candidates.tolist()
    x = list()
    y = list()
    while len(x) < n_points:
        index = _candidates.pop()
        label = target_image.get_data()[index[0], index[1], index[2]]
        point = index_to_point(index, target_image.affine)
        with np.errstate(invalid='raise'):
            try:
                 data = fetch_data_for_point(point, image, input_window, flip=flip, 
                                             interpolation=interpolation, spacing=np.asarray(spacing))
            except FloatingPointError:
                continue
        classification = classify(label, target_labels)
        x.append(data)
        y.append(classification)
    return x, y

