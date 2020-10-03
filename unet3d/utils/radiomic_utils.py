import numpy as np
from multiprocessing import Pool, Manager, Process
from functools import partial
import os
from nilearn.image import reorder_img, resample_to_img
import matplotlib.pyplot as plt
import nibabel as nib

from .utils import update_progress, move_channels_first, move_channels_last, load_single_image
from .normalize import zero_mean_normalize_image_data as unet3d_normalize
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


def view_mdfa_image(mdfa_filename, reference_filename):
    reference_image = nib.load(reference_filename)
    mdfa_image = reorder_img(nib.load(mdfa_filename), resample='linear')
    resampled_reference_image = resample_to_img(reference_image, mdfa_image,
                                                interpolation='linear')
    mdfa_data = mdfa_image.get_data()
    reference_data = resampled_reference_image.get_data()
    midway_points = np.asarray(np.divide(mdfa_image.shape[:3], 2), np.int)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), num=1)
    axes[0, 1].imshow(np.rot90(reference_data[midway_points[0], :, :]), cmap='gray')
    axes[0, 2].imshow(np.rot90(reference_data[:, midway_points[1], :]), cmap='gray')
    axes[0, 0].imshow(np.rot90(reference_data[:, :, midway_points[2]]), cmap='gray')
    axes[1, 1].imshow(np.rot90(mdfa_data[midway_points[0], :, :,0]), cmap='gray')
    axes[1, 2].imshow(np.rot90(mdfa_data[:, midway_points[1], :,0]), cmap='gray')
    axes[1, 0].imshow(np.rot90(mdfa_data[:, :, midway_points[2], 0]), cmap='gray')
    axes[2, 1].imshow(np.rot90(mdfa_data[midway_points[0], :, :, 1:]))
    axes[2, 2].imshow(np.rot90(mdfa_data[:, midway_points[1], :, 1:]))
    axes[2, 0].imshow(np.rot90(mdfa_data[:, :, midway_points[2], 1:]))
    plt.show()

    
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


def view_input_data(data):
    midway_points = np.asarray(np.divide(data.shape[:3], 2), np.int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), num=1)
    axes[0, 1].imshow(np.rot90(data[midway_points[0], :, :,0]), cmap='gray')
    axes[0, 2].imshow(np.rot90(data[:, midway_points[1], :,0]), cmap='gray')
    axes[0, 0].imshow(np.rot90(data[:, :, midway_points[2], 0]), cmap='gray')
    axes[1, 1].imshow(np.rot90(data[midway_points[0], :, :, 1:]))
    axes[1, 2].imshow(np.rot90(data[:, midway_points[1], :, 1:]))
    axes[1, 0].imshow(np.rot90(data[:, :, midway_points[2], 1:]))
    plt.show()

    
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


def compute_distance_roc_xy(distances, truth, max_distance=100.1, step=0.01):
    x = list()
    y = list()
    for t in np.arange(0, max_distance, step):
        p_pos = distances < t
        p_neg = p_pos == False
        sensitivity = np.sum(p_pos[truth])/np.sum(truth)
        specificity = np.sum(p_neg[truth == False])/np.sum(truth == False)
        y.append(sensitivity)
        x.append(1 - specificity)
    return np.asarray(x), np.asarray(y)


def predict_from_queue(model, queue, n_iterations, batch_size):
    x = list()
    y = list()
    predictions = list()
    i = 0
    item = queue.get()
    while item is not None:
        i += 1
        _x, _y = item
        if _x is not None:
            x.append(_x)
            y.append(_y)
            if len(x) >= batch_size or queue.empty():
                p = model.predict(np.asarray(x)).reshape((len(x),))
                predictions.extend(zip(p, y))
                x = list()
                y = list()
                update_progress(i/n_iterations)
        item = queue.get()
    if len(x) > 0:
        p = model.predict(np.asarray(x)).reshape((len(x),))
        predictions.extend(zip(p, y))
        update_progress(1)
    return predictions


def read_data_into_queue(args, queue, window, reorder=False, flip=False,
                         interpolation='linear', spacing=(1, 1, 1)):
    feature_filename, target_filename, indices, target_labels = args
    feature_image = load_single_image(feature_filename, reorder=reorder)
    target_image = load_single_image(target_filename, reorder=reorder)
    target_image_data = target_image.get_data()
    for i, index in enumerate(indices):
        label = target_image_data[index[0], index[1], index[2]]
        point = index_to_point(index, target_image.affine)
        with np.errstate(invalid='raise'):
            try:
                 data = fetch_data_for_point(point,
                                             feature_image, 
                                             window=window, 
                                             flip=flip, 
                                             interpolation=interpolation, 
                                             spacing=spacing)
            except FloatingPointError:
                label = 0
                data = None
        classification = int(label in target_labels)
        queue.put((data, classification))


def fill_queue(dataset, queue, pool_size, func=read_data_into_queue, window=(64, 64, 64),
               reorder=False, flip=False, interpolation='linear', spacing=(1, 1, 1)):
    with Pool(pool_size) as pool:
        pool.map(partial(func, queue=queue, window=window, spacing=spacing,
                         reorder=reorder, interpolation=interpolation,
                         flip=flip), iterable=dataset)
    queue.put(None)


def predict_validation_data(model, datasets, window=(64, 64, 64), pool_size=15, batch_size=100, max_queue_size=1000,
                            data_reading_func=read_data_into_queue, spacing=(1, 1, 1),
                            flip=False, reorder=False, interpolation='linear'):
    predictions = dict()
    for dataset_name in datasets:
        print(dataset_name)
        n_iterations = len(datasets[dataset_name]) * len(datasets[dataset_name][0][2])
        manager = Manager()
        queue = manager.Queue(max_queue_size)
        filling_process = Process(target=fill_queue,
                                  kwargs=dict(dataset=datasets[dataset_name],
                                              queue=queue,
                                              pool_size=pool_size,
                                              func=data_reading_func,
                                              window=window,
                                              flip=flip,
                                              reorder=reorder,
                                              interpolation=interpolation,
                                              spacing=spacing))
        filling_process.start()
        predictions[dataset_name] = predict_from_queue(queue=queue,
                                                       n_iterations=n_iterations,
                                                       model=model,
                                                       batch_size=batch_size)
        filling_process.join()
        del manager
        del queue
        del filling_process
    return predictions


def plot_diagonal(ax):
    ax.plot((-1, 2), (-1, 2), color='k', linestyle='--')
    

def plot_predictions(predictions):
    fig, ax = plt.subplots(1)
    for dataset_name, dataset_predictions in predictions.items():
        plot_dataset_predictions(ax, dataset_predictions, dataset_name)
    ax.legend()
    ax.set_xlabel('1 - Specifity')
    ax.set_xlim((-.01, 1.01))
    ax.set_ylabel('Sensitivity')
    ax.set_ylim((-.01, 1.01))
    plot_diagonal(ax)
    return fig


def plot_dataset_predictions(ax, dataset_predictions, dataset_name):
    dataset_predictions = np.asarray(dataset_predictions)
    p = dataset_predictions[:, 0]
    t = dataset_predictions[:, 1]
    s0 = np.sum(t)
    t = np.asarray(t, np.bool)
    s1 = np.sum(t)
    if s1 != s0:
        raise RuntimeError('Rounding Error! {} != {}'.format(s1, s0))
    roc_x, roc_y = compute_roc_xy(p, t)
    ax.plot(roc_x, roc_y, label=dataset_name)


def pick_random_list_elements(input_list, n_elements, replace=False):
    indices = range(len(input_list))
    random_indices = np.random.choice(indices, replace=replace, size=n_elements)
    random_elements = np.asarray(input_list)[random_indices]
    return random_elements.tolist()


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

