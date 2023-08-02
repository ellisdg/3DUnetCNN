Currently any MONAI normalization transform is supported.

Old options that no longer work but may be implemented in the future:
* zero_mean
    z score normalization where the mean is divided by the standard deviation.
* foreground_zero_mean_normalize_image_data
    Same as zero_mean except the foreground is masked and normalized while the background remains
    the same.
* zero_floor_normalize_image_data
* zero_one_window
* static windows
    Normalizes the data according to a set of predefined windows. This is helpful for CT normalization where the
    units are static and radiologists often have a set of windowing parameters that the use that allow them to look at
    different features in the image.
    :param data: 3D numpy array.
    :param windows:
    :param floor: defaults to 0.
    :param ceiling: defaults to 1.
    :return: Array with data windows listed in the final dimension
* radiology_style_windowing
* window_data
* hist_match