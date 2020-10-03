import numpy as np


def zero_mean_normalize_image_data(data, axis=(0, 1, 2)):
    return np.divide(data - data.mean(axis=axis), data.std(axis=axis))


def foreground_zero_mean_normalize_image_data(data, channel_dim=4, background_value=0, tolerance=1e-5):
    data = np.copy(data)
    if data.ndim == channel_dim or data.shape[channel_dim] == 1:
        # only 1 channel, so the std and mean calculations are straight forward
        foreground_mask = np.abs(data) > (background_value + tolerance)
        foreground = data[foreground_mask]
        mean = foreground.mean()
        std = foreground.std()
        data[foreground_mask] = np.divide(foreground - mean, std)
        return data
    else:
        # std and mean need to be calculated for each channel in the 4th dimension
        for channel in range(data.shape[channel_dim]):
            channel_data = data[..., channel]
            channel_mask = np.abs(channel_data) > (background_value + tolerance)
            channel_foreground = channel_data[channel_mask]
            channel_mean = channel_foreground.mean()
            channel_std = channel_foreground.std()
            channel_data[channel_mask] = np.divide(channel_foreground - channel_mean, channel_std)
            data[..., channel] = channel_data
        return data


def zero_floor_normalize_image_data(data, axis=(0, 1, 2), floor_percentile=1, floor=0):
    floor_threshold = np.percentile(data, floor_percentile, axis=axis)
    if data.ndim != len(axis):
        floor_threshold_shape = np.asarray(floor_threshold.shape * data.ndim)
        floor_threshold_shape[np.asarray(axis)] = 1
        floor_threshold = floor_threshold.reshape(floor_threshold_shape)
    background = data <= floor_threshold
    data = np.ma.masked_array(data - floor_threshold, mask=background)
    std = data.std(axis=axis)
    if data.ndim != len(axis):
        std = std.reshape(floor_threshold_shape)
    return np.divide(data, std).filled(floor)


def zero_one_window(data, axis=(0, 1, 2), ceiling_percentile=99, floor_percentile=1, floor=0, ceiling=1,
                    channels_axis=None):
    """

    :param data: Numpy ndarray.
    :param axis:
    :param ceiling_percentile: Percentile value of the foreground to set to the ceiling.
    :param floor_percentile: Percentile value of the image to set to the floor.
    :param floor: New minimum value.
    :param ceiling: New maximum value.
    :param channels_axis:
    :return:
    """
    data = np.copy(data)
    if len(axis) != data.ndim:
        floor_threshold = np.percentile(data, floor_percentile, axis=axis)
        if channels_axis is None:
            channels_axis = find_channel_axis(data.ndim, axis=axis)
        data = np.moveaxis(data, channels_axis, 0)
        for channel in range(data.shape[0]):
            channel_data = data[channel]
            # find the background
            bg_mask = channel_data <= floor_threshold[channel]
            # use background to find foreground
            fg = channel_data[bg_mask == False]
            # find threshold based on foreground percentile
            ceiling_threshold = np.percentile(fg, ceiling_percentile)
            # normalize the data for this channel
            data[channel] = window_data(channel_data, floor_threshold=floor_threshold[channel],
                                        ceiling_threshold=ceiling_threshold, floor=floor, ceiling=ceiling)
        data = np.moveaxis(data, 0, channels_axis)
    else:
        floor_threshold = np.percentile(data, floor_percentile)
        fg_mask = data > floor_threshold
        fg = data[fg_mask]
        ceiling_threshold = np.percentile(fg, ceiling_percentile)
        data = window_data(data, floor_threshold=floor_threshold, ceiling_threshold=ceiling_threshold, floor=floor,
                           ceiling=ceiling)
    return data


def find_channel_axis(ndim, axis):
    for i in range(ndim):
        if i not in axis and (i - ndim) not in axis:
            # I don't understand the second part of this if statement
            # answer: it is checking ot make sure that the axis is not indexed in reverse (i.e. axis 3 might be
            # indexed as -1)
            channels_axis = i
    return channels_axis


def static_windows(data, windows, floor=0, ceiling=1):
    """
    Normalizes the data according to a set of predefined windows. This is helpful for CT normalization where the
    units are static and radiologists often have a set of windowing parameters that the use that allow them to look at
    different features in the image.
    :param data: 3D numpy array.
    :param windows:
    :param floor: defaults to 0.
    :param ceiling: defaults to 1.
    :return: Array with data windows listed in the final dimension
    """
    data = np.squeeze(data)
    normalized_data = np.ones(data.shape + (len(windows),)) * floor
    for i, (l, w) in enumerate(windows):
        normalized_data[..., i] = radiology_style_windowing(data, l, w, floor=floor, ceiling=ceiling)
    return normalized_data


def radiology_style_windowing(data, l, w, floor=0, ceiling=1):
    upper = l + w/2
    lower = l - w/2
    return window_data(data, floor_threshold=lower, ceiling_threshold=upper, floor=floor, ceiling=ceiling)


def window_data(data, floor_threshold, ceiling_threshold, floor, ceiling):
    data = (data - floor_threshold) / (ceiling_threshold - floor_threshold)
    # set the data below the floor to equal the floor
    data[data < floor] = floor
    # set the data above the ceiling to equal the ceiling
    data[data > ceiling] = ceiling
    return data


def hist_match(source, template):
    """
    Source: https://stackoverflow.com/a/33047048
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)