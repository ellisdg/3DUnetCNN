import numpy as np
import torch
# from nilearn.image.image import check_niimg
from nilearn.image.resampling import get_bounds
from nilearn.image.image import _crop_img_to as crop_img_to
import warnings


def crop_img(img, rtol=1e-8, copy=True, return_slices=False, pad=True, percentile=None, return_affine=False,
             warn=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    pad: boolean or integer
        If True, an extra slice in each direction will be added to the image. If integer > 0 then the pad width will
        be set to that integer.
    percentile: integer or None
        If not None, then the image will be crop out slices below the given percentile
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    # img = check_niimg(img)
    data = img
    if percentile is not None:
        threshold_shape = [data.shape[0]] + [1 for i in range(len(data.shape) - 1)]  # (C, 1, 1, 1) for 4D
        passes_threshold = data > torch.as_tensor(np.percentile(data, percentile,
                                                                axis=(np.arange(1,
                                                                                data.ndim))).reshape(threshold_shape))
        # basically it thresholds per channel now, but there is a bunch of hoopla to make sure the shapes work
    else:
        infinity_norm = max(-data.min(), data.max())
        passes_threshold = torch.logical_or(data < -rtol * infinity_norm,
                                            data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = torch.any(passes_threshold, dim=0)
    coords = torch.stack(torch.where(passes_threshold))

    if coords.shape[1] == 0:
        if warn:
            warnings.warn("No foreground detected. No cropping will be performed.")
        if return_affine:
            return img.affine, torch.as_tensor(img.shape[1:])
        elif return_slices:
            return
        else:
            return img

    values_min, indices_min = torch.min(coords, dim=1)
    start = values_min

    values_max, indices_max = torch.max(coords, dim=1)
    end = values_max + 1

    if int(pad) > 0:
        pad_width = int(pad)
        # pad with one voxel to avoid resampling problems
        start = torch.maximum(start - pad_width, torch.zeros(start.shape))
        end = torch.minimum(end + pad_width, torch.as_tensor(data.shape[1:]))

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    if return_affine:
        return image_slices_to_affine(img, slices), end - start

    return crop_img_to(img, slices, copy=copy)


def image_slices_to_affine(image, slices):
    affine = image.affine

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = torch.as_tensor([s.start for s in slices])
    new_origin = old_origin + torch.matmul(linear_part, new_origin_voxel)

    new_affine = torch.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin
    return new_affine


def run_with_background_correction(func, image, background=None, returns_array=False, reset_background=True,
                                   axis=(-3, -2, -1), **kwargs):
    data = image.get_data()
    if background is None:
        background = get_background_values(data, axis=axis)

    # set background to zero
    data[:] -= background
    # perform function on image
    image = func(image, **kwargs)
    # set the background back to what it was originally
    if reset_background:
        if returns_array:
            # the function called should have returned an array
            data = image
        else:
            # the function called should have returned an image
            data = image.get_data()
        data[:] += background
    return image


def get_background_values(data, axis=(-3, -2, -1)):
    background, _ = data.min(dim=axis)
    if isinstance(background, np.ndarray):
        while len(background.shape) < len(data.shape):
            background = background[..., None]
    return background


def reorder_affine(affine, shape):
    """
    Modified from nilearn.image.resampling.reorder_img and nilearn.image.resampling.resample_img
    :param affine:
    :param shape:
    :return:
    """
    Q, R = np.linalg.qr(affine[:3, :3])
    _affine = np.diag(np.abs(np.diag(R))[np.abs(Q).argmax(axis=1)])
    target_affine = np.eye(4)
    target_affine[:3, :3] = _affine
    transform_affine = np.linalg.inv(target_affine).dot(affine)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(shape[:3], transform_affine)
    offset = target_affine[:3, :3].dot([xmin, ymin, zmin])
    target_affine[:3, 3] = offset
    return target_affine
