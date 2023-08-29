import torch
import numpy as np
import monai.inferers


def percentile_threshold(image, percentile):
    """
    image: (channels, x, y, z)
    """
    # get the foreground based on the percentile for each channel
    mask = image > torch.from_numpy(np.percentile(torch.flatten(image, start_dim=-3),
                                                  percentile * 100, axis=-1)[..., None, None, None])
    # if any of the channels has foreground at a given voxel, we want to keep that voxel in the mask
    mask = torch.any(mask, dim=-4, keepdim=True)
    return mask
