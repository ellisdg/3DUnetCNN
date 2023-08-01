import torch


def percentile_threshold(image, percentile):
    """
    image: (batch, channels, x, y, z)
    """
    # get the foreground based on the percentile for each channel of each batch input
    mask = image > torch.quantile(torch.flatten(image, start_dim=-3), percentile, dim=-1)[..., None, None, None]
    # if any of the channels has foreground at a given voxel, we want to keep that voxel in the mask
    mask = torch.any(mask, dim=1, keepdim=True)
    return mask
