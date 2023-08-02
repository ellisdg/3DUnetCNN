from unet3d.utils.hcp import get_metric_data
from unet3d.utils.utils import load_json, load_image


def get_metric_data_from_config(metric_filenames, config_filename, subject_id=100206):
    config = load_json(config_filename)
    if type(metric_filenames) == str:
        metrics = load_image([metric_filenames])
    else:
        metrics = load_image(metric_filenames)
    metric_data = get_metric_data(metrics, config["metric_names"], config["surface_names"],
                                  subject_id).T.ravel()
    return metric_data
