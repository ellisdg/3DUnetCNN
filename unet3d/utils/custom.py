from fcnn.utils.hcp import get_metric_data
from fcnn.utils.utils import load_json, nib_load_files


def get_metric_data_from_config(metric_filenames, config_filename, subject_id=100206):
    config = load_json(config_filename)
    if type(metric_filenames) == str:
        metrics = nib_load_files([metric_filenames])
    else:
        metrics = nib_load_files(metric_filenames)
    metric_data = get_metric_data(metrics, config["metric_names"], config["surface_names"],
                                  subject_id).T.ravel()
    return metric_data
