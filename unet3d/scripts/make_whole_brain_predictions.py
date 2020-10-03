import sys
import os
from fcnn.utils.utils import load_json, nib_load_files
from fcnn.utils.hcp import get_metric_data
from fcnn.predict import whole_brain_scalar_predictions
from fcnn.utils.filenames import load_subject_ids


def main():
    config_filename = sys.argv[1]
    print("Config: ", config_filename)
    config = load_json(config_filename)
    model_filename = sys.argv[2]
    print("Model: ", model_filename)

    machine_config_filename = sys.argv[3]
    print("Machine config: ", machine_config_filename)
    machine_config = load_json(machine_config_filename)

    output_directory = os.path.abspath(sys.argv[4])
    print("Output Directory:", output_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        reference_filename = sys.argv[5]
        reference_subject_id = sys.argv[6]
        reference_cifti = nib_load_files([reference_filename])
        reference_array = get_metric_data(reference_cifti, config["metric_names"], config["surface_names"],
                                          reference_subject_id)
    except IndexError:
        reference_array = None

    load_subject_ids(config)

    if "evaluation_metric" in config:
        criterion_name = config['evaluation_metric']
    else:
        criterion_name = config['loss']

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
    else:
        model_kwargs = dict()

    return whole_brain_scalar_predictions(model_filename=model_filename,
                                          subject_ids=config['validation'],
                                          hcp_dir=machine_config["directory"],
                                          output_dir=output_directory,
                                          hemispheres=config["hemispheres"],
                                          feature_basenames=config["feature_basenames"],
                                          surface_basename_template=config["surface_basename_template"],
                                          target_basenames=config["target_basenames"],
                                          model_name=config["model_name"],
                                          n_outputs=config["n_outputs"],
                                          n_features=config["n_features"],
                                          window=config["window"],
                                          criterion_name=criterion_name,
                                          metric_names=config["metric_names"],
                                          surface_names=config["surface_names"],
                                          reference=reference_array,
                                          package=config['package'],
                                          n_gpus=machine_config['n_gpus'],
                                          batch_size=config['validation_batch_size'],
                                          n_workers=machine_config["n_workers"],
                                          model_kwargs=model_kwargs)


if __name__ == '__main__':
    main()
