from unet3d.utils.utils import load_json


def get_system_config(namespace):
    if namespace.machine_config_filename:
        print("MP Config: ", namespace.machine_config_filename)
        return load_json(namespace.machine_config_filename)
    else:
        return {"n_workers": namespace.nthreads,
                "n_gpus": namespace.ngpus,
                "use_multiprocessing": namespace.nthreads > 1,
                "pin_memory": namespace.pin_memory,
                "directory": namespace.directory}
