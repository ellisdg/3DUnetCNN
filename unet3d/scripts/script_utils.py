from unet3d.utils.utils import load_json


def add_machine_config_to_parser(parser):
    parser.add_argument("--machine_config_filename",
                        help="JSON configuration file containing the number of GPUs and threads that are available "
                             "for model training.",
                        required=False)
    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use during training (default = 1). Warning: using a high number of "
                             "threads can sometimes cause the computer to run out of memory. This setting is "
                             "ignored if machine_config_filename is set.")
    parser.add_argument("--ngpus", default=1, type=int,
                        help="Number of gpus to use for training. This setting is ignored if machine_config_filename is"
                             "set.")
    parser.add_argument("--directory", default="",
                        help="Directory within which to find the training data. This setting is ignored if "
                             "machine_config_filename is set.")
    parser.add_argument("--pin_memory", action="store_true", default=False)


def get_machine_config(namespace):
    if namespace.machine_config_filename:
        print("MP Config: ", namespace.machine_config_filename)
        return load_json(namespace.machine_config_filename)
    else:
        return {"n_workers": namespace.nthreads,
                "n_gpus": namespace.ngpus,
                "use_multiprocessing": namespace.nthreads > 1,
                "pin_memory": namespace.pin_memory,
                "directory": namespace.directory}
