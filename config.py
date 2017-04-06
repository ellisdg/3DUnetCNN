import os


config = dict()
config["pool_size"] = (2, 2, 2)
config["image_shape"] = (144, 144, 144)
config["nb_channels"] = 3
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["n_labels"] = 1  # not including background
config["batch_size"] = 1
config["n_epochs"] = 50
config["data_dir"] = "./data"
config["truth_channel"] = 3
config["background_channel"] = 4
config["decay_learning_rate_every_x_epochs"] = 5
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5
config["validation_split"] = 0.8
config["hdf5_file"] = "./data.hdf5"
config["model_file"] = os.path.abspath("3d_unet_model.h5")
config["training_file"] = "./training_ids.pkl"
config["testing_file"] = "./testing_ids.pkl"
config["smooth"] = 1.
