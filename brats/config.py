import os


config = dict()
config["pool_size"] = (2, 2, 2)
config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
config["n_labels"] = 1  # not including background
config["batch_size"] = 1
config["n_epochs"] = 50
config["data_dir"] = os.path.abspath("./brats/data")
config["decay_learning_rate_every_x_epochs"] = 10
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5
config["validation_split"] = 0.8
config["hdf5_file"] = os.path.abspath("./brats/data.hdf5")
config["model_file"] = os.path.abspath("./brats/3d_unet_model.h5")
config["training_file"] = os.path.abspath("./brats/training_ids.pkl")
config["validation_file"] = os.path.abspath("./brats/validation_ids.pkl")
config["smooth"] = 1.
config["modalities"] = ["T1", "T1c", "Flair", "T2"]
config["training_modalities"] = ["T1", "T1c", "Flair"]  # set this to the modalities that you want the model to use
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["background_channel"] = config["nb_channels"] + 1
config["deconvolution"] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.
# divide the number of filters used by by a given factor. This will reduce memory consumption.
config["downsize_nb_filters_factor"] = 1
