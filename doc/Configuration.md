# Configuration File Guide

* [Introduction](#introduction)
* [Configuration Example](#example)
* [GPU Memory Constraints and Input Size](#gpu)
  * [Using "--fit_gpu_mem"](#fitgpumem)
* [Normalization](#norm)
* [Machine Configuration File](#machine)

## Introduction <a name="introduction"></a>
The configuration file determines the model architecture and how it will be trained.
This is helpful for running multiple experiments as it provides documentation for
each configuration you have experimented with. A configuration file should produce
similar results each time it is used for training.


## Configuration Example <a name="example"></a>
Example python code to setup the configuration file for BraTS 2020 data.
```
config = dict()

model_config = dict()
model_config["name"] = "DynUNet"  # network model name from MONAI
# set the network hyper-parameters
model_config["in_channels"] = 4  # 4 input images for the BraTS challenge
model_config["out_channels"] = 3   # whole tumor, tumor core, enhancing tumor
model_config["spatial_dims"] = 3   # 3D input images
model_config["deep_supervision"] = False  # do not check outputs of lower layers
model_config["strides"] = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]][:-1]  # number of downsampling convolutions
model_config["filters"] = [64, 96, 128, 192, 256, 384, 512, 768, 1024][:len(model_config["strides"])]  # number of filters per layer
model_config["kernel_size"] = [[3, 3, 3]] * len(model_config["strides"])  # size of the convolution kernels per layer
model_config["upsample_kernel_size"] = model_config["strides"][1:]  # should be the same as the strides

# put the model config in the main config
config["model"] = model_config

config["optimizer"] = {'name': 'Adam', 
                       'lr': 0.001}  # initial learning rate

# define the loss
config["loss"] = {'name': 'GeneralizedDiceLoss', # from Monai
                  'include_background': False,  # we do not have a label for the background, so this should be false
                  'sigmoid': True}  # transform the model logits to activations

# set the cross validation parameters
config["cross_validation"] = {'folds': 5,  # number of cross validation folds
                              'seed': 25},  # seed to make the generation of cross validation folds consistent across different trials
# set the scheduler parameters
config["scheduler"] = {'name': 'ReduceLROnPlateau', 
                       'patience': 10,  # wait 10 epochs with no improvement before reducing the learning rate
                       'factor': 0.5,   # multiply the learning rate by 0.5
                       'min_lr': 1e-08}  # stop reducing the learning rate once it gets to 1e-8

# set the dataset parameters
config["dataset"] = {'name': 'SegmentationDatasetPersistent',  # 'Persistent' means that it will save the preprocessed outputs generated during the first epoch
# However, using 'Persistent', does also increase the time of the first epoch compared to the other epochs, which should run faster
  'desired_shape': [128, 128, 128],  # resize the images to this shape, increase this to get higher resolution images (increases computation time and memory usage)
  'labels': [2, 1, 4],  # 1: necrotic center; 2: edema, 3: enhancing tumor
  'setup_label_hierarchy': True,  # changes the labels to whole tumor (2, 1, 4), tumor core (1, 4), and enhancing tumor (4) to be consistent with the challenge  'normalization': 'NormalizeIntensityD',  # z score normalize the input images to zero mean unit standard deviation
  'normalization_kwargs': {'channel_wise': True, "nonzero": False},  # perform the normalization channel wise and include the background
  'resample': True,  # resample the images when resizing them, otherwise the resize could crop out regions of interest
  'crop_foreground': True,  # crop the foreground of the images
                    }
config["training"] = {'batch_size': 1,  # number of image/label pairs to read at a time during training
  'validation_batch_size': 1,  # number of image/label pairs to read at atime during validation
  'amp': False,  # don't set this to true unless the model you are using is setup to use automatic mixed precision (AMP)
  'early_stopping_patience': None,  # stop the model early if the validaiton loss stops improving
  'n_epochs': 250,  # number of training epochs, reduce this if you don't want training to run as long
  'save_every_n_epochs': None,  # save the model every n epochs (otherwise only the latest model will be saved)
  'save_last_n_models': None,  # save the last n models 
  'save_best': True}  # save the model that has the best validation loss

# get the training filenames
config["training_filenames"] = list()

# if your BraTS data is stored somewhere else, change this code to fetch that data
for subject_folder in sorted(glob.glob("BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*")):
    if not os.path.isdir(subject_folder):
        continue
    image_filenames = sorted(glob.glob(os.path.join(subject_folder, "*.nii")))
    for i in range(len(image_filenames)):
        if "seg" in image_filenames[i].lower():
            label = image_filenames.pop(i)
            break
    assert len(image_filenames) == 4
    config["training_filenames"].append({"image": image_filenames, "label": label})


config["bratsvalidation_filenames"] = list()  # "validation_filenames" is reserved for the cross-validation, so we will call this bratsvalidation_filenames
for subject_folder in sorted(glob.glob("BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*")):
    if not os.path.isdir(subject_folder):
        continue
    image_filenames = sorted(glob.glob(os.path.join(subject_folder, "*.nii")))
    assert len(image_filenames) == 4
    config["bratsvalidation_filenames"].append({"image": image_filenames})
  
```
## GPU Memory Constraints and Input Size <a name="gpu"></a>
I find that an input size of 176x224x144 works well for 32GB V100 GPUs.
If you are getting out of memory errors, try decreasing the input/window size in increments of 16
(i.e. the next increment down would be 160x208x128).
Note that each input dimension must always be divisible by the number of downsampling layers squared.
The example configuration shown above has 4 downsampling layers (5 encoding layers total) and therefore each
dimension must be divisible by 16.

## Normalization <a name="norm"></a>
Normalization can utilize any function in the [normalize.py](../unet3d/utils/normalize.py) file.
To use multiple normalization functions in order, you may specify a list of normalization functions.
You may also specify ```normalization_kwargs``` to further refine the normalization techniques.
If you provided a list of normalization techniques, then any ```normalization_kwargs``` must be 
listed under the name of the respective normalization function.
See [Normalization documentation](Normalization.md) for more details.

## Machine Configuration File <a name="machine"></a>
Rather than specifying the number of GPUs and threads on the command line, you can also make a configuration file for the machine you are using
and pass this using the ```--machine_config_filename``` flag. 
Click [here](../machine_configs/v100_2gpu_32gb_config.json) to see an example machine configuration JSON file.



