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
```
{ "window": [176, 224, 144],  # image size to input into the model
  "n_features": 1,  # number of features or channels for input
  "optimizer": "Adam",  # class of PyTorch optimizer
  "loss": "DiceLoss",  # name of loss function from MONAI or Pytorch
  "reorder": true,  # changes the orientation of input images to RAS
  "n_epochs": 200,  # training lasts for n epochs
  "save_every_n_epochs": null,  # saves a model every n epochs
  "initial_learning_rate": 1e-4,  # the initial learning rate
  "early_stopping_patience": 20,  # stop training after the validation loss stops improving for n epochs
  "save_best": true,  # save the best model from training (the latest model will still be saved)
  "save_last_n_models": null,  # save the last n models from training
  "batch_size": 1,  # usually set to the number of GPUs you are using
  "validation_batch_size": 1,  # usually the same as batch_size
  "model_name": "AutocastUNet",  # name of the model class to use
  "model_kwargs": {  # arguments for initialization of the model class
    "base_width": 32,
    "encoder_blocks": [2, 2, 2, 2, 2],
    "decoder_mirrors_encoder": false,
    "input_shape": [176, 224, 144],
    "activation": "sigmoid"
  },
  "skip_validation": false,  # do not run the validation
  "iterations_per_epoch": 1,  # iterate through training n times per epoch
  "n_outputs": 1,  # number of output labels
  "sequence": "WholeVolumeSegmentationDataset",  # dataset class to use for loading data
  "sequence_kwargs": {  # arguments for initialization of all datasets (both training and validation)
    "normalization": "zero_mean",  # normalization function
    "crop": true,  # crop the input images to remove blank space
    "cropping_kwargs": {"percentile": 0.75}
    "interpolation": "linear", 
    "labels": [1],  # labels in the output label map
    "use_label_hierarchy": false  # use when the labels have heirarchical ordering such as in the BRATS challenge
  },
  "additional_training_args": {  # arguments for intitialization of training dataset only
    "flip_left_right_probability": 0.5,  # upon loading, mirrors an input/output image set L/R 50% of the time
    "random_permutation_probability": null  # if not null, will randomly permute along any axis resulting in 48 unique permutations
  },
  "additional_validation_args": {  # arguments for initialization of validation dataset only
    "random_permutation_probability": null
  },
  "training_function_kwargs": {
    "amp": true,  # use automatic mixed percision (if using "AutocastUNet", this should be true)
    "scheduler_name": "ReduceLROnPlateau",  # PyTorch scheduler class
    "scheduler_kwargs": {  # scheduler class arguments
      "patience": 10,
      "factor": 0.5,
      "min_lr": 1e-08
    }
  },
  "training_filenames = [["in_file1.nii.gz", "labelmap1.nii.gz", ...]
  "validation_filenames" = [["in_file2.nii.gz", "labelmap2.nii.gz", ...]
  "test_filenames" = [["in_file3.nii.gz", "labelmap3.nii.gz", ...]  # if applicable
```
## GPU Memory Constraints and Input Size <a name="gpu"></a>
I find that an input size of 176x224x144 works well for 32GB V100 GPUs.
If you are getting out of memory errors, try decreasing the input/window size in increments of 16
(i.e. the next increment down would be 160x208x128).
Note that each input dimension must always be divisible by the number of downsampling layers squared.
The example configuration shown above has 4 downsampling layers (5 encoding layers total) and therefore each
dimension must be divisible by 16.


### Using "--fit_gpu_mem" <a name="fitgpumem"></a>
When ```--fit_gpu_mem``` is used the ```train.py``` script will automatically set the input image size and batch size based on the amount of GPU memory and number of GPUs.
However, I recommend users setting their window/input size and batch size based on the
computer hardware available to them.
If you do not want these settings automatically set, you can adjust them yourself by making changes to the config file instead of using the
```--fit_gpu_mem``` flag.

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



