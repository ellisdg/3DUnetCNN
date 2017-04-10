# 3D U-Net Convolution Neural Network with Keras
## Background
Designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net. Currently, the network does not have the B-Spline deformations
that are mentioned in the paper. If you figure out a way to apply these to a 3D Keras CNN, let me know! PRs are always
welcome!

The code was written to be trained using the 
[BRATS](https://sites.google.com/site/braintumorsegmentation/home/brats2015) data set for brain tumors, but it can
be easily modify to be used in other 3D applications. To adapt the network, you might have to play with the input size
to get something that works for your data.

I used [Bohdan Pavlyshenko](https://www.kaggle.com/bpavlyshenko)'s 
[Kaggle kernel](https://www.kaggle.com/bpavlyshenko/data-science-bowl-2017/nodules-segmentation) for 2D U-Net
segmentation as a base for this 3D U-Net.

## How to Train Using BRATS Data
1. Download the [BRATS 2015 data set](https://sites.google.com/site/braintumorsegmentation/home/brats2015).
2. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases).
3. Convert the data to nifti format and perform image wise normalization and correction:

Import the conversion function:
```
>>> from preprocess import convert_brats_data
```
Import the configuration dictionary:
```
>>> from config import config
>>> convert_brats_data("/path/to/BRATS/BRATS2015_Training",  config["data_dir"])
```
Where ```config["data_dir"]``` is the location where the raw BRATS data will be converted to.

4. Run the training:
```
$ python training.py
```

## Configuration
The config.py file contains the default configuration of preferences.
I highly recommend looking through this file before running the training.

In training I have found that this network requires **a large amount of memory!**
For an image shape of 144x144x144 the memory required when training using cpu is **around 32GB.**
This can be reduced by reducing the image shape in the configuration file.
The code will then reduce the resolution of the input images so that they all match the given shape.

By default, the model will use the "T1", "T1c", and "FLAIR" images from the BRATS data set.
To change which modalities the model will use go to the config.py file and change the **training_modalities** list.
Adding "T2" to the list could increase the classification performance.

## Multi-label Classification
The code is setup to perform binary classification.
However, since the BRATS data does contain multi-label segmentation maps, the code could be modified to
classification of multiple labels simultaneously.
Currently, I do not have any need to add multi-label classification.
However, I would gladly accept a pull request from someone who has implemented this feature.
