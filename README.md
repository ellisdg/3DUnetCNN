# 3D U-Net Convolution Neural Network with Keras
## Background
Designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net. Currently, the network does not have the B-Spline deformations
that are mentioned in the paper. If you figure out a way to apply these to a 3D Keras CNN, let me know! PRs are always
welcome!

The code was written to be trained using the 
[BRATS](http://www.med.upenn.edu/sbia/brats2017.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications. To adapt the network, you might have to play with the input size
to get something that works for your data.

I used [Bohdan Pavlyshenko](https://www.kaggle.com/bpavlyshenko)'s 
[Kaggle kernel](https://www.kaggle.com/bpavlyshenko/data-science-bowl-2017/nodules-segmentation) for 2D U-Net
segmentation as a base for this 3D U-Net.

## How to Train Using BRATS Data
1. Download the BRATS 2017 [GBM](https://app.box.com/s/926eijrcz4qudona5vkz4z5o9qfm772d) and 
[LGG](https://app.box.com/s/ssfkb6u8fg3dmal0v7ni0ckbqntsc8fy) data. Place the unzipped folders in the 
```brats/data/original``` folder.
2. Install dependencies: 
nibabel,
keras,
pytables,
nilearn,
SimpleITK (for preprocessing only)
3. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable.
4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
5. Convert the data to nifti format and perform image wise normalization and correction:
```
$ cd brats
```
Import the conversion function and run the preprocessing:
```
$ python
>>> from preprocess import convert_brats_data
>>> convert_brats_data("data/original", "data/preprocessed")
```
6. Run the training:
```
$ python train.py
```

## Configuration
In training I have found that this network requires **a large amount of memory!**
For an image shape of 144x144x144 the memory required when training using cpu is **around 32GB.**
This can be reduced by reducing the image shape in the configuration file.
The code will then reduce the resolution of the input images so that they all match the given shape.
