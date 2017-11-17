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

## Tutorial using BRATS Data
### Training
1. Download the BRATS 2017 [GBM](https://app.box.com/s/926eijrcz4qudona5vkz4z5o9qfm772d) and 
[LGG](https://app.box.com/s/ssfkb6u8fg3dmal0v7ni0ckbqntsc8fy) data. Place the unzipped folders in the 
```brats/data/original``` folder.
2. Install dependencies: 
```
nibabel,
keras,
pytables,
nilearn,
SimpleITK,
nipype
```
(the last two are for preprocessing only)
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
 
### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python predict.py
```
A the predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.

### Configuration
By changing the configuration dictionary in the ```brats/train.py``` file, makes it easy to test out different model and
training configurations. If you are running out of memory, try reducing the "batch_size" parameter. A smaller batch size 
will feed smaller chunks of data to the CNN. If the batch size is reduced down to 1 and it still you are still running 
out of memory, you could also try changing the patch size to ```(32, 32, 32)```. Keep in mind, though, that a smaller
patch size will likely not perform as well.

## Using this code on other 3D datasets
If you want to train a 3D UNet on a different set of data, you can copy the ```brats/train.py``` file and modify to 
read in your data rather than the preprocessed BRATS data that is currently setup to train on.
