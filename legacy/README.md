# 3D U-Net Convolution Neural Network with Keras
![Tumor Segmentation Example](doc/tumor_segmentation_illusatration.gif)
## Background
The code was written to be trained using the 
[BRATS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications. 

## Tutorial using BRATS Data
### Training
1. Download the BraTS 2020 data after registering by following the steps outlined on the [BraTS 2020 competition page](https://www.med.upenn.edu/cbica/brats2020/registration.html).
Place the unzipped training and validation data folders named "MICCAI_BraTS2020_TrainingData" 
and "MICCAI_BraTS2020_ValidationData" in the ```brats/data``` folder.
2. Install Python 3 and dependencies: 
* nibabel
* keras
* pytables
* nilearn
* [SimpleITK](https://simpleitk.readthedocs.io/en/master/gettingStarted.html)
* [keras-contrib](https://github.com/keras-team/keras-contrib)

3. ```cd``` into the 3DUnetCNN repository.

4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
5. ```cd``` into the ```brats``` folder.

6. Run the training:
```
$ python train.py
```
7. Now that the model is trained, predict the BraTS validation data:
```
$ python predict.py
```
The predicted segmentations will be in the "BraTS2020_Validation_predictions".

**If you run out of memory during training:** try setting 
```config['patch_shape`] = (64, 64, 64)``` for starters. 
Also, read the "Configuration" notes at the bottom of this page.

### Write prediction images from the hold-out set of the training data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.

### Configuration
Changing the configuration dictionary in the [train.py](brats/train.py) scripts, makes it easy to test out different model and
training configurations.
I would recommend trying it out then modifying the parameters until you have satisfactory 
results. 
If you are running out of memory, try training using ```(64, 64, 64)``` shaped patches. 
Reducing the "batch_size" and "validation_batch_size" parameters will also reduce the amount of memory required for 
training as smaller batch sizes feed smaller chunks of data to the CNN. 
If the batch size is reduced down to 1 and it still you are still running 
out of memory, you could also try changing the patch size to ```(32, 32, 32)```. 
Keep in mind, though, that a smaller patch sizes may not perform as well as larger patch sizes.

## Using this code on other 3D datasets
If you want to train a 3D UNet on a different set of data, you can copy either the [train.py](brats/train.py) script and modify it to 
read in your data rather than the preprocessed BRATS data that they are currently setup to train on.

## Pre-trained Model
The following Keras model were trained on the BRATS 2020 data:
* [Model trained on BraTS2020 data](https://www.dropbox.com/s/onb87ze7t2t78h8/unet_model.h5?dl=0)
### BraTS20 Validation Set Scores:
|Dice Enhancing Tumor|Dice Whole Tumor|Dice Tumor Core|
|:-----------:|:-----------:|:---------:|
|0.65446|0.86171|0.75757|
