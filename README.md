# 3D U-Net Convolution Neural Network with Keras
![Tumor Segmentation Example](doc/tumor_segmentation_illusatration.gif)
## Background
Originally designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net.
The code was written to be trained using the 
[BRATS](http://www.med.upenn.edu/sbia/brats2017.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications. 

## Tutorial using BRATS Data
### Training
1. Download the BRATS 2020 data after registering by following the steps outlined on the [BRATS 2020 competition page](https://www.med.upenn.edu/sbia/brats2018/registration.html).
Place the unzipped training data folder named "MICCAI_BraTS2020_TrainingData" in the
```brats/data``` folder.
2. Install Python 3 and dependencies: 
```
nibabel,
keras,
pytables,
nilearn,
SimpleITK,
keras-contrib
```
3. ```cd``` into the 3DUnetCNN repository.

4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```

5. Run the training:

```
$ python train.py
```

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
Changing the configuration dictionary in the [train.py](brats/train.py) or the 
[train_isensee2017.py](brats/train_isensee2017.py) scripts, makes it easy to test out different model and
training configurations.
I would recommend trying out the Isensee et al. model first and then modifying the parameters until you have satisfactory 
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

## Pre-trained Models
The following Keras models were trained on the BRATS 2020 data:
* coming soon.
