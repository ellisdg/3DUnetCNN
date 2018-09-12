# 3D U-Net Convolution Neural Network with Keras
![Tumor Segmentation Example](doc/tumor_segmentation_illusatration.gif)
## Background
Originally designed after [this paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) on 
volumetric segmentation with a 3D U-Net.
The code was written to be trained using the 
[BRATS](http://www.med.upenn.edu/sbia/brats2017.html) data set for brain tumors, but it can
be easily modified to be used in other 3D applications. 

## Tutorial using BRATS Data and Python 3
### Training
1. Download the BRATS 2018 data by following the steps outlined on the [BRATS 2018 competition page](https://www.med.upenn.edu/sbia/brats2018/registration.html). Place the unzipped folders in the 
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
(nipype is required for preprocessing only) 

3. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable.

4. Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
5. Convert the data to nifti format and perform image wise normalization and correction:

cd into the brats subdirectory:
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

To run training using the original UNet model:
```
$ python train.py
```

To run training using an improved UNet model (recommended): 
```
$ python train_isensee2017.py
```
**If you run out of memory during training:** try setting 
```config['patch_shape`] = (64, 64, 64)``` for starters. 
Also, read the "Configuration" notes at the bottom of this page.

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.

### Results from patch-wise training using original UNet
![Patchwise training loss graph
](doc/brats_64cubedpatch_loss_graph.png)
![Patchwise boxplot scores
](doc/brats_64cubedpatch_validation_scores_boxplot.png)

In the box plot above, the 'whole tumor' area is any labeled area. The 'tumor core' area corresponds to the combination
of labels 1 and 4. The 'enhancing tumor' area corresponds to the 4 label. This is how the BRATS competition is scored.
The both the loss graph and the box plot were created by running the 
[evaluate.py](brats/evaluate.py) script in the 'brats' 
folder after training has been completed.

### Results from Isensee et al. 2017 model
I also trained a [model](unet3d/model/isensee2017.py) with the architecture as described in the [2017 BRATS proceedings
](https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf) 
on page 100. This [architecture](doc/isensee2017.png) employs a number of changes to the basic UNet including an 
[equally weighted dice coefficient](unet3d/metrics.py#L17), 
[residual weights](https://wiki.tum.de/display/lfdv/Deep+Residual+Networks), 
and [deep supervision](https://arxiv.org/pdf/1409.5185.pdf). 
This network was trained using the whole images rather than patches. 
As the results below show, this network performed much better than the original UNet. 

![Isensee training loss graph
](doc/isensee_2017_loss_graph.png)
![Isensee boxplot scores
](doc/isensee_2017_scores_boxplot.png)

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
If you want to train a 3D UNet on a different set of data, you can copy either the [train.py](brats/train.py) or the 
[train_isensee2017.py](brats/train_isensee2017.py) scripts and modify them to 
read in your data rather than the preprocessed BRATS data that they are currently setup to train on.

## Pre-trained Models
The following Keras models were trained on the BRATS 2017 data:
* Isensee et al. 2017: 
[model](https://www.dropbox.com/s/tgr0chjbj5oz2f7/isensee_2017_model.h5?dl=1)
([weights only](https://www.dropbox.com/s/0hp9p1e8db92fq8/isensee_2017_weights.h5?dl=1))
* Original U-Net: 
[model](https://www.dropbox.com/s/m99rqxunx0kmzn7/tumor_segmentation_model.h5?dl=1)
([weights only](https://www.dropbox.com/s/p9g3j9zm9btp8n0/tumor_segmentation_weights.h5?dl=1))

## Citations
GBM Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.KLXWJJ1Q

LGG Data Citation:
 * Spyridon Bakas, Hamed Akbari, Aristeidis Sotiras, Michel Bilello, Martin Rozycki, Justin Kirby, John Freymann, Keyvan Farahani, and Christos Davatzikos. (2017) Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2017.GJQ7R0EF
