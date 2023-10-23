# 3D U-Net Convolution Neural Network

[[Update August 2023 - data loading is now 10x faster!](doc/Changes.md)]

* [Tutorials](#tutorials)
* [Introduction](#introduction)
* [Quick Start Guide](#quickstart)
  * [Installation](#installation)
  * [Example](#brats2020)
* [Documentation](#documentation)
* [Citation](#citation)


## Tutorials <a name="tutorials"></a>
### [Brain Tumor Segmentation (BraTS 2020)](examples/brats2020)
[![Tumor Segmentation Example](doc/viz/tumor_segmentation_illusatration.gif)](examples/brats2020)

## Introduction <a name="introduction"></a>
We designed 3DUnetCNN to make it easy to apply and control the training and application of various deep learning models to medical imaging data.
The links above give examples/tutorials for how to use this project with data from various MICCAI challenges.


## Quick Start Guide <a name="quickstart"></a>
How to train a UNet on your own data.

### Installation <a name="installation"></a>
1. Clone the repository:<br />
```git clone https://github.com/ellisdg/3DUnetCNN.git``` <br /><br />

2. Install the required dependencies<sup>*</sup>:<br />
```pip install -r 3DUnetCNN/requirements.txt``` 

<sup>*</sup>It is highly recommended that an Anaconda environment or a virtual environment is used to 
manage dependcies and avoid conflicts with existing packages.

### Create configuration file and run training <a name="brats2020"></a>
See the [Brats 2020 example](https://github.com/ellisdg/3DUnetCNN/tree/master/examples/brats2020) for a description on how to create a configuration and train a model.


## Documentation <a name="documentation"></a>
* [Configuration Guide](doc/Configuration.md)
* [Frequently Asked Questions](doc/FAQ.md)

### Still have questions? <a name="questions"></a>
Once you have reviewed the documentation, feel free to raise an issue on GitHub, or email me at david.ellis@unmc.edu.

## Citation <a name="citation"></a>
Ellis D.G., Aizenberg M.R. (2021) Trialing U-Net Training Modifications for Segmenting Gliomas Using Open Source Deep Learning Framework. In: Crimi A., Bakas S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2020. Lecture Notes in Computer Science, vol 12659. Springer, Cham. https://doi.org/10.1007/978-3-030-72087-2_4

### Additional Citations
Ellis D.G., Aizenberg M.R. (2020) Deep Learning Using Augmentation via Registration: 1st Place Solution to the AutoImplant 2020 Challenge. In: Li J., Egger J. (eds) Towards the Automatization of Cranial Implant Design in Cranioplasty. AutoImplant 2020. Lecture Notes in Computer Science, vol 12439. Springer, Cham. https://doi.org/10.1007/978-3-030-64327-0_6

Ellis, D.G. and M.R. Aizenberg, Structural brain imaging predicts individual-level task activation maps using deep learning. bioRxiv, 2020: https://doi.org/10.1101/2020.10.05.306951
