## March 2023
* Utilizes MONAI instead of NiLearn as the image loading and processing core.
  * This results in data loading and augmentation speeds seemingly 10-20x faster in my experiments (your results may vary).
* Simplifies configuration options.
* Removes support for generating filenames as this caused more headaches than it was worth.
* Due to the above changes, old configuration files are not likely to work anymore.
* Images are now read as MONAI MetaTensor objects.
* Preprocessing is now done almost entirely in Torch.
* Adds requirements.txt
* Adds Quick Start Guide
* Adds more FAQs
* Adds Normalization documentation
* Allows for using MONAI loss classes
* Adds ATLAS example
* Removes old examples
* Using --fit_gpu_mem is no longer recommended
