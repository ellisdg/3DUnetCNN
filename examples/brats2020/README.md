# BraTS 2020 Tutorial

1. Open terminal and setup 3DUNetCNN repository:
```
git clone https://github.com/ellisdg/3DUnetCNN.git
cd 3DUnetCNN
export PYTHONPATH=${PWD}:${PYTHONPATH}
``` 
2. ```cd``` into the ```brats2020``` example directory:

```cd examples/brats2020``` 

3. Download the [BraTS2020 data](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) 
to the ```brats2020``` example directory. 
The training data folder should be named ```MICCAI_BraTS2020_TrainingData```
and the validation data folder should be named ```MICCAI_BraTS2020_ValidationData```.

4. (optional) Setup the configuration file using the "create_config_lomem.ipynb" notebook
See the "create_config_lomem.ipynb" notebook for details on creating the configuration file.
If the data is in the brats2020 folder and that is the current directory, then this step should be optional.
However, if the data is somewhere else, or the training script cannot find the data, then modify that notebook to
point to the correct location for the training/validation data.

5. Run the training
```
python /path/to/unet3d/scripts/train.py --config_filename brats2020_config.json
```

You can also set the number of gpus '--ngpus' and the number of threads '--nthreads' to use during training.
The outputs will be written to a folder called 'brats2020_config' based on the name and location of the configuration 
file.
If cross-validation is used, then there will be separate folders within that folder where the model, training, and inference
results will be saved.
Any filenames in training script will also try to predict the outputs of any of keys listed in the configuration as "_filenames"
except for the "training_filenames".
For the configuration given, this means that there will be prediction folders for bratsvalidation and validation sets.