# BraTS 2020 Tutorial

1. Open terminal and setup 3DUNetCNN repository:
```
git clone https://github.com/ellisdg/3DUnetCNN.git
cd 3DUnetCNN
export PYTHONPATH=${PWD}:${PYTHONPATH}
``` 
2. ```cd``` into the ```brats2020``` example directory:

```cd examples/brats2020``` 

3. Download the [BraTS2020 data](https://www.med.upenn.edu/cbica/brats2020/data.html) 
to the ```brats2020``` example directory. 
The training data folder should be named ```MICCAI_BraTS2020_TrainingData```
and the validation data folder should be named ```MICCAI_BraTS2020_ValidationData```.
4. Train the model:

```python ../../unet3d/scripts/train.py --config_filename ./brats_config.json --model_filename ./brats_unet3d_baseline.h5 --training_log_filename brats_baseline_training_log.csv --nthreads <nthreads> --ngpus <ngpus> --fit_gpu_mem <gpu_mem>```

```<nthreads>```,
```<ngpus>```, and
```<gpu_mem>```
should be set to the number of threads, number of GPUs, and the amount of GPU memory in GB on a single gpu that will be used for training.

5. Predict the tumor label maps for the validation data:

```python ../../unet3d/scripts/predict.py --segment --output_directory ./predictions/validation/baseline --config_filename ./brats_config_auto.json --model_filename ./brats_unet3d_baseline.h5 --replace Training Validation --group validation --output_template "BraTS20_Validation_{subject}.nii.gz" --nthreads <nthreads> --ngpus <ngpus>```

```<nthreads>``` and
```<ngpus>```
should be set to the number of threads and gpus that you are using.
The predicted tumor label map volumes will be in the folder: ```./predictions/validation/baseline```

These label maps are ready to be submitted to the [CBICA portal](https://ipp.cbica.upenn.edu/) 
that the BraTS challenge uses to score and rank submissions.

#### Notes on configuration
The ```train.py``` script will automatically set the input image size and batch size based on the amount of GPU memory and number of GPUs.
If you do not want these settings automatically set, you can adjust them yourself by making changes to the config file instead of using the
```--fit_gpu_mem``` flag. 
Rather than specifying the number of GPUs and threads on the command line, you can also make a configuration file for the machine you are using
and pass this using the ```--machine_config_filename``` flag. 
Click [here](../machine_configs/v100_2gpu_32gb_config.json) to see an example machine configuration JSON file.
