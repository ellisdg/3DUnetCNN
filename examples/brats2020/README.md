# BraTS 2020 Tutorial

1. Open terminal and setup 3DUNetCNN repository:
```
git clone https://github.com/ellisdg/3DUnetCNN.git
cd 3DUnetCNN
export PYTHONPATH=${PWD}:${PYTHONPATH}
``` 
2. ```cd``` into the ```brats2020``` example directory:

```cd examples/brats2020``` 

3. Download BraTS2020 data to the ```brats2020``` example directory. 
The training data folder should be named ```MICCAI_BraTS2020_TrainingData```.
4. Train the model:

```python ../../unet3d/scripts/train.py --config_filename ./brats_config.json --model_filename ./brats_unet3d_baseline.h5 --training_log_filename brats_baseline_training_log.csv --nthreads <nthreads> --ngpus <ngpus> --fit_gpu_mem <gpu_mem>```

```<nthreads>```,
```<ngpus>```, and
```<gpu_mem>```
should be set to the number of threads, number of GPUs, and the amount of GPU memory in GB on a single gpu that will be used for training.



#### Notes on configuration
The ```train.py``` script will automatically set the input image size and batch size based on the amount of GPU memory and number of GPUs.
If you do not want these settings automatically set, you can adjust them yourself by making changes to the config filename and not using the
```--fit_gpu_mem``` flag. 
Instead of specifying the number of GPUs and threads on the command line, you can also make a configuration file for the machine you are using
and pass this using the ```--machine_config_filename``` flag. 
Click [here](../machine_configs/v100_2gpu_32gb_config.json) to see an example machine configuration JSON file.
