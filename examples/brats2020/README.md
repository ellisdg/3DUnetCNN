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
should be set to the number of threads, number of gpus, and the amount of gpu memory on a single gpu that will be used for training.
