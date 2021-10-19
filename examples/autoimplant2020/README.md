# Automatic Cranial Implant Design: AutoImplant 2020 1st Place Solution
1. Open terminal and setup 3DUNetCNN repository:
```
git clone https://github.com/ellisdg/3DUnetCNN.git
cd 3DUnetCNN
export PYTHONPATH=${PWD}:${PYTHONPATH}
``` 
2. ```cd``` into the ```autoimplant2020``` example directory:

```cd examples/autoimplant2020``` 

3. Download the augmented dataset and un-archive it into the current directory ("examples/autoimplant2020"):

[https://zenodo.org/record/4270278](https://zenodo.org/record/4270278)

4. Train the model:

```python -W ignore::UserWarning:nilearn.image.resampling:273 ../../unet3d/scripts/train.py --config_filename ./autoimplant_config.json --model_filename ./autoimplant_unet3d_fold0.h5 --training_log_filename ./autoimplant_unet3d_fold0_training_log.csv --nthreads <nthreads> --ngpus <ngpus> --fit_gpu_mem <gpu_mem>```

```<nthreads>```,
```<ngpus>```, and
```<gpu_mem>```
should be set to the number of threads, number of GPUs, and the amount of GPU memory in GB on a single gpu that will be used for training.

```-W ignore:UserWarning:nilearn.image.resampling:273``` ignores a warning that
nilearn outputs when you resample an image with all zeros and ones using linear or continuous resampling. 
Normally, we would not want to resample a labeled image in this way, but in this case we do. The linear
resampling makes the interpolated voxels carry less weight than they would using nearest neighbor.

Note that the training will take a long time. Training on 2 V100 GPUs took 7 days to complete 35 epochs.

5. Predict the complete skulls for the held-out validation data:

```python ../../unet3d/scripts/predict.py --output_directory ./predictions/validation/fold0 --config_filename ./crossvalidation/autoimplant_config_fold0.json --model_filename ./autoimplant_unet3d_fold0.h5 --group validation --output_template "predicted_complete_skull.nii.gz" --nthreads <nthreads> --ngpus <ngpus>```

```<nthreads>``` and
```<ngpus>```
should be set to the number of threads and gpus that you are using.
The predicted complete skull volumes will be in the folder: ```./predictions/validation/fold0```

See for [Configuration.md](../../doc/Configuration.md) for tips on how to customize the configuration.

## Citation
Ellis D.G., Aizenberg M.R. (2020) Deep Learning Using Augmentation via Registration: 1st Place Solution to the AutoImplant 2020 Challenge. In: Li J., Egger J. (eds) Towards the Automatization of Cranial Implant Design in Cranioplasty. AutoImplant 2020. Lecture Notes in Computer Science, vol 12439. Springer, Cham. https://doi.org/10.1007/978-3-030-64327-0_6
