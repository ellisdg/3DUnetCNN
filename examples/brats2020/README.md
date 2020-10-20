# BraTS 2020 Tutorial

1. Open terminal and go to ```examples/brats2020```.
2. Download training data.
3. Train the model.

```python ../../unet3d/scripts/run_trial.py --config_filename ./brats_config.json --model_filename ./brats_unet3d_baseline.h5 --machine_config_filename ../machine_configs v100_2gpu_32gb_config.json --training_log_filename brats_baseline_training_log.csv```