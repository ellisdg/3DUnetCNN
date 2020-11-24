#### Notes on configuration
The ```train.py``` script will automatically set the input image size and batch size based on the amount of GPU memory and number of GPUs.
If you do not want these settings automatically set, you can adjust them yourself by making changes to the config file instead of using the
```--fit_gpu_mem``` flag. 
Rather than specifying the number of GPUs and threads on the command line, you can also make a configuration file for the machine you are using
and pass this using the ```--machine_config_filename``` flag. 
Click [here](../machine_configs/v100_2gpu_32gb_config.json) to see an example machine configuration JSON file.