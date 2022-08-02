### About configuration files
The configuration file determines the model architecture and how it will be trained.
This is helpful for running multiple experiments as it provides documentation for
each configuration you have experimented with. A configuration file should produce
close to the same results each time it is used for training, with the exception that 
batch size may change based on the number of gpus being used.

### Notes on using "--fit_gpu_min"
When ```--fit_gpu_min``` is used the ```train.py``` script will automatically set the input image size and batch size based on the amount of GPU memory and number of GPUs.
However, I recommend users setting their window/input size and batch size based on the
computer hardware available to them.
If you do not want these settings automatically set, you can adjust them yourself by making changes to the config file instead of using the
```--fit_gpu_mem``` flag. 
Rather than specifying the number of GPUs and threads on the command line, you can also make a configuration file for the machine you are using
and pass this using the ```--machine_config_filename``` flag. 
Click [here](../machine_configs/v100_2gpu_32gb_config.json) to see an example machine configuration JSON file.

