


## How can I make gif visualizations like those shown in the README?
You can use the [make_gif.py](../unet3d/scripts/make_gif.py) to make your own gif visualizations.

## How can I speed up model training?
If you are only using one thread (i.e., ```--nthreads 1```), then training will 
likely be very slow. Determine how many threads the machine you are using has and use as
many as possible.

## How do I fix "ValueError: num_samples should be a positive integer value, but got num_samples=0"?
This error comes up when the script can't find the data to train on. 
Check that the ```training_filenames``` and ```validation_filenames``` in the configuration file are valid.

## How much GPU memory do I need?
It is recommended to have at least 11GB of GPU memory.
See GPU Memory and Input Size section of the [Configuration File Guide](./Configuration.md) for instructions on how to
adjust the input image size to use less memory.

## Do I need to use a GPU?
You can run model inference without a GPU, but model training will take far too long without a GPU.





