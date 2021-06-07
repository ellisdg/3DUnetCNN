# How do I fix "ValueError: num_samples should be a positive integer value, but got num_samples=0"?
This error comes up when the script can't find the data to train on. It usually can be fixed by modifying the "generate_filenames_kwargs" part of the config file. Otherwise, it is possible that you haven't downloaded the data yet.
