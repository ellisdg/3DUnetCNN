## Input parameters
* image_shape - Shape the images will be cropped resampled to
* patch_shape - Shape of the patch input into the model. Set to None to train on the whole image.

## Output parameters
* labels - A list of the labels in the final segmented images.

## Model parameters
* pool_size - pool size for max pooling operations
* deconvolution - If false, will use upsampling instead of deconvolution.

## Training parameters
* batch_size - Batch size to use for training.
* validation_batch_size - batch size to use for validation.
* skip_blank - If true, then patches without any target labels will not be passed to the model for training.
* learning_rate - Initial model learning rate.
* n_epochs - End training after n number of epochs.

## Augmentation parameters 
* permute - Data shape must be a cube. Augments the data by permuting in various directions.
* training_patch_start_offset - Randomly offset the first patch index by up to this offset