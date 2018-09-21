import os
import timeit
import json
import glob
import nibabel as nib
from unet3d.data import combine_images, move_4d_channels_first, move_4d_channels_last
from unet3d.utils.nilearn_custom_utils.nilearn_utils import crop_img, reorder_affine
from unet3d.utils.utils import resize_affine, resample
from unet3d.normalize import normalize_data
from unet3d.training import load_old_model
from unet3d.prediction import prediction_to_image
import numpy as np
from nilearn.image import resample_to_img


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def main(threshold=0.5):
    run_times = list()
    start_time = timeit.default_timer()

    config = load_json("cascade_config.json")

    labels = [config['labels'][-1][0]]  # last item should only contain 1 label
    for label_set in config['labels'][:2][::-1]:
        labels.insert(0, set(label_set).difference(set(labels)).pop())

    models = list()
    for model_file in config['model_file']:
        models.append(load_old_model(os.path.join("./models/20180705", model_file)))

    final_model = load_old_model('final_model.h5')

    cascade_folder = os.path.abspath('cascade_predictions')
    if not os.path.exists(cascade_folder):
        os.makedirs(cascade_folder)

    cascade_plust_unet_folder = os.path.abspath('cascade_plus_unet_predictions')
    if not os.path.exists(cascade_plust_unet_folder):
        os.makedirs(cascade_plust_unet_folder)

    for validation_folder in glob.glob("./data/BRATS2018_Validation/Brats18*"):
        start = timeit.default_timer()
        print(validation_folder)
        subject_id = os.path.basename(validation_folder)
        images = list()
        for modality in ["t1", "t1ce", "flair", "t2"]:
            image_files = glob.glob(os.path.join(validation_folder, "*" + modality + ".nii.gz"))
            assert len(image_files) == 1
            image_file = os.path.abspath(image_files[0])
            image = nib.load(image_file)
            images.append(image)
        prediction_images = list()
        prediction_images_resampled = list()
        features_image = combine_images(images, axis=-1)
        features_image.to_filename(os.path.join(validation_folder, "features.nii.gz"))
        labelmap_data = np.zeros(features_image.shape[:3])
        image_to_crop = features_image

        for index, (model, label, shape) in enumerate(zip(models, labels, config['image_shape'])):
            cropped_affine, cropped_shape = crop_img(image_to_crop, return_affine=True)
            reordered_affine = reorder_affine(cropped_affine, cropped_shape)
            resized_affine = resize_affine(reordered_affine, cropped_shape, shape)
            target_affine = resized_affine
            cropped_image = resample(features_image, target_affine, shape)
            cropped_image.to_filename(os.path.join(validation_folder, "features_{}.nii.gz".format(index)))
            data = move_4d_channels_first(cropped_image.get_data())
            for i, modality in enumerate(["t1", "t1ce", "flair", "t2"]):
                feature_data = data[i]
                feature_image = cropped_image.__class__(feature_data, target_affine)
                feature_image.to_filename(os.path.join(validation_folder, "features_{}_{}.nii.gz".format(modality,
                                                                                                         index)))
            normalized_data = normalize_data(data)
            prediction_data = np.squeeze(model.predict(normalized_data[np.newaxis]))
            assert len(prediction_data.shape) == 3
            prediction_image = cropped_image.__class__(prediction_data, target_affine)
            prediction_image.to_filename(os.path.join(validation_folder, "prediction_{}.nii.gz".format(index)))
            prediction_images.append(prediction_image)
            prediction_image_resampled = resample_to_img(prediction_image, features_image, interpolation='linear')
            prediction_image_resampled.to_filename(os.path.join(validation_folder,
                                                                "prediction_resampled_{}.nii.gz".format(index)))
            prediction_images_resampled.append(prediction_image_resampled)
            prediction_index = np.greater(prediction_image_resampled.get_data(), threshold)
            label_data = np.zeros(labelmap_data.shape)
            label_data[prediction_index] = 1
            if index == 0:
                prediction_index = prediction_index
            else:
                prediction_index = np.logical_and(labelmap_data > 0, prediction_index)
            labelmap_data[prediction_index] = label
            label_image = features_image.__class__(label_data, features_image.affine)
            image_to_crop = label_image
            label_image.to_filename(os.path.join(validation_folder, "predicted_label_{}.nii.gz".format(label)))
            stop = timeit.default_timer()
            run_times.append(stop - start)

        labelmap = features_image.__class__(labelmap_data, features_image.affine)
        labelmap.to_filename(os.path.join(validation_folder, "{}.nii.gz".format(subject_id)))
        labelmap.to_filename(os.path.join(cascade_folder, "{}.nii.gz".format(subject_id)))

        image_to_crop = features_image
        shape = config['image_shape'][0]
        cropped_affine, cropped_shape = crop_img(image_to_crop, return_affine=True)
        reordered_affine = reorder_affine(cropped_affine, cropped_shape)
        resized_affine = resize_affine(reordered_affine, cropped_shape, shape)
        target_affine = resized_affine
        features_image = combine_images([features_image] + prediction_images_resampled, axis=3)
        cropped_image = resample(features_image, target_affine, shape)
        cropped_image.to_filename(os.path.join(validation_folder, "features_final.nii.gz"))
        data = move_4d_channels_first(cropped_image.get_data())
        normalized_data = normalize_data(data)
        prediction_data = np.squeeze(final_model.predict(normalized_data[np.newaxis]))
        prediction_image = cropped_image.__class__(move_4d_channels_last(prediction_data), target_affine)
        prediction_image.to_filename(os.path.join(validation_folder, "prediction_final.nii.gz"))
        prediction_image_resampled = resample_to_img(prediction_image, features_image, interpolation='linear')
        prediction_image_resampled.to_filename(os.path.join(validation_folder, "prediction_resampled_final.nii.gz"))
        resampled_prediction_data = move_4d_channels_first(prediction_image_resampled.get_data())
        labelmap = prediction_to_image(resampled_prediction_data[np.newaxis],
                                       affine=prediction_image_resampled.affine,
                                       label_map=True,
                                       labels=labels,
                                       nibabel_class=prediction_image_resampled.__class__,
                                       threshold=threshold)
        labelmap.to_filename(os.path.join(validation_folder, "{}_plus_unet.nii.gz".format(subject_id)))
        labelmap.to_filename(os.path.join(cascade_plust_unet_folder, '{}.nii.gz'.format(subject_id)))

    stop_time = timeit.default_timer()
    print("Total time: {} seconds".format(stop_time - start_time))
    print("Average time: {} seconds".format(np.mean(run_times)))
    print("Standard deviation: {} seconds".format(np.std(run_times)))


if __name__ == '__main__':
    main()
