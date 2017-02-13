import os
import glob
import pickle
import datetime

import numpy as np

from keras.layers import (Conv3D, AveragePooling3D, MaxPooling3D, Activation, UpSampling3D, merge, Input)
from keras import backend as K
from keras.models import Model,load_model
from keras.optimizers import Adam

import SimpleITK as sitk


pool_size = (2, 2, 2)
image_shape = (144, 240, 240)
n_channels = 3
input_shape = tuple([n_channels] + list(image_shape))
n_labels = 5
batch_size = 1
n_test_subjects = 40
z_crop = 155 - image_shape[0]


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


K.set_image_dim_ordering('th')
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model():
    inputs = Input(input_shape)
    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(conv4)

    conv5 = Conv3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling3D(size=pool_size)(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling3D(size=pool_size)(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling3D(size=pool_size)(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling3D(size=pool_size)(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv3D(n_labels, 1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_batch(batch, model):
    x_train = batch[:,:3]
    y_train = get_truth(batch)
    del(batch)
    model.train_on_batch(x_train, y_train)
    del(x_train)
    del(y_train)


def read_subject_folder(folder):
    flair_image = sitk.ReadImage(os.path.join(folder, "Flair.nii.gz"))
    t1_image = sitk.ReadImage(os.path.join(folder, "T1.nii.gz"))
    t1c_image = sitk.ReadImage(os.path.join(folder, "T1c.nii.gz"))
    truth_image = sitk.ReadImage(os.path.join(folder, "truth.nii.gz"))
    background_image = sitk.ReadImage(os.path.join(folder, "background.nii.gz"))
    return np.array([sitk.GetArrayFromImage(t1_image), 
                     sitk.GetArrayFromImage(t1c_image), 
                     sitk.GetArrayFromImage(flair_image),
                     sitk.GetArrayFromImage(truth_image),
                     sitk.GetArrayFromImage(background_image)])


def crop_data(data, background_channel=4):
    if np.all(data[background_channel, :z_crop] == 1):
        return data[:, z_crop:]
    elif np.all(data[background_channel, data.shape[1] - z_crop:] == 1):
        return data[:, :data.shape[1] - z_crop]
    else:
        upper = z_crop/2
        lower = z_crop - upper
        return data[:, lower:data.shape[1] - upper]


def get_truth(batch, truth_channel=3):
    truth = np.array(batch)[:, truth_channel]
    batch_list = []
    for sample_number in range(truth.shape[0]):
        sample_list = []
        for label in range(1, n_labels+1):
            array = np.zeros_like(truth[sample_number])
            array[truth[sample_number] == label] = 1
            sample_list.append(array)
        batch_list.append(sample_list)
    return np.array(batch_list)


def main(overwrite=False):
    model = unet_model()
    already_processed = glob.glob("model_*.h5")
    already_processed.sort()
    subject_dirs = glob.glob("../data/*/*")[len(already_processed):]
    if already_processed > 0 or overwrite:
        model = load_model(already_processed[-1], {dice_coef_loss})

    # reomove duplicate sessions
    subjects = dict()
    for dirname in subject_dirs:
        subjects[dirname.split('_')[-2]] = dirname

    if os.path.exists("training_ids.pkl") and not overwrite:
        training_ids = pickle.load("training_ids.pkl")
        testing_ids = pickle.load("testing_ids.pkl")

    else:
        subject_ids = subjects.keys()
        np.random.shuffle(subject_ids)

        training_ids = subject_ids[:n_test_subjects]
        testing_ids = subject_ids[n_test_subjects:]

        pickle_dump(training_ids, "training_ids.pkl")
        pickle_dump(testing_ids, "testing_ids.pkl")

    batch = []
    for subject_dir in subject_dirs:

        subject_id = subject_dir.split("_")[-2]
        if subject_id in testing_ids:
            continue

        batch.append(crop_data(read_subject_folder(subject_dir)))
        if len(batch) >= batch_size:
            train_batch(np.array(batch), model)
            del(batch)
            batch = []

            date = datetime.datetime.now().date()
            time = datetime.datetime.now().time()
            model_file = "model_{0}{1}{2}_{3}:{4}:{5}.h5".format(date.year, date.month, date.day,
                                                                 time.hour, time.minute, time.second)
            print("Saving: " + model_file)
            model.save(model_file)


if __name__ == "__main__":
    main(overwrite=True)
