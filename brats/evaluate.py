import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction) + 1e-20)


def main():
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    for case_folder in glob.glob("prediction/validation_case*"):
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
    df = pd.DataFrame.from_records(rows, columns=header)
    df.to_csv("./prediction/brats_scores.csv")

    plt.boxplot(df.values, labels=df.columns)
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")

    training_df = pd.read_csv("./training.log").set_index('epoch')

    plt.plot(training_df['loss'].values)
    plt.plot(training_df['val_loss'].values)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
