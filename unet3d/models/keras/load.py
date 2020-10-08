from keras.models import load_model as keras_load_model
from .resnet.resnet import sensitivity, specificity, j_stat, compare_scores


def load_model(filename, custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'j_stat': j_stat,
                                         'compare_scores': compare_scores}):
    return keras_load_model(filename, custom_objects=custom_objects)