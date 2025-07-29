import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

def build_feature_extractor():
    base_model = DenseNet201()
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return feature_extractor