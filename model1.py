import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import os


class Model(keras.models.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        # self.layer1 = layers.Input(shape=data.shape[1:])
        self.Conv2D_1 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.MaxPool2D_1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.Conv2D_2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.MaxPool2D_2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.Flatten = layers.Flatten()
        self.Dense = layers.Dense(128, activation="relu")
        self.Dense1 = layers.Dense(4, activation="softmax")

    def call(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.MaxPool2D_1(x)
        x = self.Conv2D_2(x)
        x = self.MaxPool2D_2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        return self.Dense1(x)


