from tensorflow import keras

from tensorflow.keras import layers
import os


def createModel(train_data=None):
    if os.path.exists('./model/model.h5') and train_data is None:
        try:
            print(__name__)
            model = keras.models.load_model('./model/model.h5')
            print("returned")
            return model
        except Exception as e:
            print("error")


    elif train_data is not None:
        model = keras.Sequential([

            keras.Input(shape=train_data.shape[1:]),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),

            layers.Dropout(0.5),
            layers.Dense(4, activation="softmax")

        ])
        return model
