from tensorflow import keras
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np

from Card.imageProcess import ImageProcess


class CardModel:
    def __init__(self, shape: tuple, imageprocess: ImageProcess):
        self.shape = shape
        self.imageprocess = imageprocess
        self.model = self._create_model()

    def _create_model(self):
        inputs = Input(shape=self.shape)

        features = Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
        features = MaxPooling2D((3, 3), strides=(3, 3))(features)
        features = Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(features)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)

        fully = Flatten()(features)
        fully = Dense(256, activation='relu')(fully)
        fully = Dense(256, activation='relu')(fully)

        outputs = Dense(8)(fully)
        outputs = keras.layers.ReLU(max_value=1)(outputs)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def load_wights(self, path: str):
        self.model.load_weights(path)

    def resize_labels(self, image, corners):
        corners[::2] = corners[::2] * image.shape[1]
        corners[1::2] = corners[1::2] * image.shape[0]
        return corners

    def predict(self, image: np.array):
        corners = self.model.predict(np.expand_dims(self.imageprocess(image), 0))[0]
        corners = self.resize_labels(image, corners)
        corners = corners.astype(int)
        return corners
