import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, GlobalAveragePooling2D

from pcd.models.tools import imagepreprocess, labelprocess


class CardModel:
    def __init__(self):
        self.IMG_SIZE = 64
        self.model = self.create_model()

    def create_model(self):
        resnet50 = keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)
        )

        model_resnet50 = Sequential()
        model_resnet50.add(resnet50)
        model_resnet50.add(GlobalAveragePooling2D())
        model_resnet50.add(Flatten())
        model_resnet50.add(Dense(8, activation=keras.layers.ReLU(max_value=1)))

        return model_resnet50

    def load_wights(self, path):
        self.model.load_weights(path)

    def predict(self, image):
        processed_image = imagepreprocess(image.copy(), (self.IMG_SIZE, self.IMG_SIZE))
        corners = self.model.predict(np.expand_dims(processed_image, 0))[0]
        corners = labelprocess(corners, image)
        return corners
