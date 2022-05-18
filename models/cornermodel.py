from tensorflow import keras
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input

from models.tools import *

class CornerModel:
    def __init__(self, alpha, init_alpha, stop):
        self.IMG_SIZE = 32
        self.alpha = alpha
        self.init_alpha = init_alpha
        self.stop = stop
        self.model = self.create_model()  # model

    def create_model(self):
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        features = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)
        features = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(features)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)
        features = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(features)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)
        features = Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(features)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)
        features = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(features)
        features = MaxPooling2D((2, 2), strides=(2, 2))(features)
        fully = Flatten()(features)
        fully = Dense(256, activation='relu')(fully)
        outputs = Dense(2)(fully)
        outputs = keras.layers.ReLU(max_value=1)(outputs)
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def load_wights(self, path):
        self.model.load_weights(path)

    def corner_predict(self, image):
        processed_image = imagepreprocess(image.copy(), (self.IMG_SIZE, self.IMG_SIZE))
        corner = self.model.predict(np.expand_dims(processed_image, 0))[0]
        corner = labelprocess(corner, image)
        return corner

    def predict(self, image, coords):
        corners_coords = []
        for ctype in ['ul', 'ur', 'br', 'bl']:
            origins = []
            cimage, origin, corner = get_corner_area(image, coords, ctype)
            origins.append(origin)

            cimage, origin, corner = cut_image(cimage, corner, self.init_alpha)
            origins.append(origin)

            while cimage.shape[0] > self.stop or cimage.shape[1] > self.stop:
                corner = self.corner_predict(cimage)
                cimage, origin, corner = cut_image(cimage, corner, self.alpha)
                origins.append(origin)

            for orig in origins:
                corner[0] += orig[0]
                corner[1] += orig[1]
            corners_coords.extend(corner)

        return corners_coords