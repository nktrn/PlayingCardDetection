import cv2
import numpy as np


class ImageProcess:
    def __init__(self, shape: tuple):
        self.shape = shape

    def __call__(self, image: np.array):
        image = cv2.resize(image, self.shape, interpolation=cv2.INTER_AREA)
        image = self.normalize(image)
        return image

    def normalize(self, image: np.array):
        image = image.astype('float32')
        image /= 255
        return image
