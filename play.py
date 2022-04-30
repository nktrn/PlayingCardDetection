import numpy as np
import cv2

from Card.model import CardModel
from Card.imageProcess import ImageProcess


URL = "http://192.168.0.13:8080/video"
THICKNESS = 7
COLOR = (0, 255, 0)


def draw_corners(image, corners):
    pts = np.array([[corners[i*2], corners[i*2+1]] for i in range(4)]).reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], True, COLOR, THICKNESS)


def video(capture, model: CardModel):
    while True:
        ret, frame = capture.read()

        corners = model.predict(frame)
        frame = draw_corners(frame, corners)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imageprocess = ImageProcess((108, 192))
    model = CardModel(shape=(192, 108, 3), imageprocess=imageprocess)
    model.load_wights(r'./models/model_w1.hdf5')
    capture = cv2.VideoCapture(URL)
    video(capture, model)
