import numpy as np
import cv2

from models.cardmodel import CardModel
from models.cornermodel import CornerModel


URL = "http://192.168.0.13:8080/video"
THICKNESS = 7
COLOR = (0, 255, 0)

cardmodel = CardModel()
cardmodel.load_wights('models_path/resnet50_1.hdf5')
cornermodel = CornerModel(alpha=0.7, init_alpha=0.3, stop=32)
cornermodel.load_wights('models_path/corner_w1.hdf5')



def draw_corners(image, corners):
    pts = np.array([[corners[i*2], corners[i*2+1]] for i in range(4)]).reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], True, COLOR, THICKNESS)


def video(capture, out, save_video):
    while True:
        ret, frame = capture.read()
        corners = cardmodel.predict(frame)
        corners = cornermodel.predict(frame, corners)

        frame = draw_corners(frame, corners)
        if save_video:
            out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture = cv2.VideoCapture(URL)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 720))
    video(capture, out, True)
