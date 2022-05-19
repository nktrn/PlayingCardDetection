import io
import cv2
import base64
import numpy as np

from PIL import Image

from pcd import cardmodel, cornermodel


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def decode_image(img_str):
    image = np.asarray(bytearray(img_str), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def encode_image(image):
    data = io.BytesIO()
    image = Image.fromarray(image)
    image = image.convert('RGB')
    image.save(data, "JPEG")
    encoded_image_data = base64.b64encode(data.getvalue())
    return encoded_image_data.decode()


def draw_corners(image, corners):
    pts = np.array([[corners[i*2], corners[i*2+1]] for i in range(4)]).reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], True, (0, 0, 0), 14)


def predict(image):
    corners = cardmodel.predict(image)
    corners = cornermodel.predict(image, corners)
    return corners
