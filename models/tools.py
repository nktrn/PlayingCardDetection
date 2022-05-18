import cv2
import numpy as np


def imagepreprocess(image, shape):
    image = cv2.resize(image, shape)
    image = image.astype('float32')
    image /= 255
    return image


def labelprocess(labels, image):
    labels = np.array([
        labels[i] * image.shape[1] if i % 2 == 0
        else labels[i] * image.shape[0]
        for i in range(len(labels))
    ]).astype(int)
    return labels


def crop(image, top, bottom, left, right):
    return image[top:bottom, left:right]


def new_corner_features(image, corner, top, bottom, left, right):
    area = np.array([top, bottom, left, right]).astype(int)
    origin = area[[2, 0]]
    new_corner = corner - origin
    return crop(image.copy(), *area), origin, new_corner


def get_corner_area(image, corners, ctype):
    h, w, d = image.shape
    top_left = corners[[0, 1]]
    top_right = corners[[2, 3]]
    bottom_right = corners[[4, 5]]
    bottom_left = corners[[6, 7]]

    if ctype == 'ul':
        # extract top left corner
        corner = top_left

        right = (top_right[0] - top_left[0]) / 2 + top_left[0]
        left = max(0, 2 * top_left[0] - right)
        bottom = (bottom_left[1] - top_left[1]) / 2 + top_left[1]
        top = max(0, 2 * top_left[1] - bottom)

    if ctype == 'ur':
        # extract top right corner
        corner = top_right

        left = (top_right[0] - top_left[0]) / 2 + top_left[0]
        right = min(w, 2 * top_right[0] - left)
        bottom = (bottom_right[1] - top_right[1]) / 2 + top_right[1]
        top = max(0, 2 * top_right[1] - bottom)

    if ctype == 'br':
        # extract bottom right corner
        corner = bottom_right

        left = (bottom_right[0] - bottom_left[0]) / 2 + bottom_left[0]
        right = min(w, 2 * bottom_right[0] - left)
        top = (bottom_right[1] - top_right[1]) / 2 + top_right[1]
        bottom = min(h, 2 * bottom_right[1] - top)

    if ctype == 'bl':
        # extract bottom left corner
        corner = bottom_left

        right = (bottom_right[0] - bottom_left[0]) / 2 + bottom_left[0]
        left = max(0, 2 * bottom_left[0] - right)
        top = (bottom_left[1] - top_left[1]) / 2 + top_left[1]
        bottom = min(h, 2 * bottom_left[1] - top)

    return new_corner_features(image, corner, top, bottom, left, right)


def new_boarders(coord, l, alpha):
    """
    :param coord: corner coord x or y
    :param l: image width or hright
    :param alpha: alpha (0 < alpha < 1)
    :return:
    """
    n = int(l * alpha)
    n2 = n // 2
    if coord - n2 < 0:
        f = 0
        s = n
    elif coord + n2 > l:
        s = l
        f = l - n
    else:
        f = coord - n2
        s = coord + n2
    return f, s


def cut_image(image, corner, alpha):
    top, bottom = new_boarders(corner[1], image.shape[0], alpha)
    left, right = new_boarders(corner[0], image.shape[1], alpha)

    return new_corner_features(image, corner, top, bottom, left, right)
