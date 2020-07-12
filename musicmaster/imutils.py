import numpy as np
import cv2
from matplotlib import pyplot as plt


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def resize(image, width=None, height=None):
    if width is not None:
        ratio = width / image.shape[1]
        dimensions = (width, int(ratio * image.shape[0]))

    if height is not None:
        ratio = height / image.shape[0]
        dimensions = (int(ratio * image.shape[1]), height)

    else:
        dimensions = (image.shape[1], image.shape[0])

    resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    return resized


def flip(image, direction):
    flipped = cv2.flip(image, direction)

    return flipped


def plot_histogram(image, title, mask=None):
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.ylabel("No. of pixels")
    plt.xlabel('bins')

    for (chan, color) in zip(channels, colors):
        hist = cv2.calcHist(chan, [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()
