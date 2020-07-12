from __future__ import print_function
from matplotlib import pyplot as plt
import mahotas
import argparse
import cv2
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())
image = cv2.imread(args['image'])
# shifted = imutils.translate(image, -100, 60)
# cv2.imshow('Image', image)
# cv2.imshow('shifted', shifted)
# cv2.imshow('rotated', imutils.rotate(image, 45, None, 2.0))
# cv2.imshow('resized', imutils.resize(image, width=150))
# rectangle = np.zeros((300, 300), dtype='uint8')
# cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
# circle = np.zeros((300, 300), dtype='uint8')
# cv2.circle(circle, (150, 150), 150, 255, -1)
# modified = cv2.bitwise_not(cv2.bitwise_xor(rectangle, circle))
# cv2.imshow('components r', r)
# cv2.imshow('components g', g)
# cv2.imshow('components b', b)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# cv2.imshow('gray', gray)
# cv2.imshow('hsv', hsv)
# cv2.imshow('lab', lab)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# channels = cv2.split(image)
# colors = ('b', 'g', 'r')
# plt.figure()
# plt.title('Image Histogram')
# plt.xlabel('bins')
# plt.ylabel('no. of pixels')
# red = np.zeros((3, 300, 300), dtype='uint8')
# red[0][0:300][0:300] = 200
# red[2][0:300][0:300] = 100
# cv2.imshow('test', red)
# for (chan, color) in zip(channels, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])

# fig = plt.figure()
# ax = fig.add_subplot(131)
# hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# p = ax.imshow(hist, interpolation='nearest')
# ax.set_title('3D color histogram for G and B and R')
# plt.colorbar(p)
# imutils.plot_histogram(image, 'test title')
# avg = cv2.blur(image, (5, 5))
# gauss = cv2.GaussianBlur(image, (5, 5), 0)
# cv2.imshow('average', avg)
# cv2.imshow('gaussian', gauss)
# image = cv2.bitwise_not(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
medianblur_gray = cv2.medianBlur(gray, 5)
bilateralblur_gray = cv2.bilateralFilter(gray, 3, 40, 40)
# cv2.imshow('gray_blurred', blurred_gray)
# (T, thresh) = cv2.threshold(blurred_gray, 52, 255, cv2.THRESH_BINARY_INV)
# thresh = cv2.adaptiveThreshold(blurred_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
# cv2.imshow('adaptive thresh', thresh)
# andmasked = cv2.bitwise_and(image, image, mask=thresh)
# cv2.imshow('threshold', andmasked)
# gradient = cv2.Laplacian(gray, cv2.CV_64F)
# lap = np.uint8(np.absolute(gradient))
# cv2.imshow('gradients', lap)
sobel = cv2.Sobel(bilateralblur_gray, cv2.CV_64F, 1, 0)
sobelx = np.uint8(np.absolute(sobel))
sobel = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1)
sobely = np.uint8(np.absolute(sobel))
# cv2.imshow('sobelx', sobelx)
# cv2.imshow('sobely', sobely)
blurred_sobely = cv2.GaussianBlur(sobely, (11, 11), 0) #.bilateralFilter(sobely, 3, 40, 40)
cv2.imshow('sobely', blurred_sobely)
# contour = cv2.Canny(blurred_gray, 50, 100)
count, hierarchy = cv2.findContours(sobelx.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('# of notes in the image: {}'.format(len(count)))
notes = image.copy()
contours = list(count)
total = 0
for k in range(len(count)):
    total += len(count[k])
avg_contour_size = total/len(count)
for j in range(len(count)):
    i = 0
    for i in range(len(contours)):
        if len(contours[i]) < avg_contour_size:
            contours.pop(i)
            break

print('# of notes after filter: {}'.format(len(contours)))
print(contours[1][1])
cv2.drawContours(notes, contours, 1, (0, 255, 0), 2)
# cv2.imshow('contour', notes)
# cv2.imwrite('test_write.jpg', notes)
cv2.waitKey(0)
