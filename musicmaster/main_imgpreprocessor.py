from __future__ import print_function
from matplotlib import pyplot as plt
import mahotas
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bilateralblur_gray = cv2.bilateralFilter(gray, 3, 40, 40)
sobel = cv2.Sobel(bilateralblur_gray, cv2.CV_64F, 1, 0)
sobelx = np.uint8(np.absolute(sobel))
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
cv2.drawContours(notes, contours, -1, (0, 255, 0), 2)
cv2.imshow('contour', notes)
cv2.waitKey(0)
