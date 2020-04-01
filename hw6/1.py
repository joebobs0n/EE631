import cv2 as cv
import numpy as np
import os
import sys
import imutils

os.chdir(sys.path[0])

coeff = 3
max_points = 20
single_book = cv.imread('resources/angel.jpg')
orig_shape = np.flip(np.shape(single_book)[0:2])
small_single = imutils.resize(single_book, width=int(orig_shape[0]/coeff), height=int(orig_shape[1]/coeff))
gray_single = cv.cvtColor(small_single, cv.COLOR_BGR2GRAY)
pts = cv.goodFeaturesToTrack(gray_single, maxCorners=max_points, qualityLevel=0.001, minDistance=20)
disp_frame = small_single.copy()

for pt in pts:
    cv.circle(disp_frame, tuple(pt[0]), 8, (0, 255, 0), 2)
    
cv.imshow('viewer', disp_frame)
cv.imwrite('results/1-reference-pts.jpg', disp_frame)

cv.waitKey(0)
    
cv.destroyAllWindows()
exit()
