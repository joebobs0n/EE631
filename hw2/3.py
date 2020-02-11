import cv2 as cv
import numpy as np
import glob
import os

file = np.load('params.npz')  # read in parameters from task 2 calibration

fl = file['fl']  # read in focal length
intr = file['intr']  # read in intrinsic parameters
dist = file['dist']  # read in distortion parameters

images = glob.glob('correction_imgs/*.jpg')  # group all images in correction folder into array

for img in images:  # work through each image
    frame = cv.imread(img)  # read in image
    undistorted = cv.undistort(frame, intr, dist, None, intr)  # undistort image based on calibration paramters
    disp = cv.absdiff(frame, undistorted)  # calculate absolute difference between distorted and undistorted images
    name = 'task3-' + os.path.relpath(img, 'correction_imgs/')  # create save file name
    cv.imwrite(name, disp)  # save image

cv.destroyAllWindows()  # destroy all windows created by script
exit()  # close out script
