import cv2 as cv
import numpy as np
import imutils

file = np.load('myParams.npz')  # read in parameters for my camera (generated in task 5)
intr = file['intr']  # read in intrinsic matrix
dist = file['dist']  # read in distortion coefficients
cam = cv.VideoCapture(0)  # setup camera
counter = 0  # name counter

while True:  # run unconditionally
    frame = cam.read()[1]  # read in frame from camera
    frame = imutils.rotate_bound(frame, -90)  # rotate image

    if frame is not None:  # check if there is a frame
        undist = cv.undistort(frame, intr, dist, None, intr)  # undistort image based on parameters
        diff = cv.absdiff(frame, undist)  # calculate difference frame between read and undistorted frames
        disp = np.hstack((frame, undist))  # stack original frame and undistorted frame side by side
        disp = np.hstack((disp, diff))  # stack display frame and difference frame side by side
        cv.imshow('frames', disp)  # display all three frames

        k = cv.waitKey(1) & 0xff  # check for button press
        if k is 27:  # check if escape button pressed
            break  # break out of unconditional loop
        elif k is ord('s'):  # check if s button pressed
            cv.imwrite('task6-img' + str(counter).zfill(2) + '.jpg', disp)  # save frame
            counter += 1  # increment name counter

cv.destroyAllWindows()  # close all windows generated by script
exit()  # close out script
