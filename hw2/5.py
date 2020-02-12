import cv2 as cv
import numpy as np
import imutils

cam = cv.VideoCapture(0)  # setup camera
chessboard = [9, 7]  # chessboard corner resolution
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for subpixel detection
savePath = 'own_imgs/'  # save path for own calibration images
continueFlag = True  # continue flag for while loop
frameCounter = 0  # counts which frame is currently being taken
pixel = -1  # pixel size in mm (-1 means information unknown)
targetNum = 40  # target number of necessary frames for calibration
board_points = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)  # create vector to store corner world locations
board_points[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)  # generate world locations
world_points = []  # array to store full world points
image_points = []  # array to store full image/frame points

while continueFlag:  # continue as long as flag true
    frame = []  # initialize frame array
    while True:  # unconditional while loop for finding suitable frame
        ret, temp = cam.read()  # read frame from camera
        if not ret:  # no frame found
            cv.destroyAllWindows()  # close all windows created by script
            print('No frame available. Exiting.')  # inform user
            exit()  # close out script

        temp = imutils.rotate_bound(temp, -90)  # rotate image by 90 degrees (since my camera is mounted sideways)
        cv.imshow('preview', temp)  # display rotated frame

        k = cv.waitKey(1) & 0xff  # check for keystroke
        if k is ord(' '):  # if space key detected
            cv.imwrite(savePath + 'img' + str(frameCounter).zfill(2) + '.jpg', temp)  # save image to directory
            frame = np.copy(temp)  # save image to frame array for processing
            break  # break from unconditional loop
        elif k is 27:  # if escape key detected
            cv.destroyAllWindows()  # close all windows created by script
            print('Exiting script')  # inform user
            exit()  # close out script

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert frame data to grayscale
    corners = cv.findChessboardCorners(gray, (chessboard[0], chessboard[1]), None)[1]  # detect corners

    if corners is None:  # if no corners detected
        print('No chessboard corners detected. Try again.')  # inform user and try again
    else:  # corners are detected
        subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine using subpixel detection
        image_points.append(subpix)  # append image subpixel points to image points array
        world_points.append(board_points)  # append new set of world coords to world array
        disp = cv.drawChessboardCorners(frame, (chessboard[0], chessboard[1]), subpix, True)  # draw chessboard on frame
        cv.imshow('process frame preview', disp)  # display processed frame with chessboard points

        frameCounter += 1  # increment frame counter
        print(frameCounter)  # printout current frame counter

    if frameCounter >= targetNum:  # check if target number of frames reached
        continueFlag = False  # break out of flagged while loop

temp_img = cv.imread(savePath + 'img00.jpg')  # read in resolution/shape of first saved image
# run camera calibration on captured images/points
ret, intrinsic, distortion = cv.calibrateCamera(world_points, image_points, np.shape(temp_img)[0:2], None, None)[0:3]
distortion = distortion.T  # transpose distortion coefficients

fSx = intrinsic[0, 0]  # find focal length in pixels
focal_length = fSx * pixel if pixel > 0 else -1  # calculate focal length in mm if there is pixel size data

f = open('task5.txt', 'w')  # open file stream for writing
f.write(f'focal length in pixels:\n{fSx}\n\n')  # store focal length in pixels
f.write(f'focal length in mm:\n{focal_length}\n\n')  # store focal length in mm
f.write(f'intrinsic paramters:\n{intrinsic}\n\n')  # store intrinsic parameters
f.write(f'distortion paramters:\n{distortion}')  # store distortion parameters
f.close()  # close file

np.savez('myParams.npz', flpx=fSx, flmm=focal_length, intr=intrinsic, dist=distortion)  # save my camera params

cv.destroyAllWindows()  # close all windows created by script
exit()  # close out script
