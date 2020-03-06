import cv2 as cv
import numpy as np
import glob

chessboard = [10, 7]  # shape of calibration chessboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for corner detection
pixel = 7.4e-3  # focal length in mm (-1 for unknown)
board_points = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)  # variable initialization for board points
board_points[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)  # calculate board points (global)

for side in ['left', 'right']:  # iterate through both cameras
    world_points = []  # initialize array for world points
    image_points = []  # initialize array for image points

    file_path = 'images/mono/' + side + '/*.png'
    images = glob.glob(file_path)

    for image in images:  # loop until all calibration images (for current camera) are selected
        frame = cv.imread(image)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert from color to grayscale
        corners = cv.findChessboardCorners(gray, (chessboard[0], chessboard[1]), None)[1]  # detect corners

        if corners is None:  # if no corners detected
            print('No chessboard detected. ' + image + ' is a bad image. Try again.')  # inform user
        else:  # corners detected
            subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine corner detection
            image_points.append(subpix)  # copy corner locations to image points
            world_points.append(board_points)  # append another set of world locations

    # calculate camera calibration
    ret, intrinsic, distortion = cv.calibrateCamera(world_points, image_points, (640, 480), None, None)[0:3]
    distortion = distortion.T  # transpose distortion vector

    fSx = intrinsic[0, 0]  # pull out focal length in pixels
    focal_length = fSx * pixel if pixel > 0 else -1  # calculate focal length in mm (if possbile)

    # save camera parametrs in text file and in numpy save file
    f = open(side + '_params.txt', 'w')
    f.write(f'focal length in pixels:\n{fSx}\n\n')
    f.write(f'focal length in mm:\n{focal_length}\n\n')
    f.write(f'intrinsic paramters:\n{intrinsic}\n\n')
    f.write(f'distortion paramters:\n{distortion}')
    f.close()
    np.savez(side + '_params.npz', flpx=fSx, flmm=focal_length, intr=intrinsic, dist=distortion)

exit()  # exit out of program
