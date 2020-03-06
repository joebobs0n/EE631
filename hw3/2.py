import cv2 as cv
import numpy as np
import glob

left = np.load('left_params.npz')
right = np.load('right_params.npz')

left_intr = left['intr']
right_intr = right['intr']
left_dist = left['dist']
right_dist = right['dist']

images_left = 'images/stereo/left/*.png'
images_right = 'images/stereo/right/*.png'
left = glob.glob(images_left)
right = glob.glob(images_right)

chessboard = [10, 7]  # shape of calibration chessboard
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for corner detection
pixel = 7.4e-3  # focal length in mm (-1 for unknown)
board_points = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)  # variable initialization for board points
board_points[:, :2] = (np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2))*3.88  # calculate board points (global)

image_points_left = []
image_points_right = []
world_points = []

if len(left) is len(right):
    for i in range(len(left)):
        frame0 = cv.imread(left[i])
        frame1 = cv.imread(right[i])

        gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        corners0 = cv.findChessboardCorners(gray0, (chessboard[0], chessboard[1]), None)[1]  # detect corners
        corners1 = cv.findChessboardCorners(gray1, (chessboard[0], chessboard[1]), None)[1]  # detect corners

        if corners0 is not None:
            subpix = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)  # refine corner detection
            image_points_left.append(subpix)  # copy corner locations to image points
            world_points.append(board_points)  # append another set of world locations

        if corners1 is not None:
            subpix = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)  # refine corner detection
            image_points_right.append(subpix)  # copy corner locations to image points

    (_, _, _, _, _, R, T, E, F) = cv.stereoCalibrate(world_points, image_points_left, image_points_right, left_intr, left_dist, right_intr, right_dist, (640, 480), criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC)

    # save camera parametrs in text file and in numpy save file
    f = open('stereo_params.txt', 'w')
    f.write(f'R:\n{R}\n\n')
    f.write(f'T:\n{T}\n\n')
    f.write(f'E:\n{E}\n\n')
    f.write(f'F:\n{F}')
    f.close()
    np.savez('stereo_params.npz', R=R, T=T, E=E, F=F)

exit()
