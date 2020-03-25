import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

chessboard_dims = (10, 7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_points_left = []
image_points_right = []
board_points = np.zeros((10 * 7, 3), np.float32)
board_points[:, :2] = (np.mgrid[0:10, 0:7].T.reshape(-1, 2))*3.88636
world_points = [board_points]

left = np.load('resources/left_params.npz')
intr_left = left['intr']
dist_left = left['dist']

right = np.load('resources/right_params.npz')
intr_right = right['intr']
dist_right = right['dist']

stereo = np.load('resources/stereo_params.npz')
R = stereo['R']
T = stereo['T']

frame_left = cv.imread('resources/stereo_L0.png')
frame_right = cv.imread('resources/stereo_R0.png')

gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

corners_left = cv.findChessboardCorners(gray_left, chessboard_dims, None)[1]
corners_right = cv.findChessboardCorners(gray_right, chessboard_dims, None)[1]

subpix_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
subpix_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

corner_points_left = np.array([subpix_left[0], subpix_left[9], subpix_left[60], subpix_left[69]])
corner_points_right = np.array([subpix_right[0], subpix_right[9], subpix_right[60], subpix_right[69]])

image_points_left.append(subpix_left)
image_points_right.append(subpix_right)

R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(intr_left, dist_left, intr_right, dist_right, (640, 480), R, T)

dist_points_left = cv.undistortPoints(corner_points_left, intr_left, dist_left, R=R1, P=P1)
dist_points_right = cv.undistortPoints(corner_points_right, intr_right, dist_right, R=R2, P=P2)

cv.drawChessboardCorners(frame_left, (2, 2), corner_points_left, True)
cv.drawChessboardCorners(frame_right, (2, 2), corner_points_right, True)

disparity = np.array([[dist_points_left[0][0][0]-dist_points_right[0][0][0]],
                      [dist_points_left[1][0][0]-dist_points_right[1][0][0]],
                      [dist_points_left[2][0][0]-dist_points_right[2][0][0]],
                      [dist_points_left[3][0][0]-dist_points_right[3][0][0]]])

dist_points_left = np.array(dist_points_left).reshape((4, 2))
dist_points_left = np.hstack([dist_points_left, disparity]).reshape((4, 1, 3))
dist_points_right = np.array(dist_points_right).reshape((4, 2))
dist_points_right = np.hstack([dist_points_right, disparity]).reshape((4, 1, 3))

obj_dist_left = cv.perspectiveTransform(dist_points_left, Q)
obj_dist_right = cv.perspectiveTransform(dist_points_right, Q)

f = open('1-measurements.txt', 'w')
f.write(f'object distances left (x, y, z) in inches:\n{obj_dist_left}\n\n')
f.write(f'object distances right (x, y, z) in inches:\n{obj_dist_right}')
f.close()

cv.imwrite('1-left.jpg', frame_left)
cv.imwrite('1-right.jpg', frame_right)

exit()
