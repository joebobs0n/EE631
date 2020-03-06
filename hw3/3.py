import cv2 as cv
import numpy as np

chessboard = [10, 7]

stereo_params = np.load('stereo_params.npz')
F = stereo_params['F']

left_params = np.load('left_params.npz')
left_intr = left_params['intr']
left_dist = left_params['dist']

right_params = np.load('right_params.npz')
right_intr = right_params['intr']
right_dist = right_params['dist']

image_left = cv.imread('images/stereo/left/stereo_L0.png')
image_right = cv.imread('images/stereo/right/stereo_R0.png')

undistort_left = cv.undistort(image_left, left_intr, left_dist, None, left_intr)
undistort_right = cv.undistort(image_right, right_intr, right_dist, None, right_intr)

gray_left = cv.cvtColor(undistort_left, cv.COLOR_BGR2GRAY)
gray_right = cv.cvtColor(undistort_right, cv.COLOR_BGR2GRAY)

corners_left = cv.findChessboardCorners(gray_left, (chessboard[0], chessboard[1]), None)[1]
corners_right = cv.findChessboardCorners(gray_right, (chessboard[0], chessboard[1]), None)[1]

left_points = [0, 12, 24]
right_points = [30, 42, 54]

lines_l = []
for point in left_points:
    pt = corners_left[point]
    x = pt[0][0]
    y = pt[0][1]
    cv.circle(image_left, (x, y), 15, (0, 0, 255), 2)
    lines_l.append(cv.computeCorrespondEpilines(pt, 1, F))

lines_r = []
for point in right_points:
    cv.circle(image_right, (corners_right[point][0][0], corners_right[point][0][1]), 15, (0, 0, 255), 2)
    pt = corners_right[point]
    x = pt[0][0]
    y = pt[0][1]
    cv.circle(image_right, (x, y), 15, (0, 0, 255), 2)
    lines_r.append(cv.computeCorrespondEpilines(pt, 2, F))

x0 = 0
x1 = 640

for r in lines_l:
    y0 = int(-(r.item(0)*x0 + r.item(2)) / r.item(1))
    y1 = int(-(r.item(0)*x1 + r.item(2)) / r.item(1))
    cv.line(image_right, (x0, y0), (x1, y1), (0, 255, 0), 2)

for r in lines_r:
    y0 = int(-(r.item(0)*x0 + r.item(2)) / r.item(1))
    y1 = int(-(r.item(0)*x1 + r.item(2)) / r.item(1))
    cv.line(image_left, (x0, y0), (x1, y1), (0, 255, 0), 2)

cv.imshow('left', image_left)
cv.imshow('right', image_right)
cv.waitKey(0)

cv.destroyAllWindows()
exit()
