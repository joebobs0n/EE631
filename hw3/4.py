import cv2 as cv
import numpy as np

image_l = cv.imread('images/stereo/left/stereo_L0.png')
image_r = cv.imread('images/stereo/right/stereo_R0.png')

intr_l = np.load('left_params.npz')['intr']
dist_l = np.load('left_params.npz')['dist']
intr_r = np.load('right_params.npz')['intr']
dist_r = np.load('right_params.npz')['dist']
R_stereo = np.load('stereo_params.npz')['R']
T_stereo = np.load('stereo_params.npz')['T']

R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(intr_l, dist_l, intr_r, dist_r, (640, 480), R_stereo, T_stereo)

map_l1, map_l2 = cv.initUndistortRectifyMap(intr_l, dist_l, R1, P1, (640, 480), 5)
map_r1, map_r2 = cv.initUndistortRectifyMap(intr_l, dist_l, R1, P1, (640, 480), 5)

rectify_l = cv.remap(image_l, map_l1, map_l2, cv.INTER_LINEAR)
rectify_r = cv.remap(image_r, map_l1, map_l2, cv.INTER_LINEAR)

diff_l = cv.absdiff(image_l, rectify_l)
diff_r = cv.absdiff(image_r, rectify_r)

disp_l = np.copy(image_l)
disp_r = np.copy(image_r)

num_lines = 12
for i in range(num_lines):
    x0 = 0
    x1 = 640
    y = int(480 - 480/num_lines*i - 480/(2*num_lines))
    cv.line(disp_l, (x0, y), (x1, y), (0, 255, 0), 2)
    cv.line(disp_r, (x0, y), (x1, y), (0, 255, 0), 2)

cv.imshow('original left', image_l)
cv.imshow('original right', image_r)
cv.imshow('diff left', diff_l)
cv.imshow('diff right', diff_r)
cv.imshow('lines left', disp_l)
cv.imshow('lines right', disp_r)

cv.waitKey(0)

cv.destroyAllWindows()
exit()
