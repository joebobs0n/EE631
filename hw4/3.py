import cv2 as cv
import numpy as np
import imutils

chessboard_dims = [10, 7]

cap_left = cv.VideoCapture('resources/footage_left.avi')
cap_right = cv.VideoCapture('resources/footage_right.avi')

subregion = [130, 150]
offset = [[295, 20],
          [212, 20]]

xl = offset[0][0]
yl = offset[0][1]
xr = offset[1][0]
yr = offset[1][1]

fast_forward = 1
slow_mo = 0
var_delay = fast_forward
threshold_val = 60
itr = 2
kernel = np.ones((5, 5), np.uint8)

left = np.load('resources/left_params.npz')
intr_left = left['intr']
dist_left = left['dist']

right = np.load('resources/right_params.npz')
intr_right = right['intr']
dist_right = right['dist']

stereo = np.load('resources/stereo_params.npz')
R = stereo['R']
T = stereo['T']

R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(intr_left, dist_left, intr_right, dist_right, (640, 480), R, T)


def find_circle_info(cont):
    if len(cont) > 0:
        c = max(cont, key=cv.contourArea)
        moments = cv.moments(c)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return int(center_x), int(center_y)
    else:
        return -1, -1


while True:
    ret_l, left_frame = cap_left.read()
    ret_r, right_frame = cap_right.read()

    k = cv.waitKey(var_delay) & 0xff
    if k is 27:
        break

    if ret_l is False or ret_r is False:
        break

    left_gray = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)

    left_crop = left_gray[yl:yl + subregion[1], xl:xl + subregion[0]]
    right_crop = right_gray[yr:yr + subregion[1], xr:xr + subregion[0]]

    left_thresh = cv.threshold(left_crop, threshold_val, 255, cv.THRESH_BINARY)[1]
    right_thresh = cv.threshold(right_crop, threshold_val, 255, cv.THRESH_BINARY)[1]

    left_erode = cv.erode(left_thresh, kernel, itr)
    right_erode = cv.erode(right_thresh, kernel, itr)

    left_dilate = cv.dilate(left_erode, kernel, itr)
    right_dilate = cv.dilate(right_erode, kernel, itr)

    left_cont = imutils.grab_contours(cv.findContours(left_dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE))
    right_cont = imutils.grab_contours(cv.findContours(right_dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE))

    x, y = find_circle_info(left_cont)
    left_circle = [x, y]

    x, y = find_circle_info(right_cont)
    right_circle = [x, y]

    left_disp = left_frame.copy()
    right_disp = right_frame.copy()
    if left_circle[0] is not -1 and right_circle[0] is not -1:
        left_circle[0] += xl
        left_circle[1] += yl
        right_circle[0] += xr
        right_circle[1] += yr

        var_delay = slow_mo

        left_dist_points = cv.undistortPoints(np.array([[[left_circle[0], left_circle[1]]]], np.float32), intr_left, dist_left, R=R1, P=P1)
        right_dist_points = cv.undistortPoints(np.array([[[right_circle[0], right_circle[1]]]], np.float32), intr_right, dist_right, R=R2, P=P2)
        disparity = np.array([[left_dist_points[0][0][0] - right_dist_points[0][0][0]]])

        left_dist_points = np.array(left_dist_points).reshape((1, 2))
        left_dist_points = np.hstack([left_dist_points, disparity]).reshape((1, 1, 3))

        right_dist_points = np.array(right_dist_points).reshape((1, 2))
        right_dist_points = np.hstack([right_dist_points, disparity]).reshape((1, 1, 3))

        left_obj_dist = cv.perspectiveTransform(left_dist_points, Q)
        right_obj_dist = cv.perspectiveTransform(right_dist_points, Q)

        cv.circle(left_disp, (left_circle[0], left_circle[1]), 2, (0, 255, 0), -1)
        cv.circle(right_disp, (right_circle[0], right_circle[1]), 2, (0, 255, 0), -1)

        cv.putText(left_disp, f'({int(left_obj_dist[0][0][0])} in, {int(left_obj_dist[0][0][1])} in, {int(left_obj_dist[0][0][2]/12)} ft)', (left_circle[0]-50, left_circle[1]-10), cv.QT_FONT_NORMAL, .6, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(right_disp, f'({int(right_obj_dist[0][0][0])} in, {int(right_obj_dist[0][0][1])} in, {int(right_obj_dist[0][0][2]/12)} ft)', (right_circle[0]-50, right_circle[1]-10), cv.QT_FONT_NORMAL, .6, (0, 0, 255), 1, cv.LINE_AA)
    else:
        var_delay = fast_forward

    cv.imshow('left', left_disp)
    cv.imshow('right', right_disp)

exit()
