import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
import os
import sys

os.chdir(sys.path[0])

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
slow_mo = 50
var_delay = fast_forward
threshold_val = 60
itr = 2
kernel = np.ones((5, 5), np.uint8)
counter = 0

left = np.load('resources/left_params.npz')
intr_left = left['intr']
dist_left = left['dist']

right = np.load('resources/right_params.npz')
intr_right = right['intr']
dist_right = right['dist']

stereo = np.load('resources/stereo_params.npz')
R = stereo['R']
T = stereo['T']

R1, R2, P1, P2, Q = cv.stereoRectify(intr_left, dist_left, intr_right, dist_right, (640, 480), R, T)[0:5]

points = []
camera_to_catcher = [11.5, 29.5, 21.5]

file_out = open('3-points_and_impacts.txt', 'w')


def find_circle_info(cont):
    if len(cont) > 0:
        c = max(cont, key=cv.contourArea)
        ((center_x, center_y), center_radius) = cv.minEnclosingCircle(c)
        return int(center_x), int(center_y), int(center_radius)
    else:
        return -1, -1, -1


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

    x, y, rad_l = find_circle_info(left_cont)
    left_circle = [x, y]

    x, y, rad_r = find_circle_info(right_cont)
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

        pos_to_catcher = left_obj_dist - camera_to_catcher

        points.append(pos_to_catcher[0][0])

        cv.circle(left_disp, (left_circle[0], left_circle[1]), rad_l, (0, 255, 0), 2)
        cv.circle(right_disp, (right_circle[0], right_circle[1]), rad_r, (0, 255, 0), 2)

        cv.putText(left_disp, f'({int(left_obj_dist[0][0][0])} in, {int(left_obj_dist[0][0][1])} in, {int(left_obj_dist[0][0][2]/12)} ft)', (left_circle[0]-10, left_circle[1]-15), cv.QT_FONT_NORMAL, .5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(right_disp, f'({int(right_obj_dist[0][0][0])} in, {int(right_obj_dist[0][0][1])} in, {int(right_obj_dist[0][0][2]/12)} ft)', (right_circle[0]-10, right_circle[1]-15), cv.QT_FONT_NORMAL, .5, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(left_disp, f'({int(pos_to_catcher[0][0][0])} in, {int(pos_to_catcher[0][0][1])} in, {int(pos_to_catcher[0][0][2]/12)} ft)', (10, 60), cv.QT_FONT_NORMAL, .6, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(left_disp, f'Useable Frames: {len(points)}', (10, 30), cv.QT_FONT_NORMAL, .6, (0, 0, 255), 1, cv.LINE_AA)
    else:
        var_delay = fast_forward
        if len(points) > 0:
            counter += 1
            points = np.asarray(points)

            temp_z = np.linspace(min(points[:, 2]), max(points[:, 2]), len(points))
            x_coeffs = np.polyfit(points[:, 2], points[:, 0], 2)
            p = np.poly1d(x_coeffs)
            x_fit = p(temp_z)
            x_guess = p(0)
            y_coeffs = np.polyfit(points[:, 2], points[:, 1], 2)
            p = np.poly1d(y_coeffs)
            y_fit = p(temp_z)
            y_guess = p(0)
            impact_guess = [x_guess, y_guess]
            print(f'Set {counter}:\nPoints:\n{points}\nImpact guess: ({impact_guess[0]}, {impact_guess[1]})\n')
            file_out.write(f'Set {counter}:\nPoints:\n{points}\nImpact guess: ({impact_guess[0]}, {impact_guess[1]})\n\n')

            fig = plt.figure(1)
            plt.subplot(121)
            plt.scatter(points[:, 2], points[:, 0])
            plt.plot(temp_z, x_fit, 'r')
            ax = plt.gca()
            ax.invert_xaxis()
            plt.ylabel('X position relative to catcher (in)')
            plt.xlabel('Z position relative to catcher (in)')
            plt.grid('on')
            plt.subplot(122)
            plt.scatter(points[:, 2], points[:, 1])
            plt.plot(temp_z, y_fit, 'r')
            ax = plt.gca()
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.ylabel('Y position relative to catcher (in)')
            plt.xlabel('Z position relative to catcher (in)')
            plt.grid('on')
            plt.subplots_adjust(top=0.91, wspace=0.4)
            fig.suptitle(f'Impact Guess: ({np.around(impact_guess[0], 3)}, {np.around(impact_guess[1], 3)})')
            figname = '3-set' + str(counter).zfill(2) + '_plot.png'
            plt.savefig(figname)
            plt.close(1)

            points = []

    mask = np.hstack([left_dilate, right_dilate])
    display = np.hstack([left_disp, right_disp])

    cv.imshow('masked', mask)
    cv.imshow('detected', display)

cv.destroyAllWindows()
file_out.close()
exit()
