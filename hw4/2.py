import cv2 as cv
import numpy as np
import imutils

cap_left = cv.VideoCapture('resources/footage_left.avi')
cap_right = cv.VideoCapture('resources/footage_right.avi')

subregion = [130, 150]
offset_reset = [[295, 20],
                [212, 20]]
ball_found_flg = False
counter = 0
delay_val = 1
captured_images = False

threshold_value = 60

while True:
    ret_l, frame_left = cap_left.read()
    ret_r, frame_right = cap_right.read()

    k = cv.waitKey(delay_val) & 0xff
    if k is 27:
        break
    elif k is ord(' '):
        cv.imwrite('2-frame' + str(counter) + '.jpg', disp_frame)
        counter += 5

    if counter > 20:
        delay_val = 1
        captured_images = True

    if ret_l is False or ret_r is False:
        break

    xl = offset_reset[0][0]
    yl = offset_reset[0][1]
    xr = offset_reset[1][0]
    yr = offset_reset[1][1]

    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

    cropped_left = gray_left[yl:yl + subregion[1], xl:xl + subregion[0]]
    cropped_right = gray_right[yr:yr + subregion[1], xr:xr + subregion[0]]

    threshold_left = cv.threshold(cropped_left, threshold_value, 255, cv.THRESH_BINARY)[1]
    threshold_right = cv.threshold(cropped_right, threshold_value, 255, cv.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)

    erode_left = cv.erode(threshold_left, kernel, 2)
    erode_right = cv.erode(threshold_right, kernel, 2)

    dilate_left = cv.dilate(erode_left, kernel, 2)
    dilate_right = cv.dilate(erode_right, kernel, 2)

    contours_left = cv.findContours(dilate_left.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_right = cv.findContours(dilate_right.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours_left = imutils.grab_contours(contours_left)
    contours_right = imutils.grab_contours(contours_right)

    left_found = frame_left.copy()
    if len(contours_left) > 0:
        ball_found_flg = True
        if captured_images is False:
            delay_val = 0
        c = max(contours_left, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        if radius > 2:
            cv.circle(left_found, (int(x+offset_reset[0][0]), int(y+offset_reset[0][1])), int(radius), (0, 255, 0), 2)
            cv.circle(left_found, (int(x+offset_reset[0][0]), int(y+offset_reset[0][1])), 2, (0, 0, 255), -1)

    right_found = frame_right.copy()
    if len(contours_right) > 0:
        c = max(contours_right, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        if radius > 2:
            cv.circle(right_found, (int(x + offset_reset[1][0]), int(y + offset_reset[1][1])), int(radius), (0, 255, 0),
                      2)
            cv.circle(right_found, (int(x + offset_reset[1][0]), int(y + offset_reset[1][1])), 2, (0, 0, 255), -1)

    disp_frame = np.hstack([left_found, right_found])

    cv.imshow('left mask', dilate_left)
    cv.imshow('right mask', dilate_right)
    cv.imshow('detect', disp_frame)

cv.destroyAllWindows()
exit()
