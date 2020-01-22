import cv2 as cv
import numpy as np

cont = True
cap1 = cv.VideoCapture('ball_left_vid.avi')
cap2 = cv.VideoCapture('ball_right_vid.avi')
cap1.set(cv.CAP_PROP_FPS, 60)
cap2.set(cv.CAP_PROP_FPS, 60)

delay_time = 10
threshold_val = 10
erode_dilate_it = 3
frame_ctr = 0
disp_left = []
left_first = []
disp_right = []
right_first = []
recordFlag = False
vid_out = cv.VideoWriter('3.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 480))

while cont:
    ret1, frame_left = cap1.read()
    ret2, frame_right = cap2.read()

    if ret1 == ret2:
        if ret1:
            if frame_ctr is 0:
                left_first = frame_left
                right_first = frame_right

            disp_left = cv.absdiff(frame_left, left_first)
            disp_right = cv.absdiff(frame_right, right_first)

            cv.cvtColor(disp_left, cv.COLOR_BGR2GRAY)
            cv.cvtColor(disp_right, cv.COLOR_BGR2GRAY)
            full = np.hstack((disp_left, disp_right))

            full = cv.threshold(full, threshold_val, 255, 0)[1]
            eroded = cv.erode(full, None, iterations=erode_dilate_it)
            dilated = cv.dilate(eroded, None, iterations=erode_dilate_it)

            full_disp = np.hstack((frame_left, frame_right))
            full_disp[np.where((dilated == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

            cv.imshow('Ball Detect from Pitcher', full_disp)

            frame_ctr += 1

            if recordFlag:
                vid_out.write(full_disp)
        else:
            cap1.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
            cap2.set(cv.CAP_PROP_POS_AVI_RATIO, 0)

            frame_ctr = 0
    elif ret1 != ret2:
        print('one video finished, but the other did not')
        cont = False
    else:
        print('reached impossible state')
        cont = False

    k = cv.waitKey(delay_time) & 0xFF
    if k == 27:
        cont = False
    elif k == ord('c'):
        recordFlag = not recordFlag
    else:
        pass

cap1.release()
cap2.release()
vid_out.release()
cv.destroyAllWindows()
exit()
