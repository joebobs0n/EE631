import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

cam = cv.VideoCapture('resources/desktop.mp4')

def drawOpticalFlow(frame, prevPts, nxtPts):
    ret_frame = frame.copy()
    for pt in prevPts:
        cv.circle(ret_frame, int(tuple(pt)), 5, (0, 255, 0), 2)
    for i in range(len(prevPts)):
        cv.line(ret_frame, tuple((prevPts[i])), tuple((nxtPts[i])), (0, 0, 255), 2)
    return ret_frame

def trimBadPoints(pts, shape):
    ret_pts = pts.copy()
    it = 0
    while True:
        x = ret_pts[it][0]
        y = ret_pts[it][1]
        if x > shape[0] or x < 0 or y > shape[1] or y < 0:
            ret_pts = np.delete(ret_pts, it, axis=0)
        else:
            it += 1
        if it >= len(ret_pts):
            break
    return ret_pts

def getFrame(num_skip):
    ret = True
    last = False
    frame = None
    skipped = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            if skipped > 0:
                ret = True
                last = True
                break
            else:
                break
        else:
            if skipped >= num_skip:
                break
            else:
                skipped += 1
    return ret, last, frame, skipped

run, frame_prev = cam.read()
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
frame_shape = np.flip(np.shape(gray_prev))
pts_prev = cv.goodFeaturesToTrack(gray_prev, maxCorners=100, qualityLevel=0.01, minDistance=10)
pts_prev = pts_prev.reshape((len(pts_prev), 2))
skip_frames = 10  # iterate through for 0, 2, 5, 10
pyr_level = 4  # iterate through for 0, 2, 4
out_vid = cv.VideoWriter(f'results/1-pyr{pyr_level}_skip{skip_frames}.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30/(skip_frames if skip_frames != 0 else 1), tuple(frame_shape))

while run:
    ret, last, frame, skipped = getFrame(skip_frames)
    cv.waitKey(1) & 0xff
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        nxt_pts = cv.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None, maxLevel=pyr_level, winSize=(20, 20))[0]
        nxt_pts = nxt_pts.reshape((len(nxt_pts), 2))
        disp_frame = drawOpticalFlow(cv.cvtColor(gray_prev, cv.COLOR_GRAY2BGR), pts_prev, nxt_pts)
        
        out_vid.write(disp_frame)
        cv.imshow(f'Skipped frames: {skip_frames} | Pyrmaid level: {pyr_level}', disp_frame)

        nxt_pts = trimBadPoints(nxt_pts, frame_shape)
        gray_prev = gray.copy()
        pts_prev = nxt_pts.copy()
    else:
        break

cam.release()
out_vid.release()
cv.destroyAllWindows()
exit()
