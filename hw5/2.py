import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

cam = cv.VideoCapture('resources/desktop.mp4')

run, frame_prev = cam.read()
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
frame_shape = np.flip(np.shape(gray_prev))
skip_frames = 10  # iterate through for 0, 2, 5, 10
orig_prev = (890, 305)
tmpl_dims = (150, 100)
tmpl_prev = frame_prev[orig_prev[1]:orig_prev[1]+tmpl_dims[1], orig_prev[0]:orig_prev[0]+tmpl_dims[0]]
pts_prev = cv.goodFeaturesToTrack(cv.cvtColor(tmpl_prev, cv.COLOR_BGR2GRAY), maxCorners=10, qualityLevel=0.01, minDistance=5)

out_vid = cv.VideoWriter(f'results/2-skip{skip_frames}.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30/(skip_frames if skip_frames != 0 else 1), tuple(frame_shape))

def drawOpticalFlow(frame, prevPts, nxtPts, prevOrig=(0, 0), nxtOrig=(0, 0)):
    prevPts = prevPts.reshape(-1, 2)
    nxtPts = nxtPts.reshape(-1, 2)
    
    ret_frame = frame.copy()
    for pt in prevPts:
        cv.circle(ret_frame, (int(pt[0] + prevOrig[0]), int(pt[1] + prevOrig[1])), 5, (0, 255, 0), 2)
    for i in range(len(prevPts)):
        norms = np.asarray(np.zeros(len(nxtPts)))
        for j in range(len(nxtPts)):
            norms[j] = np.linalg.norm([prevPts[i][0] - nxtPts[j][0], prevPts[i][1] - nxtPts[j][1]], ord=2)
        index = np.argmin(norms)
        cv.line(ret_frame, (int(prevPts[i][0] + prevOrig[0]), int(prevPts[i][1] + prevOrig[1])), (int(nxtPts[index][0] + nxtOrig[0]), int(nxtPts[index][1] + nxtOrig[1])), (0, 0, 255), 2)
    return ret_frame

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

while run:
    ret, last, frame_next, skipped = getFrame(skip_frames)
    # cv.waitKey(1)
    if ret == True:
        match_frame = cv.matchTemplate(frame_next, tmpl_prev, cv.TM_CCOEFF_NORMED)
        orig_next = cv.minMaxLoc(match_frame)[3]
        tmpl_next = frame_prev[orig_next[1]:orig_next[1]+tmpl_dims[1], orig_next[0]:orig_next[0]+tmpl_dims[0]]
        pts_next = cv.goodFeaturesToTrack(cv.cvtColor(tmpl_next, cv.COLOR_BGR2GRAY), maxCorners=10, qualityLevel=0.01, minDistance=5)

        disp_frame = cv.cvtColor(cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        disp_frame = drawOpticalFlow(disp_frame, pts_prev, pts_next, orig_prev, orig_next)
        cv.rectangle(disp_frame, orig_prev, (orig_prev[0] + tmpl_dims[0], orig_prev[1] + tmpl_dims[1]), (255, 220, 150), 2)
        
        frame_prev = frame_next.copy()
        orig_prev = orig_next
        pts_prev = pts_next
        
        out_vid.write(disp_frame)
        # cv.imshow('view', disp_frame)
    else:
        break

print('Done')
cam.release()
out_vid.release()
cv.destroyAllWindows()
exit()
