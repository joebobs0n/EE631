import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

cam = cv.VideoCapture('resources/desktop.mp4')

skip_frames = 0
resize_coeff = 2
corners = 50
run, prev_frame = cam.read()
prev_frame = cv.resize(prev_frame, (int(1920/resize_coeff), int(1080/resize_coeff)))
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=corners, qualityLevel=0.001, minDistance=10)

out_vid = cv.VideoWriter(f'results/3-skip{skip_frames}.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30/(skip_frames if skip_frames != 0 else 1), tuple(np.flip(np.shape(prev_gray))))

def drawOpticalFlow(frame, prevPts, nxtPts):
    ret_frame = frame.copy()
    
    for pt in prevPts:
        cv.circle(ret_frame, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), 2)
    for i in range(len(prevPts)):
        cv.line(ret_frame, (int(prevPts[i][0][0]), int(prevPts[i][0][1])), (int(nxtPts[i][0][0]), int(nxtPts[i][0][1])), (0, 0, 255), 2)
        
    return ret_frame

def findNextPoints(prevFrame, nextFrame, prevPts, windowSize=(20, 20)):
    frame_shape = np.flip(np.shape(prevFrame)[0:2])
    prevFrame = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    nextFrame = cv.cvtColor(nextFrame, cv.COLOR_BGR2GRAY)
    ret_prev = prevPts.copy()
    ret_next = prevPts.copy()
    
    it = 0
    while True:
        prev_x = ret_prev[it][0][0]
        prev_y = ret_prev[it][0][1]
        
        if (prev_x >= windowSize[0]/2) and (prev_x < (frame_shape[0] - windowSize[0]/2)) and (prev_y >= windowSize[1]/2) and (prev_y < (frame_shape[1] - windowSize[1]/2)):
            template = prevFrame[int(prev_y - windowSize[1]/2):int(prev_y + windowSize[1]/2), int(prev_x - windowSize[0]/2):int(prev_x + windowSize[0]/2)]
            res = cv.matchTemplate(nextFrame, template, cv.TM_CCOEFF_NORMED)
            loc = cv.minMaxLoc(res)[3]
            ret_next[it] = [[loc[0]+windowSize[0]/2, loc[1]+windowSize[1]/2]]
            it += 1
        else:
            ret_prev = np.delete(ret_prev, it, axis=0)
            ret_next = np.delete(ret_next, it, axis=0)
        
        if it >= len(ret_prev):
            break
    
    return ret_prev, ret_next

def drawPoints(frame, pts, clr):
    ret_frame = frame.copy()
    
    for pt in pts:
        cv.circle(ret_frame, tuple(pt[0]), 5, clr, 2)
        
    return ret_frame

def removePts(mask, prevPts, nextPts):
    ret_prev = prevPts.copy()
    ret_next = nextPts.copy()
    
    it = len(mask)-1
    while it > 0:
        if mask[it] == 0:
            ret_prev = np.delete(ret_prev, it, axis=0)
            ret_next = np.delete(ret_next, it, axis=0)
        it = it - 1
    
    return ret_prev, ret_next

def getFrame(num_skip):
    ret = True
    last = False
    frame = None
    skipped = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            if skipped > 0:
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
    ret, last, next_frame, skipped = getFrame(skip_frames)
    
    if ret == True or last == True:
        # k = cv.waitKey(1) & 0xff
        # if k == 27:
        #     break
        
        next_frame = cv.resize(next_frame, (int(1920/resize_coeff), int(1080/resize_coeff)))
        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        prev_pts, next_pts = findNextPoints(prev_frame, next_frame, prev_pts, windowSize=(30, 30))
        
        ret, mask = cv.findFundamentalMat(prev_pts, next_pts, method=cv.FM_RANSAC)
        good_pts_prev, good_pts_next = removePts(mask, prev_pts, next_pts)
        
        disp_frame = cv.cvtColor(cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        disp_frame = drawOpticalFlow(disp_frame, good_pts_prev, good_pts_next)
        
        out_vid.write(disp_frame)
        # cv.imshow('viewer', disp_frame)
        
        prev_frame = next_frame
        prev_pts = next_pts
    else:
        break

print('Done')
cv.destroyAllWindows()
out_vid.release()
cam.release()
exit()
