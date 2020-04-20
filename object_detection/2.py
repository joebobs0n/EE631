import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

orb = cv.ORB_create()
bf = cv.BFMatcher.create(normType=cv.NORM_HAMMING, crossCheck=True)

scale = 0.25
target = cv.imread('resources/book_template.jpg')
target = cv.resize(target, (int(target.shape[1]*scale), int(target.shape[0]*scale)))
target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
tgt_pts, tgt_desc = orb.detectAndCompute(target, None)
mask = np.ones(target.shape) * 255

cam = cv.VideoCapture(1)
vid_out = cv.VideoWriter(f'results/2-object_detect.avi', cv.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))

while True:
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    
    ret, frame = cam.read()
    if ret == True:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_pts, frame_desc = orb.detectAndCompute(frame, None)
        
        matches = bf.match(tgt_desc, frame_desc)
        matches = sorted(matches, key=lambda x:x.distance)
        
        disp_matches = cv.drawMatches(target, tgt_pts, frame, frame_pts, matches[:50], None, flags=2)
        cv.imshow('matches', disp_matches)
        
        tgt_points = []
        frame_points = []
        
        ii = 50
        if len(tgt_pts) < ii:
            ii = len(tgt_pts)
        if len(frame_pts) < ii:
            if len(frame_pts) < len(tgt_pts):
                ii = len(frame_pts)
        for i in range(ii):
            tgt_points.append(tgt_pts[matches[i].queryIdx].pt)
            frame_points.append(frame_pts[matches[i].trainIdx].pt)
        
        tgt_points = np.asarray(tgt_points).reshape(-1, 1, 2)
        frame_points = np.asarray(frame_points).reshape(-1, 1, 2)
        
        homography = cv.findHomography(tgt_points, frame_points, method=cv.RANSAC)[0]
        
        warp_mask = cv.warpPerspective(mask, homography, tuple(np.flip(frame.shape)))
        thresh_mask = cv.threshold(warp_mask, 1, 255, cv.THRESH_BINARY)[1]
        
        disp_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        contours = cv.findContours(thresh_mask.astype(int), mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)[0]
        if len(contours) > 0:
            fill_gaps = cv.drawContours(disp_frame, contours, 0, (0, 255, 0), 2)
        
        cv.imshow('contour', disp_frame)
        vid_out.write(disp_frame)
    
cv.destroyAllWindows()
cam.release()
vid_out.release()
exit()
        