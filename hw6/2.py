import cv2 as cv
import numpy as np
import os
import sys

os.chdir(sys.path[0])

coeff = 2
target = cv.imread('resources/angel.jpg')
tgt_shape = np.flip(np.shape(target)[0:2])
target = cv.resize(target, (int(tgt_shape[0]/coeff), int(tgt_shape[1]/coeff)))
replacement = cv.imread('resources/book_template.jpg')
replacement = cv.resize(replacement, (int(tgt_shape[0]/coeff), int(tgt_shape[1]/coeff)))
video = cv.VideoCapture('resources/angel.mp4')
orb = cv.ORB_create()
bf = cv.BFMatcher.create(normType=cv.NORM_HAMMING, crossCheck=True)

tgt_pts, tgt_desc = orb.detectAndCompute(target, None)

match_out = cv.VideoWriter(f'results/2-matches.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1963, 826))
add_out = cv.VideoWriter(f'results/2-superimposed.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (960, 540))


def readFrame():
	ret, frame = video.read()
	if ret == True:
		shape = np.flip(np.shape(frame)[0:2])
		frame = cv.resize(frame, (int(shape[0]/coeff), int(shape[1]/coeff)))
	return ret, frame


while True:
	k = cv.waitKey(1) & 0xff
	if k == 27:
		break
	
	ret, frame = readFrame()
	if ret is not True:
		break
	else:
		vid_pts, vid_desc = orb.detectAndCompute(frame, None)
		matches = bf.match(tgt_desc, vid_desc)
		matches = sorted(matches, key=lambda x:x.distance)

		pt0 = []
		pt1 = []
		for i in range(15):
			pt0.append(tgt_pts[matches[i].queryIdx].pt)
			pt1.append(vid_pts[matches[i].trainIdx].pt)

		pt0 = np.asarray(pt0).reshape(-1, 1, 2)
		pt1 = np.asarray(pt1).reshape(-1, 1, 2)
		h, status = cv.findHomography(pt0, pt1)
 
		warp_out = cv.warpPerspective(replacement, h, (replacement.shape[1], replacement.shape[0]))
		warp_crop = warp_out[:frame.shape[0], :frame.shape[1]]
		warp_mask = cv.cvtColor(cv.threshold(cv.cvtColor(warp_crop, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY_INV)[1], cv.COLOR_GRAY2BGR)

		match_frame = cv.drawMatches(target, tgt_pts, frame, vid_pts, matches[:20], None, flags=2)
		add_frame = frame & warp_mask
		add_frame = add_frame | warp_crop

		cv.imshow('match frame', match_frame)
		cv.imshow('add frame', add_frame)
  
		match_out.write(match_frame)
		add_out.write(add_frame)

match_out.release()
add_out.release()
cv.destroyAllWindows()
exit()
