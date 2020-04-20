import cv2 as cv
import os
import sys

os.chdir(sys.path[0])

scale = 0.25
target = cv.imread('resources/book_template.jpg')
target = cv.resize(target, (int(target.shape[1]*scale), int(target.shape[0]*scale)))
target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

frame = cv.imread('resources/multiple_books.jpg')
frame = cv.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()
bf = cv.BFMatcher.create(normType=cv.NORM_HAMMING, crossCheck=True)

tgt_pts, tgt_desc = orb.detectAndCompute(target, None)
frame_pts, frame_desc = orb.detectAndCompute(frame, None)
matches = bf.match(tgt_desc, frame_desc)
matches = sorted(matches, key=lambda x:x.distance)

disp_frame = cv.drawMatches(target, tgt_pts, frame, frame_pts, matches[:20], None, flags=2)
cv.imshow('match', disp_frame)

cv.imwrite('results/1-match.jpg', disp_frame)

cv.waitKey(0)
cv.destroyAllWindows()
exit()