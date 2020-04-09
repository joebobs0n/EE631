import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import sys

os.chdir(sys.path[0])

print('Starting task 2...')

pics = glob.glob('resources/images/*.jpg')

dist_per_frame = 15.25
frame_prev = cv.imread(pics[0])
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
frame_shape = np.flip(gray_prev.shape)
tmp_origin = [278, 100]
tmp_size = [90, 272]
template = frame_prev[tmp_origin[1]:tmp_origin[1] +
                      tmp_size[1], tmp_origin[0]:tmp_origin[0]+tmp_size[0]]
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
pts_prev = cv.goodFeaturesToTrack(
    template_gray, maxCorners=120, qualityLevel=0.01, minDistance=6)
for i in range(len(pts_prev)):
    pts_prev[i][0] = [pts_prev[i][0][0] +
                      tmp_origin[0], pts_prev[i][0][1]+tmp_origin[1]]
pts_prev = pts_prev.reshape((-1, 2))
pyr_level = 3


it = 1
d = []
while True:
    if it >= len(pics):
        break

    frame_next = cv.imread(pics[it])
    it += 1

    k = cv.waitKey(1) & 0xff
    if k == 27:
        break

    gray_next = cv.cvtColor(frame_next, cv.COLOR_BGR2GRAY)
    pts_next = cv.calcOpticalFlowPyrLK(
        gray_prev, gray_next, pts_prev, None, maxLevel=pyr_level, winSize=(20, 20))[0]

    a = []
    for i in range(len(pts_prev)):
        x_ = np.abs(pts_next[i][0] - frame_shape[0]/2)
        x = np.abs(pts_prev[i][0] - frame_shape[0]/2)
        if x != 0:
            a.append(x_ / x)
    a = np.mean(a)
    d_ = (a / (a - 1)) * dist_per_frame
    d.append(d_)

    gray_prev = gray_next.copy()
    pts_prev = pts_next.copy()

frame_nums = np.linspace(1, len(d), num=len(d)) * dist_per_frame
coeffs = np.polyfit(frame_nums, d, 1)
p = np.poly1d(coeffs)
x_fit = p(frame_nums)
x_guess = p(0)

fig = plt.figure(1)
plt.scatter(frame_nums, d)
plt.plot(frame_nums, x_fit, 'r')
plt.grid('on')
plt.xlabel('Distance (mm)')
plt.ylabel('Estimated Distance to Impact')
plt.title(f'Distance to Impact = {x_guess} mm')
plt.savefig('results/2-frame_estimate.png')
plt.close(1)

cv.destroyAllWindows()
print('Done.')
exit()
