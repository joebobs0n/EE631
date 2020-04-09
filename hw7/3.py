import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import sys

os.chdir(sys.path[0])

print('Starting task 3...')

pics = glob.glob('resources/images/*.jpg')

f = open('resources/cam_params.txt')
mat = []
dist = []
for line in f:
    data = [float(x) for x in line.split()]
    if len(data) == 1:
        dist.append(data)
    elif len(data) == 3:
        mat.append(data)
mat = np.asarray(mat).reshape((3, 3))
dist = np.asarray(dist).reshape((5, 1))

pixel_size = 7.4e-3
fsx = mat[0][0]
can_w = 59

frame_prev = cv.imread(pics[0])
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
pts = np.array([[282, 225], [366, 225]], dtype=np.float32)

d = []
for pic in pics:
    frame = cv.imread(pic)
    undistort = cv.undistort(frame, mat, dist)
    gray = cv.cvtColor(undistort, cv.COLOR_BGR2GRAY)
    new_pts = cv.calcOpticalFlowPyrLK(
        gray_prev, gray, pts, None, maxLevel=3, winSize=(20, 20))[0]

    pixel_w = np.abs(new_pts[0][0] - new_pts[1][0])
    d_ = can_w * fsx / pixel_w
    print(f'w={pixel_w} | d={d_}')
    d.append(d_)
    
frame_nums = np.linspace(1, len(pics), num=len(pics))
coeffs = np.polyfit(frame_nums, d, 1)
p = np.poly1d(coeffs)
x_fit = p(frame_nums)
x_guess = p(0)

fig = plt.figure(1)
plt.scatter(frame_nums, d)
plt.plot(frame_nums, x_fit, 'r')
plt.grid('on')
plt.xlabel('Frame Number')
plt.ylabel('Estimated Distance to Impact (mm)')
plt.title(f'Distance to Impact = {x_guess} mm')
plt.savefig('results/3-frame_estimate.png')
plt.close(1)

print('Done.')
exit()
